/**
 * The jobs worker: its own process (`npm run worker`, started alongside the agent by
 * `npm run dev`), from the same codebase as the agent service. Parsing a 60-page PDF is CPU
 * work, and here it can never stall somebody's answer stream (DESIGN.md, Components).
 *
 * index_document → GridFS read → parse (page / heading / line locators) → chunk → embed in
 *                  batches → upsert into chunks → READ-YOUR-WRITE PROBE → status: 'indexed'
 *
 * Robustness, as DESIGN.md (Communication) commits to it:
 *   - The claim is one atomic findOneAndUpdate, so two workers never take the same job.
 *   - A claimed job carries a heartbeat. A job silent for JOB_STALE_MS (2 min) is stale, and
 *     the sweeper returns it to `pending`; a slow PDF that is still beating is never taken twice.
 *   - Progress is saved after each stage and after each embedding batch, so a reclaimed job
 *     resumes after the last thing it finished. Chunk ids are `<docId>:<ord>`, so a retry
 *     overwrites the chunks it already wrote instead of duplicating them.
 *   - JOB_MAX_ATTEMPTS (3) attempts, crashes included; then the document is `failed` with an
 *     error and every chunk it wrote is deleted, so nothing from it can ever be cited.
 *   - "Upserted" is not "searchable": `indexed` only after $vectorSearch returns the last chunk
 *     written. Atlas Search indexes are eventually consistent.
 *   - The process writes its own heartbeat too, which /health reads to show a dead worker.
 */
import { MongoError, type AnyBulkWriteOperation } from 'mongodb';
import { hostname } from 'node:os';
import { randomUUID } from 'node:crypto';
import { setTimeout as sleep } from 'node:timers/promises';
import pino from 'pino';
import { COLLECTIONS, SEARCH_INDEXES, type ChunkDoc } from '@lumina/contract';
import { db } from './db.js';
import { embed } from './embeddings.js';
import { env } from './env.js';
import { chunkSegments, parseDocument, type Piece } from './ingest.js';
import { jobs, workers, type JobProgress, type StoredJob } from './jobs.js';
import { deleteFiles, documents, putFile, readFile, type StoredDocument } from './spaces.js';

const log = pino({ level: env.logLevel }).child({ component: 'worker' });
const workerId = `${hostname()}:${process.pid}:${randomUUID().slice(0, 6)}`;

type StoredChunk = Omit<ChunkDoc, 'createdAt'> & { createdAt: Date };
const chunks = async () => (await db()).collection<StoredChunk>(COLLECTIONS.chunks);

/** Deterministic, so a retried batch replaces what an earlier attempt wrote. */
const chunkId = (docId: string, ord: number) => `${docId}:${ord}`;

/** pct milestones. Embedding fills the space between parsed and probing. */
const PCT = { parsing: 5, parsed: 15, embedded: 95, probing: 97 } as const;

const SWEEP_EVERY_MS = 30_000;
const RETRY_BASE_MS = 5_000;
const PROBE_INTERVAL_MS = 1_500;
const LOOP_ERROR_BACKOFF_MS = 5_000;

/** This worker no longer owns the job: the sweeper judged it stale and someone else may have it. */
class LostClaim extends Error {}
class ShuttingDown extends Error {}

// ---------------------------------------------------------------- one job

type Stage = 'parsing' | 'embedding' | 'probing';

/**
 * Runs the job from wherever its last attempt got to. Throws on any failure; the caller
 * decides between a retry and `failed`.
 */
async function indexDocument(job: StoredJob, at: { stage: Stage }, signal: AbortSignal): Promise<void> {
  const { docId, spaceId, fileId, mimeType } = job.payload;
  const { userId } = job;
  const [docs, queue, chunkCol] = await Promise.all([documents(), jobs(), chunks()]);
  let progress: JobProgress = job.progress ?? {};

  const setDoc = async (set: Partial<StoredDocument>) => {
    signal.throwIfAborted();
    const r = await docs.updateOne({ _id: docId }, { $set: set });
    if (!r.matchedCount) throw new Error(`document ${docId} no longer exists`);
  };
  const saveProgress = async (next: JobProgress) => {
    signal.throwIfAborted();
    progress = { ...progress, ...next };
    const r = await queue.updateOne({ _id: job._id, workerId, status: 'running' }, { $set: { progress } });
    if (!r.matchedCount) throw new LostClaim();
  };

  // 1 · parse and chunk, unless an earlier attempt already saved the result
  if (!progress.parsedFileId) {
    at.stage = 'parsing';
    await setDoc({ status: 'parsing', pct: PCT.parsing });
    const parsed = await parseDocument(await readFile(fileId), mimeType);
    const pieces = chunkSegments(parsed.segments, env.chunkSizeChars, env.chunkOverlapChars);
    if (!pieces.length) throw new Error('no extractable text in this file (a scanned PDF needs OCR first)');
    // The chunks go to GridFS, not the job row: a 25 MB text file is past a document's 16 MB limit.
    await deleteFiles({ kind: 'parsed', docId }); // left by an attempt that died before saving progress
    const parsedFileId = await putFile(Buffer.from(JSON.stringify(pieces)), `${docId}.chunks.json`, { kind: 'parsed', docId });
    await saveProgress({ parsedFileId, chunkCount: pieces.length, embedded: 0, ...(parsed.pages ? { pages: parsed.pages } : {}) });
    await setDoc({ pct: PCT.parsed, ...(parsed.pages ? { pages: parsed.pages } : {}) });
    log.info({ jobId: job._id, docId, pages: parsed.pages, chunks: pieces.length }, 'parsed');
  }

  // 2 · embed and upsert, batch by batch, starting after the last batch already written
  at.stage = 'embedding';
  const pieces = JSON.parse((await readFile(progress.parsedFileId!)).toString('utf8')) as Piece[];
  await setDoc({ status: 'embedding' });
  for (let from = progress.embedded ?? 0; from < pieces.length; from += env.embedBatchSize) {
    const batch = pieces.slice(from, from + env.embedBatchSize);
    const vectors = await embed(batch.map((p) => p.text), signal);
    const ops: AnyBulkWriteOperation<StoredChunk>[] = batch.map((p, i) => ({
      replaceOne: {
        filter: { _id: chunkId(docId, p.ord) },
        replacement: { docId, spaceId, userId, text: p.text, locator: p.locator, ord: p.ord, embedding: vectors[i]!, createdAt: new Date() },
        upsert: true
      }
    }));
    signal.throwIfAborted();
    await chunkCol.bulkWrite(ops, { ordered: false });
    const embedded = from + batch.length;
    await saveProgress({ embedded });
    await setDoc({ pct: Math.round(PCT.parsed + ((PCT.embedded - PCT.parsed) * embedded) / pieces.length) });
  }
  // A chunk past the end can only be left from a different parse of this document.
  await chunkCol.deleteMany({ docId, ord: { $gte: pieces.length } });

  // 3 · the read-your-write probe: the last chunk written must come back from the vector index
  at.stage = 'probing';
  await setDoc({ pct: PCT.probing });
  const tries = await probe(spaceId, userId, chunkId(docId, pieces.length - 1), signal);

  signal.throwIfAborted();
  await docs.updateOne(
    { _id: docId },
    { $set: { status: 'indexed', pct: 100, chunks: pieces.length, ...(progress.pages ? { pages: progress.pages } : {}) }, $unset: { error: '' } }
  );
  log.info({ jobId: job._id, docId, chunks: pieces.length, probeTries: tries }, 'probe found the chunk: indexed');
}

/** Polls $vectorSearch, filtered by Space and user inside the stage, until the chunk comes back. */
async function probe(spaceId: string, userId: string, id: string, signal: AbortSignal): Promise<number> {
  const col = await chunks();
  const chunk = await col.findOne({ _id: id }, { projection: { embedding: 1 } });
  if (!chunk) throw new Error(`probe: chunk ${id} is not in the collection`);
  // The local fallback scans the collection itself, which is read-your-write by construction.
  if (env.vectorBackend === 'mongo-cosine-scan') return 1;

  const deadline = Date.now() + env.probeTimeoutMs;
  for (let tries = 1; ; tries++) {
    signal.throwIfAborted();
    const hits = await col
      .aggregate<{ _id: string }>([
        {
          $vectorSearch: {
            index: SEARCH_INDEXES.chunksVector,
            path: 'embedding',
            queryVector: chunk.embedding,
            numCandidates: 100,
            limit: 10,
            filter: { $and: [{ spaceId: { $eq: spaceId } }, { userId: { $eq: userId } }] }
          }
        },
        { $project: { _id: 1 } }
      ])
      .toArray();
    if (hits.some((h) => h._id === id)) return tries;
    if (Date.now() > deadline) {
      throw new Error(`read-your-write probe: ${SEARCH_INDEXES.chunksVector} did not return chunk ${id} within ${env.probeTimeoutMs / 1000}s`);
    }
    await sleep(PROBE_INTERVAL_MS, undefined, { signal });
  }
}

// ---------------------------------------------------------------- outcomes

/** What the UI shows. A driver message can name the cluster host, so that one stays in the log. */
const publicError = (err: unknown) =>
  err instanceof MongoError ? 'database error (see the worker log)' : String((err as Error)?.message ?? err).slice(0, 300);

/** Retry with backoff while attempts remain; after the last one, fail the document. */
async function settleFailure(job: StoredJob, stage: Stage, err: unknown): Promise<void> {
  const error = `${stage}: ${publicError(err)}`;
  if (job.attempts < env.jobMaxAttempts) {
    const retryInMs = RETRY_BASE_MS * job.attempts;
    await (await jobs()).updateOne(
      { _id: job._id, workerId },
      {
        $set: { status: 'pending', error, runAfter: new Date(Date.now() + retryInMs) },
        $unset: { workerId: '', claimedAt: '', heartbeatAt: '' }
      }
    );
    log.warn({ jobId: job._id, docId: job.payload.docId, attempt: job.attempts, retryInMs, err }, `attempt failed while ${stage}; will retry`);
    return;
  }
  await failJob(job, `${error} (gave up after ${job.attempts} attempts)`);
}

/**
 * Terminal. Chunks go first: if this dies halfway, the job is still `running`, the sweeper
 * finds it out of attempts, and comes back here.
 */
async function failJob(job: StoredJob, error: string): Promise<void> {
  const { docId } = job.payload;
  const { deletedCount } = await (await chunks()).deleteMany({ docId });
  await (await documents()).updateOne({ _id: docId }, { $set: { status: 'failed', error } });
  await deleteFiles({ kind: 'parsed', docId });
  await (await jobs()).updateOne(
    { _id: job._id },
    { $set: { status: 'failed', error, finishedAt: new Date() }, $unset: { workerId: '', heartbeatAt: '' } }
  );
  log.error({ jobId: job._id, docId, error, chunksDeleted: deletedCount }, 'document failed');
}

// ---------------------------------------------------------------- claim, heartbeat, sweep

let current: { job: StoredJob; ctl: AbortController } | null = null;
const shutdown = new AbortController();

async function claim(): Promise<StoredJob | null> {
  const now = new Date();
  return (await jobs()).findOneAndUpdate(
    { status: 'pending', $or: [{ runAfter: { $exists: false } }, { runAfter: { $lte: now } }] },
    { $set: { status: 'running', claimedAt: now, heartbeatAt: now, workerId }, $inc: { attempts: 1 } },
    { sort: { createdAt: 1 }, returnDocument: 'after' }
  );
}

/** The job's heartbeat. If the row is no longer ours, stop working on it at once. */
async function beatJob(job: StoredJob, ctl: AbortController): Promise<void> {
  try {
    const r = await (await jobs()).updateOne({ _id: job._id, workerId, status: 'running' }, { $set: { heartbeatAt: new Date() } });
    if (!r.matchedCount) ctl.abort(new LostClaim());
  } catch (err) {
    log.error({ jobId: job._id, err }, 'job heartbeat failed');
  }
}

/** The process heartbeat /health reads. */
async function beatWorker(): Promise<void> {
  try {
    await (await workers()).updateOne(
      { _id: workerId },
      { $set: { heartbeatAt: new Date(), pid: process.pid, host: hostname(), jobId: current?.job._id ?? null }, $setOnInsert: { startedAt: new Date() } },
      { upsert: true }
    );
  } catch (err) {
    log.error({ err }, 'worker heartbeat failed');
  }
}

/**
 * Stale `running` jobs: out of attempts → failed; otherwise → pending, to resume from their
 * saved progress. Each is taken with its own atomic update, so two sweepers never both act.
 */
async function sweep(): Promise<void> {
  const queue = await jobs();
  const stale = () => ({ status: 'running' as const, heartbeatAt: { $lt: new Date(Date.now() - env.jobStaleMs) } });

  for (;;) {
    const dead = await queue.findOneAndUpdate(
      { ...stale(), attempts: { $gte: env.jobMaxAttempts } },
      { $set: { workerId, heartbeatAt: new Date() } },
      { returnDocument: 'after' }
    );
    if (!dead) break;
    await failJob(dead, `worker stopped responding (no heartbeat for ${env.jobStaleMs / 1000}s) on all ${dead.attempts} attempts`);
  }
  for (;;) {
    const job = await queue.findOneAndUpdate(
      { ...stale(), attempts: { $lt: env.jobMaxAttempts } },
      {
        $set: { status: 'pending', error: `worker stopped responding (no heartbeat for ${env.jobStaleMs / 1000}s)` },
        $unset: { workerId: '', claimedAt: '', heartbeatAt: '', runAfter: '' }
      },
      { returnDocument: 'after' }
    );
    if (!job) break;
    log.warn({ jobId: job._id, docId: job.payload.docId, attempts: job.attempts, progress: job.progress }, 'stale job returned to pending');
  }
}

async function runJob(job: StoredJob): Promise<void> {
  const ctl = new AbortController();
  current = { job, ctl };
  const timer = setInterval(() => void beatJob(job, ctl), env.workerHeartbeatMs);
  const at: { stage: Stage } = { stage: 'parsing' };
  const started = Date.now();
  log.info({ jobId: job._id, docId: job.payload.docId, title: job.payload.title, attempt: job.attempts, resumeFrom: job.progress }, 'claimed');

  try {
    await indexDocument(job, at, ctl.signal);
    await (await jobs()).updateOne(
      { _id: job._id, workerId },
      { $set: { status: 'done', finishedAt: new Date() }, $unset: { error: '', runAfter: '', heartbeatAt: '' } }
    );
    // The intermediate chunk list is only for resuming; the raw upload stays.
    await deleteFiles({ kind: 'parsed', docId: job.payload.docId }).catch((err: unknown) =>
      log.error({ jobId: job._id, err }, 'could not delete the parsed chunk list')
    );
    log.info({ jobId: job._id, docId: job.payload.docId, ms: Date.now() - started }, 'done');
  } catch (err) {
    const reason = err instanceof LostClaim ? err : ctl.signal.aborted ? ctl.signal.reason : err;
    if (reason instanceof LostClaim) {
      log.warn({ jobId: job._id }, 'lost the claim (judged stale); another worker has the job');
    } else if (reason instanceof ShuttingDown) {
      // Not the job's fault: hand it back now, without spending an attempt, instead of waiting out the stale window.
      await (await jobs()).updateOne(
        { _id: job._id, workerId },
        { $set: { status: 'pending' }, $inc: { attempts: -1 }, $unset: { workerId: '', claimedAt: '', heartbeatAt: '' } }
      );
      log.info({ jobId: job._id, stage: at.stage }, 'released the job on shutdown');
    } else {
      await settleFailure(job, at.stage, err);
    }
  } finally {
    clearInterval(timer);
    current = null;
  }
}

// ---------------------------------------------------------------- the loop

async function main(): Promise<void> {
  log.info(
    {
      workerId,
      vectorStore: env.vectorBackend,
      chunk: { sizeChars: env.chunkSizeChars, overlapChars: env.chunkOverlapChars },
      embedBatch: env.embedBatchSize,
      staleAfterMs: env.jobStaleMs,
      maxAttempts: env.jobMaxAttempts
    },
    'jobs worker up'
  );
  await beatWorker();
  const beat = setInterval(() => void beatWorker(), env.workerHeartbeatMs);
  let lastSweep = 0;

  while (!shutdown.signal.aborted) {
    try {
      if (Date.now() - lastSweep >= SWEEP_EVERY_MS) {
        await sweep();
        lastSweep = Date.now();
      }
      const job = await claim();
      if (job) {
        await beatWorker();
        await runJob(job);
      } else {
        await sleep(env.workerPollMs, undefined, { signal: shutdown.signal }).catch(() => {});
      }
    } catch (err) {
      // Mongo unreachable, most likely. Keep the process up; the stopped heartbeat tells /health.
      log.error({ err }, 'worker loop error');
      await sleep(LOOP_ERROR_BACKOFF_MS, undefined, { signal: shutdown.signal }).catch(() => {});
    }
  }
  clearInterval(beat);
  log.info({ workerId }, 'jobs worker stopped');
  process.exit(0);
}

for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, () => {
    if (shutdown.signal.aborted) process.exit(1); // a second signal means now
    log.info({ signal: sig }, 'stopping');
    current?.ctl.abort(new ShuttingDown());
    shutdown.abort();
  });
}

void main();
