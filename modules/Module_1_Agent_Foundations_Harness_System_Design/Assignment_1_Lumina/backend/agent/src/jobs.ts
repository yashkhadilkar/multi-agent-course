/**
 * The `jobs` queue and the worker's heartbeat: the only things the agent service and the jobs
 * worker share. They never call each other (DESIGN.md, Communication). The API inserts a job
 * and returns 202; the worker claims it, and /health reads the worker's heartbeat.
 *
 * Two heartbeats, two questions:
 *   jobs.heartbeatAt     is the worker on THIS job still alive? Stale after JOB_STALE_MS, the
 *                        sweeper puts the job back to pending and it resumes where it stopped.
 *   workers.heartbeatAt  is ANY worker alive? /health goes degraded when the newest one is
 *                        older than the same window, so a dead worker shows up instead of
 *                        documents quietly freezing at `pending`.
 */
import { COLLECTIONS, JobDoc } from '@lumina/contract';
import { db } from './db.js';
import { env } from './env.js';
import type { AcceptedType } from './ingest.js';

export type IndexPayload = { docId: string; spaceId: string; fileId: string; mimeType: AcceptedType; title: string };

/** What an attempt finished, so the next one starts after it instead of from scratch. */
export type JobProgress = { parsedFileId?: string; chunkCount?: number; pages?: number; embedded?: number };

export type StoredJob = Omit<JobDoc, 'payload' | 'createdAt' | 'claimedAt'> & {
  payload: IndexPayload;
  createdAt: Date;
  claimedAt?: Date;
  heartbeatAt?: Date;
  /** Retry backoff: a failed attempt is not claimable again before this. */
  runAfter?: Date;
  finishedAt?: Date;
  progress?: JobProgress;
};

export type WorkerBeat = { _id: string; heartbeatAt: Date; startedAt: Date; pid: number; host: string; jobId: string | null };

/** Not a contract collection: operational state, one row per worker process. */
export const WORKERS_COLLECTION = 'workers';

export const jobs = async () => (await db()).collection<StoredJob>(COLLECTIONS.jobs);
export const workers = async () => (await db()).collection<WorkerBeat>(WORKERS_COLLECTION);

/** One job per document, so its id is derived from the document's. */
export function indexJob(userId: string, payload: IndexPayload): StoredJob {
  const job: StoredJob = {
    _id: `job_${payload.docId}`,
    kind: 'index_document',
    status: 'pending',
    payload,
    userId,
    attempts: 0,
    createdAt: new Date()
  };
  JobDoc.parse(job);
  return job;
}

export type WorkerHealth = { status: 'ok' | 'stale'; lastHeartbeat: string | null };

/** The newest heartbeat any worker wrote. None at all, or one older than JOB_STALE_MS, is stale. */
export async function workerHealth(): Promise<WorkerHealth> {
  const newest = await (await workers()).find({}, { projection: { heartbeatAt: 1 } }).sort({ heartbeatAt: -1 }).limit(1).next();
  if (!newest) return { status: 'stale', lastHeartbeat: null };
  const age = Date.now() - newest.heartbeatAt.getTime();
  return { status: age <= env.jobStaleMs ? 'ok' : 'stale', lastHeartbeat: newest.heartbeatAt.toISOString() };
}
