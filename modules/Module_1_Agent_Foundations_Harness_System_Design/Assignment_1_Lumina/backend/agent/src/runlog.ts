/**
 * Run logs and request records, and the /stats computed from them.
 *
 * Every answer writes two documents, keyed by requestId:
 *   runs      the quality kit's RunLog shape (what `npm run export:runs` dumps to runs/),
 *             plus what makes it queryable.
 *   requests  one row per answer with its status, tokens, cost and timings, which /stats reads.
 *
 * Both live in Mongo rather than on disk because Fly wipes a machine's disk on redeploy.
 * /stats is computed from `requests` on every call; there are no running counters to drift
 * away from the logs.
 */
import { COLLECTIONS, RequestDoc, RunDoc, StatsResponse, type Depth, type RunLog } from '@lumina/contract';
import { db } from './db.js';
import { deepToday } from './deepcap.js';
import { env } from './env.js';

export type AnswerRecord = {
  requestId: string;
  userId: string;
  threadId: string;
  answerId: string;
  query: string;
  route: string;
  /** The status the answer ended with: 200 for done or cap, else the error's status. */
  status: number;
  run: RunLog;
  tokensIn: number;
  tokensOut: number;
  ttftMs: number;
  searchCached: boolean;
};

/**
 * Written before `done` goes out, so a client that has seen `done` can already see the
 * answer in /stats. Upserts on requestId: a reused X-Request-Id replaces its earlier run in
 * both collections, which keeps runs and requests one to one.
 */
export async function recordAnswer(a: AnswerRecord): Promise<void> {
  const createdAt = new Date();
  const run = RunDoc.parse({
    ...a.run,
    requestId: a.requestId,
    userId: a.userId,
    threadId: a.threadId,
    answerId: a.answerId,
    query: a.query,
    createdAt
  });
  const request = {
    ...RequestDoc.parse({
      requestId: a.requestId,
      userId: a.userId,
      route: a.route,
      status: a.status,
      ms: Math.round(a.run.wallClockSec * 1000),
      tokensIn: a.tokensIn,
      tokensOut: a.tokensOut,
      costUsd: a.run.costUsd,
      toolCalls: a.run.toolCalls.length,
      terminated: a.run.terminated,
      depth: a.run.depth,
      createdAt
    }),
    // Beyond the contract's key fields: what /stats needs for its latency and cache numbers.
    ttftMs: a.ttftMs,
    searchCached: a.searchCached
  };
  const d = await db();
  await Promise.all([
    d.collection(COLLECTIONS.runs).replaceOne({ requestId: a.requestId }, run, { upsert: true }),
    d.collection(COLLECTIONS.requests).replaceOne({ requestId: a.requestId }, request, { upsert: true })
  ]);
}

/** Nearest-rank, the same definition benchmark/lib.mjs uses, so the two p95s agree. */
function p95(values: number[]): number {
  const xs = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!xs.length) return 0;
  const rank = Math.ceil(0.95 * xs.length);
  return xs[Math.min(xs.length - 1, Math.max(0, rank - 1))]!;
}

const utcMidnight = (now = new Date()) => new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));

type Row = { terminated?: string; costUsd?: number; ttftMs?: number; searchCached?: boolean; depth?: Depth; createdAt: Date };

/**
 * Counts cover every stored record; cost and deep searches cover the current UTC day.
 * An answer is a run that sent `done` (terminated done or cap), the same thing the bench
 * counts. deepToday is this user's; everything else is service-wide.
 */
export async function computeStats(userId: string): Promise<StatsResponse> {
  const since = utcMidnight();
  const rows = await (await db())
    .collection<Row>(COLLECTIONS.requests)
    .find({}, { projection: { _id: 0, terminated: 1, costUsd: 1, ttftMs: 1, searchCached: 1, depth: 1, createdAt: 1 } })
    .toArray();

  const answers = rows.filter((r) => r.terminated === 'done' || r.terminated === 'cap');
  const today = rows.filter((r) => r.createdAt >= since);
  const costUsdToday = today.reduce((sum, r) => sum + (r.costUsd ?? 0), 0);

  return StatsResponse.parse({
    requests: rows.length,
    answers: answers.length,
    searchCacheHitRatePct: answers.length
      ? Math.round((answers.filter((r) => r.searchCached).length / answers.length) * 1000) / 10
      : 0,
    ttftP95Ms: p95(answers.map((r) => r.ttftMs ?? NaN)),
    costUsdToday: Math.round(costUsdToday * 1e6) / 1e6,
    // The gate's own counter, not the records: a slot is spent when a run starts, and a run
    // that errored before its record was written still spent one.
    deepToday: await deepToday(userId),
    deepDailyCap: env.deepDailyCap
  });
}
