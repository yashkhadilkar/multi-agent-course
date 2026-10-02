/**
 * The deep-search spend gate: DEEP_DAILY_CAP deep searches per X-User-Id per UTC day.
 *
 * One row per user per day in `deepUsage`, `_id` = `<userId>|<YYYY-MM-DD>`. Checking the cap
 * and taking a slot are one findOneAndUpdate whose filter only matches while the count is
 * under the cap (DESIGN.md, State), so two requests arriving at four of five cannot both get
 * through. A slot is taken before anything streams and is never given back: a deep run that
 * later errors or caps still spent money (DESIGN.md, Trade-offs).
 *
 * /stats reads deepToday from this same row, so the number a user sees is the number the
 * gate enforces.
 */
import { MongoServerError } from 'mongodb';
import { db } from './db.js';
import { env } from './env.js';

const COLLECTION = 'deepUsage';

type UsageRow = { _id: string; userId: string; day: string; count: number; expiresAt: Date };

const dayOf = (now: Date) => now.toISOString().slice(0, 10);

/** The next UTC midnight: when today's slots come back, and the 429's resetsAt. */
export function nextUtcMidnight(now = new Date()): Date {
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1));
}

let ttlIndex: Promise<unknown> | null = null;

async function usage() {
  const col = (await db()).collection<UsageRow>(COLLECTION);
  // Yesterday's rows are dead weight once the day is over; a TTL index clears them.
  ttlIndex ??= col.createIndex({ expiresAt: 1 }, { expireAfterSeconds: 0 }).catch((err: unknown) => {
    ttlIndex = null;
    throw err;
  });
  await ttlIndex;
  return col;
}

export type DeepSlot = { ok: true; used: number } | { ok: false; used: number; resetsAt: Date };

/**
 * Takes one of today's deep slots, or refuses. The increment only matches a row under the
 * cap; when no row matches, the upsert tries to insert today's row, and a duplicate key
 * means the row exists and is full. Two first requests of the day racing on the insert
 * both see that duplicate once, so the loser retries and increments the winner's row.
 */
export async function takeDeepSlot(userId: string, now = new Date()): Promise<DeepSlot> {
  const day = dayOf(now);
  const resetsAt = nextUtcMidnight(now);
  if (env.deepDailyCap <= 0) return { ok: false, used: 0, resetsAt };
  const col = await usage();
  const _id = `${userId}|${day}`;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const row = await col.findOneAndUpdate(
        { _id, count: { $lt: env.deepDailyCap } },
        // A day later than the reset, so the TTL monitor never races today's reads.
        { $inc: { count: 1 }, $setOnInsert: { userId, day, expiresAt: new Date(resetsAt.getTime() + 86_400_000) } },
        { upsert: true, returnDocument: 'after' }
      );
      return { ok: true, used: row?.count ?? 1 };
    } catch (err) {
      if (!(err instanceof MongoServerError && err.code === 11000)) throw err;
      const row = await col.findOne({ _id });
      if (row && row.count >= env.deepDailyCap) return { ok: false, used: row.count, resetsAt };
    }
  }
  throw new Error('deep cap: could not take a slot after a concurrent insert');
}

/** Deep searches this user has started today, from the row the gate enforces. */
export async function deepToday(userId: string, now = new Date()): Promise<number> {
  const row = await (await usage()).findOne({ _id: `${userId}|${dayOf(now)}` }, { projection: { count: 1 } });
  return row?.count ?? 0;
}
