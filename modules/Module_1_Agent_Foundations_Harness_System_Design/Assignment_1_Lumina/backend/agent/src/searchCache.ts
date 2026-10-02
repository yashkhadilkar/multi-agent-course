/**
 * The search cache: an in-process LRU in front of the `searchCache` Mongo collection, keyed
 * by sha256(normalized query + provider). Two tiers because the LRU is free and the Mongo
 * tier survives a restart and is shared across processes; the TTL index on `expiresAt`
 * (scripts/indexes.json) is what actually expires a row.
 *
 * A read of either tier failing is a cache miss, never a failed run: the search cache is an
 * optimization, and a broken cache must not take the answer down with it (DESIGN.md).
 */
import { createHash } from 'node:crypto';
import pino from 'pino';
import { COLLECTIONS, type SearchCacheDoc } from '@lumina/contract';
import { env } from './env.js';
import { db } from './db.js';
import { webSearch, type SearchResult } from './search.js';

const log = pino({ level: env.logLevel });

/** A query a cache would serve stale: "today", "latest", or a year that has arrived. */
const CURRENT_YEAR = new Date().getFullYear();
export function isTimeSensitive(query: string): boolean {
  if (/\b(today|latest|now|current(ly)?)\b/i.test(query)) return true;
  const years = query.match(/\b(20\d{2})\b/g);
  return years !== null && years.some((y) => Number(y) >= CURRENT_YEAR);
}

function normalize(query: string): string {
  return query.trim().toLowerCase().replace(/\s+/g, ' ');
}

function cacheKey(query: string, provider: string): string {
  return createHash('sha256').update(`${normalize(query)}:${provider}`).digest('hex');
}

// ---------------------------------------------------------------- in-process LRU

/** Small enough that a single process's memory never has to think about it. */
const LRU_MAX_ENTRIES = 500;

class Lru<V> {
  private readonly map = new Map<string, V>();
  constructor(private readonly max: number) {}

  get(key: string): V | undefined {
    const v = this.map.get(key);
    if (v === undefined) return undefined;
    // Re-insert so the key becomes the most recently used (Map iterates in insertion order).
    this.map.delete(key);
    this.map.set(key, v);
    return v;
  }

  set(key: string, value: V): void {
    this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.max) {
      const oldest = this.map.keys().next().value;
      if (oldest !== undefined) this.map.delete(oldest);
    }
  }
}

type Entry = { results: SearchResult[]; expiresAt: number };
const lru = new Lru<Entry>(LRU_MAX_ENTRIES);

// ---------------------------------------------------------------- the cached search

export type CachedSearch = { results: SearchResult[]; hit: boolean };

/**
 * Same contract as `webSearch`: failures throw. A cache lookup never turns a down provider
 * into a quiet empty result — it only ever saves a call that would have succeeded anyway.
 */
export async function cachedWebSearch(query: string, signal: AbortSignal): Promise<CachedSearch> {
  if (isTimeSensitive(query)) {
    return { results: await webSearch(query, signal), hit: false };
  }

  const key = cacheKey(query, env.searchProvider);
  const now = Date.now();

  const fromLru = lru.get(key);
  if (fromLru && fromLru.expiresAt > now) return { results: fromLru.results, hit: true };

  try {
    const collection = (await db()).collection<SearchCacheDoc>(COLLECTIONS.searchCache);
    const doc = await collection.findOne({ _id: key });
    if (doc && new Date(doc.expiresAt).getTime() > now) {
      const results = doc.results as unknown as SearchResult[];
      lru.set(key, { results, expiresAt: new Date(doc.expiresAt).getTime() });
      return { results, hit: true };
    }
  } catch (err) {
    log.warn({ err, key }, 'search cache read failed; treating as a miss');
  }

  const results = await webSearch(query, signal);
  const expiresAt = new Date(now + env.searchCacheTtlSeconds * 1000);
  lru.set(key, { results, expiresAt: expiresAt.getTime() });

  try {
    const collection = (await db()).collection<SearchCacheDoc>(COLLECTIONS.searchCache);
    await collection.updateOne(
      { _id: key },
      { $set: { _id: key, provider: env.searchProvider, query: normalize(query), results, expiresAt, createdAt: new Date() } },
      { upsert: true }
    );
  } catch (err) {
    // The answer already has its results; a failed write only costs the next caller a hit.
    log.warn({ err, key }, 'search cache write failed; continuing without it');
  }

  return { results, hit: false };
}
