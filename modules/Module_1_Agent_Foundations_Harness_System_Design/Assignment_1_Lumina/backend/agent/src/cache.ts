/**
 * The search cache: an in-process LRU in front of the searchCache collection, whose TTL
 * index on expiresAt expires rows. Keyed by sha256 of (normalized query, provider), so a
 * Tavily result is never served to a SerpApi run. Shared across users on purpose: a web
 * search result is the same whoever asked for it.
 *
 * Only search results are cached, never answers or page reads.
 *
 * A cache read that fails is a miss, and it is logged at error level every time: a broken
 * cache must show up in the log, not hide behind a low hit rate. A write that fails is
 * logged too; the search that produced it already succeeded, so the run carries on.
 */
import { createHash } from 'node:crypto';
import type { Logger } from 'pino';
import { COLLECTIONS, type SearchCacheDoc } from '@lumina/contract';
import { db } from './db.js';
import { env } from './env.js';
import { webSearch, type SearchResult } from './search.js';

/** Entries kept in process. Five results of a few hundred characters each: small. */
const LRU_MAX = 1000;
/** A cache read is on the path to the first token. Slower than this is a failed read. */
const READ_TIMEOUT_MS = 1000;

export type CachedSearch = {
  results: SearchResult[];
  /** `lru` and `mongo` are hits; `miss` and `bypass` went to the provider. */
  from: 'lru' | 'mongo' | 'miss' | 'bypass';
  /** Set when the cache read failed and the search went to the provider instead. */
  cacheError?: string;
};

export function normalizeQuery(query: string): string {
  return query
    .normalize('NFKC')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/[\s?!.,;:]+$/, '');
}

export function cacheKey(query: string, provider: string = env.searchProvider): string {
  return createHash('sha256').update(JSON.stringify([normalizeQuery(query), provider])).digest('hex');
}

/**
 * SPEC §5.2: a query that says "today" or "latest", or names a year at or after the
 * current one, wants fresh results, so it neither reads nor writes the cache.
 */
export function isTimeSensitive(query: string): boolean {
  if (/\b(today|latest)\b/i.test(query)) return true;
  const year = new Date().getUTCFullYear();
  return [...query.matchAll(/\b(\d{4})\b/g)].some((m) => Number(m[1]) >= year);
}

// ---------------------------------------------------------------- tier 1: in process

type Entry = { results: SearchResult[]; expiresAt: number };

/** A Map iterates in insertion order, so re-inserting on read makes the first key the least recent. */
const lru = new Map<string, Entry>();

function lruGet(key: string): SearchResult[] | null {
  const e = lru.get(key);
  if (!e) return null;
  lru.delete(key);
  if (e.expiresAt <= Date.now()) return null;
  lru.set(key, e);
  return e.results;
}

function lruSet(key: string, e: Entry): void {
  lru.delete(key);
  lru.set(key, e);
  while (lru.size > LRU_MAX) lru.delete(lru.keys().next().value!);
}

// ---------------------------------------------------------------- tier 2: Mongo

const collection = async () => (await db()).collection<SearchCacheDoc>(COLLECTIONS.searchCache);

async function mongoGet(key: string): Promise<Entry | null> {
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`no reply within ${READ_TIMEOUT_MS}ms`)), READ_TIMEOUT_MS);
  });
  try {
    // Mongo's TTL sweep runs about once a minute, so an expired row can still be there.
    const doc = await Promise.race([
      collection().then((c) => c.findOne({ _id: key, expiresAt: { $gt: new Date() } })),
      timeout
    ]);
    if (!doc) return null;
    return { results: doc.results as SearchResult[], expiresAt: new Date(doc.expiresAt).getTime() };
  } finally {
    clearTimeout(timer);
  }
}

async function mongoSet(key: string, query: string, e: Entry): Promise<void> {
  const doc: SearchCacheDoc = {
    _id: key,
    provider: env.searchProvider,
    query: normalizeQuery(query),
    results: e.results,
    // A Date, not a string: the TTL index ignores rows whose expiresAt is not a BSON date.
    expiresAt: new Date(e.expiresAt),
    createdAt: new Date()
  };
  await (await collection()).replaceOne({ _id: key }, doc, { upsert: true });
}

// ---------------------------------------------------------------- the one entry point

/**
 * web_search through the cache. Provider failures throw exactly as webSearch does; only
 * the cache's own failures are absorbed, and those are logged.
 */
export async function cachedSearch(
  query: string,
  signal: AbortSignal,
  opts: { bypass: boolean; log: Logger; requestId: string }
): Promise<CachedSearch> {
  const { log, requestId } = opts;
  if (opts.bypass) return { results: await webSearch(query, signal), from: 'bypass' };

  const key = cacheKey(query);
  const inProcess = lruGet(key);
  if (inProcess) return { results: inProcess, from: 'lru' };

  let cacheError: string | undefined;
  try {
    const row = await mongoGet(key);
    if (row) {
      lruSet(key, row);
      return { results: row.results, from: 'mongo' };
    }
  } catch (err) {
    cacheError = `search cache read failed: ${(err as Error).message}`;
    log.error({ requestId, err, key, provider: env.searchProvider }, 'search cache read failed; treating as a miss');
  }

  const results = await webSearch(query, signal);
  // An empty page may be a provider hiccup; pinning it for the whole TTL would hide a real answer.
  if (results.length) {
    const entry = { results, expiresAt: Date.now() + env.searchCacheTtlSeconds * 1000 };
    lruSet(key, entry);
    mongoSet(key, query, entry).catch((err: unknown) =>
      log.error({ requestId, err, key, provider: env.searchProvider }, 'search cache write failed')
    );
  }
  return { results, from: 'miss', ...(cacheError ? { cacheError } : {}) };
}
