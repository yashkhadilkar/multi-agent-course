/**
 * Hybrid retrieval over a Space's chunks, behind the search_documents tool.
 *
 *   dense    $vectorSearch on chunks.embedding (chunks_vector)
 *   lexical  $search BM25 on chunks.text (chunks_text)
 *   fusion   reciprocal rank fusion: score(c) = Σ 1 / (RAG_RRF_K + rank), ranks 1-based
 *
 * Both retrievers filter by userId AND spaceId inside the search stage itself. A $match
 * after the stage would let another Space's chunks take the top slots and then hide them,
 * which quietly under-retrieves.
 *
 * Neither index can filter by document status, so (DESIGN.md, State) each retriever
 * over-fetches RAG_CANDIDATES hits, chunks of documents that are not `indexed` are dropped
 * after fusion, and RAG_TOP_K are returned. If the drop leaves fewer than top-k while a
 * retriever still had more to give, the fetch doubles and runs again.
 *
 * No re-ranker: DESIGN.md (Trade-offs) leaves it out to protect the quick TTFT budget, and
 * it is the first thing to add if recall@5 falls below the gold-set threshold.
 */
import { MongoError } from 'mongodb';
import { COLLECTIONS, SEARCH_INDEXES, type ChunkDoc, type Locator } from '@lumina/contract';
import { db } from './db.js';
import { cosine, embedOne, EmbeddingError } from './embeddings.js';
import { env } from './env.js';
import { terms } from './passages.js';
import { documents } from './spaces.js';

/** Atlas guidance: numCandidates 10 to 20 times the limit. */
const NUM_CANDIDATES_PER_HIT = 15;
/** The over-fetch never grows past this many hits per retriever. */
const MAX_CANDIDATES = 200;
/** BM25's usual parameters, for the local cosine-scan fallback only; Atlas Search uses its own. */
const BM25_K1 = 1.2;
const BM25_B = 0.75;

export type RetrievedChunk = {
  id: string;
  docId: string;
  /** The document's title (its filename), from the documents collection. */
  title: string;
  text: string;
  locator: Locator;
  /** Fused RRF score. */
  score: number;
  /** 1-based rank in each retriever's list; absent when that retriever did not return it. */
  vectorRank?: number;
  textRank?: number;
};

export type DocumentSearch = {
  chunks: RetrievedChunk[];
  /** Hits each retriever returned on the last pass, before fusion. */
  vectorHits: number;
  textHits: number;
  /** Fused chunks dropped because their document is not `indexed`. */
  dropped: number;
};

/** Document search failed: the embedding provider or the store. Not the same as no hits. */
export class DocumentSearchError extends Error {
  override name = 'DocumentSearchError';
}

type Hit = Pick<ChunkDoc, '_id' | 'docId' | 'text' | 'locator'>;
type Candidates = { vector: Hit[]; text: Hit[] };

const chunks = async () => (await db()).collection<ChunkDoc>(COLLECTIONS.chunks);
const PROJECT = { _id: 1, docId: 1, text: 1, locator: 1 } as const;

export async function searchDocuments(userId: string, spaceId: string, query: string, signal?: AbortSignal): Promise<DocumentSearch> {
  try {
    const vector = await embedOne(query, signal);
    for (let limit = env.ragCandidates; ; limit = Math.min(limit * 2, MAX_CANDIDATES)) {
      signal?.throwIfAborted();
      const lists =
        env.vectorBackend === 'mongo-cosine-scan'
          ? await scanCandidates(userId, spaceId, query, vector, limit)
          : await atlasCandidates(userId, spaceId, query, vector, limit);
      signal?.throwIfAborted();
      const fused = fuse(lists);
      const titles = await indexedTitles(userId, spaceId, [...new Set(fused.map((c) => c.docId))]);
      const kept = fused.filter((c) => titles.has(c.docId));
      const exhausted = lists.vector.length < limit && lists.text.length < limit;
      if (kept.length >= env.ragTopK || exhausted || limit >= MAX_CANDIDATES) {
        return {
          chunks: kept.slice(0, env.ragTopK).map((c) => ({ ...c, title: titles.get(c.docId)! })),
          vectorHits: lists.vector.length,
          textHits: lists.text.length,
          dropped: fused.length - kept.length
        };
      }
    }
  } catch (err) {
    if (signal?.aborted || err instanceof DocumentSearchError) throw err;
    // A driver message can name the cluster host; the caller logs `cause`, the trace gets this.
    const message =
      err instanceof EmbeddingError ? err.message : err instanceof MongoError ? 'document store error (see the agent log)' : (err as Error).message;
    throw new DocumentSearchError(`document search: ${message}`, { cause: err });
  }
}

// ---------------------------------------------------------------- the two retrievers

async function atlasCandidates(userId: string, spaceId: string, query: string, vector: number[], limit: number): Promise<Candidates> {
  const col = await chunks();
  const [dense, lexical] = await Promise.all([
    col
      .aggregate<Hit>([
        {
          $vectorSearch: {
            index: SEARCH_INDEXES.chunksVector,
            path: 'embedding',
            queryVector: vector,
            numCandidates: limit * NUM_CANDIDATES_PER_HIT,
            limit,
            filter: { $and: [{ spaceId: { $eq: spaceId } }, { userId: { $eq: userId } }] }
          }
        },
        { $project: PROJECT }
      ])
      .toArray(),
    col
      .aggregate<Hit>([
        {
          $search: {
            index: SEARCH_INDEXES.chunksText,
            compound: {
              must: [{ text: { query, path: 'text' } }],
              filter: [
                { equals: { path: 'spaceId', value: spaceId } },
                { equals: { path: 'userId', value: userId } }
              ]
            }
          }
        },
        { $limit: limit },
        { $project: PROJECT }
      ])
      .toArray()
  ]);
  return { vector: dense, text: lexical };
}

/**
 * The documented local fallback (VECTOR_BACKEND=mongo-cosine-scan): a plain mongod has no
 * search indexes, so the Space's chunks are read once and both rankings are computed here.
 */
async function scanCandidates(userId: string, spaceId: string, query: string, vector: number[], limit: number): Promise<Candidates> {
  const rows = await (await chunks()).find({ spaceId, userId }, { projection: { ...PROJECT, embedding: 1 } }).toArray();
  const strip = ({ _id, docId, text, locator }: ChunkDoc): Hit => ({ _id, docId, text, locator });

  const dense = rows
    .map((r) => ({ r, s: cosine(vector, r.embedding) }))
    .sort((a, b) => b.s - a.s)
    .slice(0, limit)
    .map(({ r }) => strip(r));

  const q = [...new Set(terms(query))];
  const bags = rows.map((r) => terms(r.text));
  const avgLen = bags.reduce((n, b) => n + b.length, 0) / (bags.length || 1);
  const df = new Map(q.map((t) => [t, bags.filter((b) => b.includes(t)).length]));
  const idf = (t: string) => Math.log(1 + (rows.length - df.get(t)! + 0.5) / (df.get(t)! + 0.5));
  const lexical = rows
    .map((r, i) => {
      const bag = bags[i]!;
      let s = 0;
      for (const t of q) {
        const tf = bag.filter((w) => w === t).length;
        if (tf) s += idf(t) * ((tf * (BM25_K1 + 1)) / (tf + BM25_K1 * (1 - BM25_B + (BM25_B * bag.length) / avgLen)));
      }
      return { r, s };
    })
    .filter((x) => x.s > 0)
    .sort((a, b) => b.s - a.s)
    .slice(0, limit)
    .map(({ r }) => strip(r));

  return { vector: dense, text: lexical };
}

// ---------------------------------------------------------------- fusion and the status filter

/** Reciprocal rank fusion. Scores are ignored; only ranks count. Best first. */
function fuse({ vector, text }: Candidates): Omit<RetrievedChunk, 'title'>[] {
  const byId = new Map<string, Omit<RetrievedChunk, 'title'>>();
  const add = (hits: Hit[], which: 'vectorRank' | 'textRank') =>
    hits.forEach((h, i) => {
      const rank = i + 1;
      const c = byId.get(h._id) ?? { id: h._id, docId: h.docId, text: h.text, locator: h.locator, score: 0 };
      c.score += 1 / (env.ragRrfK + rank);
      c[which] = rank;
      byId.set(h._id, c);
    });
  add(vector, 'vectorRank');
  add(text, 'textRank');
  const best = (c: Omit<RetrievedChunk, 'title'>) => Math.min(c.vectorRank ?? Infinity, c.textRank ?? Infinity);
  return [...byId.values()].sort((a, b) => b.score - a.score || best(a) - best(b));
}

/** Titles of the given documents that are `indexed`, scoped to the user and Space. */
async function indexedTitles(userId: string, spaceId: string, docIds: string[]): Promise<Map<string, string>> {
  if (!docIds.length) return new Map();
  const rows = await (await documents())
    .find({ _id: { $in: docIds }, spaceId, userId, status: 'indexed' }, { projection: { title: 1 } })
    .toArray();
  return new Map(rows.map((d) => [d._id, d.title]));
}

// ---------------------------------------------------------------- the Space, for the prompt

export type SpaceContents = {
  id: string;
  name: string;
  /** Searchable now. */
  indexed: { title: string; pages?: number }[];
  /** Uploaded but not searchable yet (pending, parsing, embedding). Failed ones are left out. */
  indexing: string[];
};

export async function spaceContents(userId: string, spaceId: string, name: string): Promise<SpaceContents> {
  const rows = await (await documents())
    .find({ spaceId, userId, status: { $ne: 'failed' } }, { projection: { title: 1, status: 1, pages: 1 } })
    .sort({ createdAt: 1 })
    .toArray();
  return {
    id: spaceId,
    name,
    indexed: rows.filter((d) => d.status === 'indexed').map((d) => ({ title: d.title, ...(d.pages ? { pages: d.pages } : {}) })),
    indexing: rows.filter((d) => d.status !== 'indexed').map((d) => d.title)
  };
}
