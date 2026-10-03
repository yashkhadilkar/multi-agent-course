import { config } from 'dotenv';
import { resolve } from 'node:path';
import { SEARCH_PROVIDERS, VECTOR_BACKENDS } from '@lumina/contract';

// The single .env at the assignment root. Provider keys are read HERE and nowhere else.
config({ path: resolve(process.cwd(), '../../.env') });
config({ path: resolve(process.cwd(), '.env') });

const num = (v: string | undefined, fallback: number) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};

/** `FOO=` in .env is an empty string, not undefined; treat it as unset. */
const str = (v: string | undefined, fallback: string) => v?.trim() || fallback;

/**
 * /health names these, so an unrecognised value must stop the process, not be reported
 * as though it were live. `SEARCH_PROVIDER=serpApi` has no code path behind it.
 */
function oneOf<T extends string>(name: string, allowed: readonly T[], fallback: T): T {
  const v = str(process.env[name], fallback);
  if (!(allowed as readonly string[]).includes(v)) {
    throw new Error(`${name}=${v} is not supported; use one of: ${allowed.join(', ')}`);
  }
  return v as T;
}

export const env = {
  port: num(process.env.PORT_AGENT ?? process.env.PORT, 8000),
  mongoUri: process.env.MONGODB_URI ?? '',
  mongoDb: str(process.env.MONGODB_DB, 'lumina'),
  vectorBackend: oneOf('VECTOR_BACKEND', VECTOR_BACKENDS, 'atlas-vector-search'),

  llmProvider: str(process.env.LLM_PROVIDER, 'anthropic'),
  /** Deep search's model, and quick's unless QUICK_LLM_MODEL says otherwise. */
  llmModel: str(process.env.LLM_MODEL, 'claude-sonnet-5'),
  /** The quick gear's model. Priced at its own rate (pricing.ts), whatever deep runs on. */
  quickModel: str(process.env.QUICK_LLM_MODEL, str(process.env.LLM_MODEL, 'claude-sonnet-5')),

  searchProvider: oneOf('SEARCH_PROVIDER', SEARCH_PROVIDERS, 'tavily'),
  searchCacheTtlSeconds: num(process.env.SEARCH_CACHE_TTL_SECONDS, 21600),
  /**
   * Tavily's search_depth. Not part of the search cache key, which SPEC.md fixes as (query,
   * provider): after a change, rows cached at the old depth are served until they expire
   * (SEARCH_CACHE_TTL_SECONDS). fast measured p50 640 ms / p95 735 ms against basic's
   * 1190 / 1526 at the same credit, but its one end-to-end pass was slower to first token
   * (its pages read slower), so basic stays the default until a pass on questions Tavily has
   * not cached says otherwise. ultra-fast returns one page summary per result instead of
   * passages.
   */
  tavilySearchDepth: oneOf('TAVILY_SEARCH_DEPTH', ['advanced', 'basic', 'fast', 'ultra-fast'] as const, 'basic'),
  /**
   * How long an idle connection to api.tavily.com stays open for the next call. Node's
   * default is 4 s, so each new question paid a fresh TLS handshake (~150-250 ms). Tavily
   * held an idle connection past 330 s; keep this well under that.
   */
  tavilyKeepAliveMs: num(process.env.TAVILY_KEEPALIVE_MS, 60_000),

  /** A single page read gives up after this; one slow publisher must not eat the budget. */
  fetchTimeoutMs: num(process.env.FETCH_TIMEOUT_MS, 10000),
  /** The up-front reads of the top results get much less: they are on the path to the first token. */
  prefetchTimeoutMs: num(process.env.PREFETCH_TIMEOUT_MS, 1500),

  // The quick gear's path to the first token (quick.ts). Each is a latency/grounding trade.
  /** How many of the first search's results are read before the model is called. */
  quickPrefetchPages: num(process.env.QUICK_PREFETCH_PAGES, 3),
  /** The up-front reads stop waiting once this many pages are in and citable... */
  quickPrefetchEnough: num(process.env.QUICK_PREFETCH_ENOUGH, 2),
  /** ...giving the rest this much longer. */
  quickPrefetchGraceMs: num(process.env.QUICK_PREFETCH_GRACE_MS, 200),
  /** Text is committed as the answer once it cites a source or runs this long. */
  quickCommitAfterChars: num(process.env.QUICK_COMMIT_AFTER_CHARS, 40),

  // Model prices behind done.costUsd live in pricing.ts, per model. Tavily's are here.
  searchUsdPerCall: num(process.env.SEARCH_USD_PER_CALL, 0.008),
  /** sla.json max_cost_per_answer_usd: a quick run makes no extra model call that would take it past this. */
  maxCostPerAnswerUsd: num(process.env.MAX_COST_PER_ANSWER_USD, 0.05),

  embeddingModel: process.env.EMBEDDING_MODEL ?? 'text-embedding-3-small',

  // Ingestion, on the jobs worker. Chunk sizes are characters (English runs about four to a token).
  chunkSizeChars: num(process.env.CHUNK_SIZE_CHARS, 1000),
  chunkOverlapChars: num(process.env.CHUNK_OVERLAP_CHARS, 200),
  /** Chunks per embeddings request. A resumed job restarts after the last batch it finished. */
  embedBatchSize: num(process.env.EMBED_BATCH_SIZE, 64),
  /** DESIGN.md: a running job with no heartbeat for this long is stale; /health uses the same window for the worker. */
  jobStaleMs: num(process.env.JOB_STALE_MS, 120_000),
  jobMaxAttempts: num(process.env.JOB_MAX_ATTEMPTS, 3),
  workerHeartbeatMs: num(process.env.WORKER_HEARTBEAT_MS, 10_000),
  workerPollMs: num(process.env.WORKER_POLL_MS, 1000),
  /** How long the read-your-write probe waits for Atlas to make a fresh chunk searchable. */
  probeTimeoutMs: num(process.env.PROBE_TIMEOUT_MS, 90_000),

  // Hybrid retrieval (search_documents): $vectorSearch and BM25 $search, fused by RRF.
  /** Chunks search_documents returns, after fusion and after unindexed documents are dropped. */
  ragTopK: num(process.env.RAG_TOP_K, 5),
  /** RRF's damping constant: score = Σ 1 / (k + rank). 60 is the original paper's value. */
  ragRrfK: num(process.env.RAG_RRF_K, 60),
  /**
   * How many hits each retriever returns before fusion. Over-fetched past top-k so that
   * chunks of documents not yet `indexed` can be dropped and top-k still filled.
   */
  ragCandidates: num(process.env.RAG_CANDIDATES, 20),

  // Deep search is the expensive gear, so its limits are configuration, not code.
  deepSubQuestionsMin: num(process.env.DEEP_SUB_QUESTIONS_MIN, 3),
  deepSubQuestionsMax: num(process.env.DEEP_SUB_QUESTIONS_MAX, 6),
  deepDailyCap: num(process.env.DEEP_DAILY_CAP, 5),
  /** Sub-questions researched at once (DESIGN.md: "a few at a time"). */
  deepConcurrency: num(process.env.DEEP_CONCURRENCY, 3),
  /** One page read on a deep run gives up after this; a slow publisher must not hold up its wave. */
  deepFetchTimeoutMs: num(process.env.DEEP_FETCH_TIMEOUT_MS, 6000),

  // The hard caps from AGENTS.md. Raising these to make a gate pass is the failure mode
  // the caps exist to catch. Two gears, two envelopes.
  maxToolCalls: num(process.env.MAX_TOOL_CALLS, 8),
  maxWallClockSec: num(process.env.MAX_WALL_CLOCK_SEC, 90),
  maxToolCallsDeep: num(process.env.MAX_TOOL_CALLS_DEEP, 24),
  maxWallClockSecDeep: num(process.env.MAX_WALL_CLOCK_SEC_DEEP, 240),
  /** expectations.json trajectory.maxConsecutiveSameTool (rule A3): the longest run of one tool. */
  maxConsecutiveSameTool: num(process.env.MAX_CONSECUTIVE_SAME_TOOL, 4),

  logLevel: process.env.LOG_LEVEL ?? 'info'
} as const;

if (env.ragTopK < 1 || env.ragCandidates < env.ragTopK || env.ragRrfK < 0) {
  throw new Error(`RAG_TOP_K (${env.ragTopK}) must be at least 1, RAG_CANDIDATES (${env.ragCandidates}) at least RAG_TOP_K, and RAG_RRF_K (${env.ragRrfK}) not negative`);
}

// An overlap as long as the chunk never advances; refuse it at boot rather than spin.
if (env.chunkOverlapChars < 0 || env.chunkOverlapChars >= env.chunkSizeChars) {
  throw new Error(`CHUNK_OVERLAP_CHARS (${env.chunkOverlapChars}) must be at least 0 and below CHUNK_SIZE_CHARS (${env.chunkSizeChars})`);
}

if (env.quickPrefetchPages < 0 || env.quickPrefetchEnough < 1 || env.quickPrefetchGraceMs < 0 || env.quickCommitAfterChars < 0) {
  throw new Error(
    `QUICK_PREFETCH_PAGES (${env.quickPrefetchPages}) and QUICK_PREFETCH_GRACE_MS (${env.quickPrefetchGraceMs}) must not be negative, ` +
      `QUICK_PREFETCH_ENOUGH (${env.quickPrefetchEnough}) must be at least 1, and QUICK_COMMIT_AFTER_CHARS (${env.quickCommitAfterChars}) not negative`
  );
}

/** Never log or return these. /health names the model; it never echoes a key. */
export const secrets = {
  anthropic: process.env.ANTHROPIC_API_KEY ?? '',
  openai: process.env.OPENAI_API_KEY ?? '',
  tavily: process.env.TAVILY_API_KEY ?? '',
  serpapi: process.env.SERPAPI_API_KEY ?? ''
} as const;
