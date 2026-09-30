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
  llmModel: str(process.env.LLM_MODEL, 'claude-sonnet-5'),

  searchProvider: oneOf('SEARCH_PROVIDER', SEARCH_PROVIDERS, 'tavily'),
  searchCacheTtlSeconds: num(process.env.SEARCH_CACHE_TTL_SECONDS, 21600),

  /** A single page read gives up after this; one slow publisher must not eat the budget. */
  fetchTimeoutMs: num(process.env.FETCH_TIMEOUT_MS, 10000),
  /** The up-front reads of the top results get much less: they are on the path to the first token. */
  prefetchTimeoutMs: num(process.env.PREFETCH_TIMEOUT_MS, 1500),

  // Prices behind done.costUsd. The defaults are the declared table in
  // benchmark/sla.json (cost_model), so the agent and the bench price a run the same way.
  llmInputUsdPerMtok: num(process.env.LLM_INPUT_USD_PER_MTOK, 3.0),
  llmOutputUsdPerMtok: num(process.env.LLM_OUTPUT_USD_PER_MTOK, 15.0),
  searchUsdPerCall: num(process.env.SEARCH_USD_PER_CALL, 0.008),

  embeddingModel: process.env.EMBEDDING_MODEL ?? 'text-embedding-3-small',

  // Deep search is the expensive gear, so its limits are configuration, not code.
  deepSubQuestionsMin: num(process.env.DEEP_SUB_QUESTIONS_MIN, 3),
  deepSubQuestionsMax: num(process.env.DEEP_SUB_QUESTIONS_MAX, 6),
  deepDailyCap: num(process.env.DEEP_DAILY_CAP, 5),

  // The hard caps from AGENTS.md. Raising these to make a gate pass is the failure mode
  // the caps exist to catch. Two gears, two envelopes.
  maxToolCalls: num(process.env.MAX_TOOL_CALLS, 8),
  maxWallClockSec: num(process.env.MAX_WALL_CLOCK_SEC, 90),
  maxToolCallsDeep: num(process.env.MAX_TOOL_CALLS_DEEP, 24),
  maxWallClockSecDeep: num(process.env.MAX_WALL_CLOCK_SEC_DEEP, 240),

  logLevel: process.env.LOG_LEVEL ?? 'info',
  /** Where the per-answer run logs land. quality/check.mjs reads this folder. */
  runsDir: resolve(process.cwd(), '../../runs')
} as const;

/** Never log or return these. /health names the model; it never echoes a key. */
export const secrets = {
  anthropic: process.env.ANTHROPIC_API_KEY ?? '',
  openai: process.env.OPENAI_API_KEY ?? '',
  tavily: process.env.TAVILY_API_KEY ?? '',
  serpapi: process.env.SERPAPI_API_KEY ?? ''
} as const;
