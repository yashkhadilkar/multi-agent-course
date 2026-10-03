/**
 * What a run costs, priced the same way by both gears. Input tokens are billed three ways:
 * uncached input at the base rate, cache writes (5-minute TTL) at 1.25x, and cache reads at
 * 0.1x. `done.tokens.in` still reports all three together; only the price differs.
 *
 * Each gear can run a different model (QUICK_LLM_MODEL, LLM_MODEL), so every price is looked
 * up by the model that made the call.
 */
import type Anthropic from '@anthropic-ai/sdk';
import { env } from './env.js';

/**
 * USD per million tokens. Sonnet 5 is priced at benchmark/sla.json's declared table
 * (cost_model), not its published $2/$10, so the agent and the bench price a run the same way
 * until the course's price table is settled. Haiku 4.5 is its published rate.
 */
const MODEL_PRICES: Record<string, { input: number; output: number }> = {
  'claude-sonnet-5': { input: 3.0, output: 15.0 },
  'claude-haiku-4-5': { input: 1.0, output: 5.0 }
};

type Rates = { input: number; output: number; cacheWrite: number; cacheRead: number };

/** A dated snapshot id (claude-haiku-4-5-20251001) is priced as its model. */
function ratesFor(model: string): Rates {
  const p = MODEL_PRICES[model.replace(/-\d{8}$/, '')];
  if (!p) throw new Error(`no published price for model ${model}: add it to MODEL_PRICES in pricing.ts`);
  return { ...p, cacheWrite: 1.25 * p.input, cacheRead: 0.1 * p.input };
}

// A model that cannot be priced would report every answer as free; refuse it at boot.
ratesFor(env.llmModel);
ratesFor(env.quickModel);

export type Usage = {
  /** Uncached input tokens. */
  in: number;
  cacheWrite: number;
  cacheRead: number;
  out: number;
  /** Search provider calls that went to the provider; a cache hit is not one. */
  searches: number;
  /** Tavily extract calls. */
  extracts: number;
};

export const emptyUsage = (): Usage => ({ in: 0, cacheWrite: 0, cacheRead: 0, out: 0, searches: 0, extracts: 0 });

/** Adds one model response's usage. */
export function addMessageUsage(usage: Usage, u: Anthropic.Usage): void {
  usage.in += u.input_tokens;
  usage.cacheWrite += u.cache_creation_input_tokens ?? 0;
  usage.cacheRead += u.cache_read_input_tokens ?? 0;
  usage.out += u.output_tokens;
}

/** Every input token the model read, cached or not. */
export const inputTokens = (u: Usage) => u.in + u.cacheWrite + u.cacheRead;

export function llmCostUsd(model: string, t: { in: number; cacheWrite: number; cacheRead: number; out: number }): number {
  const r = ratesFor(model);
  return (t.in * r.input + t.cacheWrite * r.cacheWrite + t.cacheRead * r.cacheRead + t.out * r.output) / 1e6;
}

/** Tavily bills extract at one credit per five pages, a fifth of a search. */
export const toolCostUsd = (t: { searches: number; extracts: number }) =>
  t.searches * env.searchUsdPerCall + t.extracts * (env.searchUsdPerCall / 5);

export const costUsd = (model: string, u: Usage) => llmCostUsd(model, u) + toolCostUsd(u);
