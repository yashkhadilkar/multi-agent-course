/**
 * What a run costs, priced the same way by both gears. Input tokens are billed three ways:
 * uncached input at the base rate, cache writes (5-minute TTL) at 1.25x, and cache reads at
 * 0.1x (env.ts). `done.tokens.in` still reports all three together; only the price differs.
 */
import type Anthropic from '@anthropic-ai/sdk';
import { env } from './env.js';

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

export function llmCostUsd(t: { in: number; cacheWrite: number; cacheRead: number; out: number }): number {
  return (
    (t.in * env.llmInputUsdPerMtok +
      t.cacheWrite * env.llmCacheWriteUsdPerMtok +
      t.cacheRead * env.llmCacheReadUsdPerMtok +
      t.out * env.llmOutputUsdPerMtok) /
    1e6
  );
}

/** Tavily bills extract at one credit per five pages, a fifth of a search. */
export const toolCostUsd = (t: { searches: number; extracts: number }) =>
  t.searches * env.searchUsdPerCall + t.extracts * (env.searchUsdPerCall / 5);

export const costUsd = (u: Usage) => llmCostUsd(u) + toolCostUsd(u);
