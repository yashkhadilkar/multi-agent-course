/**
 * The QUICK gear: plan → choose tool → observe → repeat → answer, with web_search and
 * fetch_page, streamed as trace* → sources → token* → done.
 *
 * How grounding is kept, in order:
 *   1. Only a page fetch_page actually read becomes a source. It is numbered at the moment
 *      it is read, and the model sees that number next to the passages it may cite.
 *   2. The source's snippet is cut from the fetched text (passages.ts). The model never
 *      writes it.
 *   3. The source list is final before the first token: once the answer starts, no more
 *      tools run, so the numbers the model was shown are the only ones that exist.
 *   4. Any [n] in the streamed text that is not in that list is dropped before it reaches
 *      the client, and logged.
 *   5. No page readable at all → the answer falls back to the search snippets; nothing
 *      retrieved → the answer says so and cites nothing.
 *
 * Step one needs no model call: the question is searched as asked, and the top results
 * are read in parallel under a short timeout, keeping whatever pages return in time. The
 * model's first turn already has those pages, so a simple question is answered in one
 * LLM call; a harder one can still search and read more, inside the same caps.
 *
 * recall_memory runs in that same step, in parallel with the search, on every ask: the
 * user's saved preferences reach the model whether or not it would have thought to look.
 * save_memory is the model's to call, and only when the user states something durable.
 *
 * The final answer is the model's last turn, streamed as it is written. Text the model
 * writes before a tool call (a preamble) is held back until the turn shows whether it is
 * an answer, and discarded if it is not.
 */
import Anthropic from '@anthropic-ai/sdk';
import type { Response } from 'express';
import type { Logger } from 'pino';
import { z } from 'zod';
import { newId, type DoneEvent, type RunLog, type Source, type Terminated, type ToolName } from '@lumina/contract';
import { cachedSearch, isTimeSensitive } from './cache.js';
import { env, secrets } from './env.js';
import { MAX_MEMORY_CHARS, recallMemories, saveMemory, type RecalledMemory } from './memory.js';
import { selectPassages } from './passages.js';
import { recordAnswer } from './runlog.js';
import { FetchError, SearchProviderError, fetchPage, hostOf, type SearchResult } from './search.js';
import { CitationFilter, SseStream } from './sse.js';
import type { SavedAnswer } from './threads.js';

/** Words of each fetched page the model reads. Four pages of this stay well inside quick's cost budget. */
const PAGE_WORDS = 700;
/** Research stops starting new turns this long before the wall-clock cap, to leave room to answer. */
const ANSWER_RESERVE_MS = 15_000;
/** Thinking counts against this too. It bounds the cost of any one turn. */
const MAX_TOKENS_PER_TURN = 2048;
/** How many of the first search's results are read before the model is called. */
const PREFETCH_PAGES = 3;
/**
 * With pages in hand the model may answer at once, or open with a preamble and then call a
 * tool. Its text is committed as the answer once it cites a source or runs this long.
 */
const COMMIT_AFTER_CHARS = 160;

// ---------------------------------------------------------------- tools

/**
 * The quick toolbelt. plan_research is not in it, and dispatch below refuses anything not
 * in it: a prompt asking the model not to escalate is a suggestion, this is the gate.
 */
const QUICK_TOOLS: Anthropic.Tool[] = [
  {
    name: 'web_search',
    description:
      'Search the web. Returns up to 5 results with title, url and a short snippet. Snippets are only for ' +
      'choosing what to read: they cannot be cited.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The search query.' },
        reason: { type: 'string', description: 'A short phrase: why this search.' }
      },
      required: ['query'],
      additionalProperties: false
    },
    eager_input_streaming: true
  },
  {
    name: 'fetch_page',
    description:
      'Read one page that web_search returned. Returns the source number for citing it and the passages of ' +
      'the page most relevant to the question. Only pages read with this tool can be cited.',
    input_schema: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'A url exactly as web_search returned it.' },
        reason: { type: 'string', description: 'A short phrase: why this page.' }
      },
      required: ['url'],
      additionalProperties: false
    },
    eager_input_streaming: true
  },
  {
    name: 'save_memory',
    description:
      'Remember a stable preference or fact the user has stated about themselves, for all their future ' +
      'conversations. Only for what they said about themselves (how they want answers, what they work with, ' +
      'who they are), never for facts from search results or one-off details of this question. A memory that ' +
      'restates one already saved is not saved again.',
    input_schema: {
      type: 'object',
      properties: {
        text: {
          type: 'string',
          description: 'One short standalone sentence about the user, e.g. "Prefers code examples in TypeScript."'
        },
        reason: { type: 'string', description: 'A short phrase: why this is worth remembering.' }
      },
      required: ['text'],
      additionalProperties: false
    }
  }
];
const QUICK_TOOL_NAMES = new Set(QUICK_TOOLS.map((t) => t.name));

// With eager input streaming the API no longer validates tool inputs, so we do.
const WebSearchInput = z.object({ query: z.string().trim().min(1).max(400), reason: z.string().optional() });
const FetchPageInput = z.object({ url: z.string().url(), reason: z.string().optional() });
const SaveMemoryInput = z.object({ text: z.string().trim().min(1).max(MAX_MEMORY_CHARS), reason: z.string().optional() });

function systemPrompt(): string {
  const today = new Date().toISOString().slice(0, 10);
  return `You are LUMINA, a search assistant that answers only from web pages it has actually read. Today is ${today}.

How to research:
- Before your first turn the question has already been searched once, as asked, and the top results read. Their passages come with the question.
- If those passages answer the question, answer straight away. If not, use web_search and fetch_page to look further. Make independent calls in parallel, in the same turn.
- A run has at most ${env.maxToolCalls} tool calls, including the ones already made for you; the question says how many are left.
- At most ${env.maxConsecutiveSameTool} calls in a row may use the same tool, counting the reads already made for you. A call past that is refused: answer from what you have, or use a different tool.
- Write no text before or between tool calls. The only text you write is the final answer.

Memory:
- What you know about the user from earlier conversations comes with the question, under "About this user". Follow their stated preferences in your answer. Memories are not sources: never cite them.
- When the user states a lasting preference or fact about themselves ("remember that…", "I always want…", "I work in…"), call save_memory with it as one short sentence, then briefly confirm. Do not save facts from pages, or details that only matter for this one question.

How to answer:
- Put the direct answer in the first sentence, then only what the question needs. Keep it under about 150 words unless the question asks for depth; no headings for a short answer.
- Each page you read has a source number. Cite a claim by putting that number in square brackets right after it, like this [2]. One number per bracket: [1][3], never [1, 3].
- Cite only the source numbers given with the pages you read. Search results are not sources.
- Make claims only from the passages you read. If they do not answer the question, say that plainly instead of guessing.

Earlier turns:
- If this is a follow-up, the conversation so far comes before the question. Use it to work out what the question refers to.
- Earlier answers had their citation numbers removed, and the pages behind them cannot be cited now. Cite only pages read for this question.`;
}

let anthropic: Anthropic | null = null;
const llm = () => (anthropic ??= new Anthropic({ apiKey: secrets.anthropic, maxRetries: 1 }));

// ---------------------------------------------------------------- the run

export type QuickAsk = {
  requestId: string;
  userId: string;
  threadId: string;
  query: string;
  /** Earlier turns of the thread, already trimmed, as alternating user/assistant messages. */
  history: Anthropic.MessageParam[];
  /** The thread's earlier questions, whole and oldest first, for a follow-up's first search. */
  earlierQuestions: string[];
  /** Persists a done or capped answer. Runs before `done` is sent; a throw ends the run as an error. */
  saveAnswer: (answer: SavedAnswer) => Promise<void>;
};

type Page = { n: number; url: string; title: string; snippet: string };

/** Why the run was stopped from outside the loop. */
class WallClockCap extends Error {}
class ClientGone extends Error {}
/** The model declined. Not an exception from the SDK, but the answer failed upstream all the same. */
class ModelRefusal extends Error {}

const asInput = (raw: unknown): Record<string, unknown> =>
  raw && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {};

const normUrl = (u: string) => u.replace(/#.*$/, '').replace(/\/+$/, '');

/** The recalled memories as the model sees them, most relevant first. Empty when there are none. */
function memoryNote({ memories, error }: { memories: RecalledMemory[]; error?: string }): string {
  if (error) return 'About this user: saved memories could not be loaded for this question.';
  if (!memories.length) return '';
  const lines = memories.map((m) => `- ${m.text} (saved ${m.createdAt.toISOString().slice(0, 10)})`);
  return `About this user (from earlier conversations; follow these preferences, and where two disagree the newer wins; not citable):\n${lines.join('\n')}`;
}

// ---------------------------------------------------------------- follow-ups

/** Search queries are capped at this many characters. */
const MAX_QUERY_CHARS = 400;
const POINTS_BACK = /\b(?:it|its|they|them|this|that|these|those)\b/gi;
const FOLLOW_UP_OPENER = /^\s*(?:what about|how about|and)\b/i;
/**
 * The question names its own subject before the pronoun, so the pronoun points inside the
 * question: a "what is X" opener ("What is GridFS and when should you use it…"), or a
 * capitalised name after the first word ("How does SerpApi price its…"; "I" does not count).
 */
const WHAT_IS_OPENER = /^\s*(?:what|who)\s+(?:is|are)\s+(?!(?:it|its|they|them|this|that|these|those)\b)\w/i;
const NAME_AFTER_FIRST_WORD = /\S\s+(?!I\b)[A-Z]/;
/** "What would it cost to…", "how long does it take to…": an "it" standing in for the rest of the sentence. */
const PLACEHOLDER_IT = /^it\s+\w+\s+to\b/i;

/** Whether a question leans on an earlier turn for what it is about. */
export function refersBack(query: string): boolean {
  if (FOLLOW_UP_OPENER.test(query)) return true;
  return [...query.matchAll(POINTS_BACK)].some((m) => {
    if (PLACEHOLDER_IT.test(query.slice(m.index))) return false;
    const before = query.slice(0, m.index);
    return !WHAT_IS_OPENER.test(before) && !NAME_AFTER_FIRST_WORD.test(before);
  });
}

/**
 * The first search for a follow-up that points back: the latest earlier question that names
 * its own subject, then this one. Pairing with the previous question alone fails on a chain
 * ("does it…", then "and its…"), where the previous question has no subject either.
 * Null for everything else, which is searched exactly as asked, so a standalone question
 * keeps its cache key however long the thread is.
 */
function firstSearchQuery(ask: QuickAsk): string | null {
  if (!ask.earlierQuestions.length || !refersBack(ask.query) || ask.query.length >= MAX_QUERY_CHARS - 20) return null;
  const anchor = ask.earlierQuestions.filter((q) => !refersBack(q)).at(-1) ?? ask.earlierQuestions.at(-1)!;
  const room = MAX_QUERY_CHARS - ask.query.length - 1;
  return `${anchor.replace(/\s+/g, ' ').trim().slice(0, room)} ${ask.query}`;
}

export async function runQuick(ask: QuickAsk, res: Response, log: Logger): Promise<void> {
  const started = Date.now();
  const deadline = started + env.maxWallClockSec * 1000;
  const sse = new SseStream(res);
  const answerId = newId('ans');

  // One signal for everything this run starts: the LLM stream and every tool call.
  const abort = new AbortController();
  const hardStop = setTimeout(() => abort.abort(new WallClockCap()), env.maxWallClockSec * 1000);
  res.on('close', () => {
    if (!res.writableFinished) abort.abort(new ClientGone('client disconnected'));
  });

  // The thread so far, then the question's message, built from the up-front search and reads.
  const messages: Anthropic.MessageParam[] = [...ask.history];
  const searchResults = new Map<string, SearchResult>(); // normalized url → first result that named it
  const searchQueries: string[] = [];
  const pages: Page[] = [];
  const pageByUrl = new Map<string, Page>();
  // Pushed as calls finish; the run log lists them by step, the order they were made.
  const calls: { step: number; call: RunLog['toolCalls'][number] }[] = [];
  /** Calls the thrash guard refused. Traced, but not in the run log, since they never ran. */
  const refused: { step: number; tool: ToolName }[] = [];
  // `searches` is provider calls, the ones that cost money; a cache hit is not one.
  const usage = { in: 0, out: 0, searches: 0, extracts: 0 };
  const cache = { searches: 0, hits: 0, readErrors: 0 };
  // A time-sensitive question keeps every search in the run fresh, not only the ones
  // whose own wording says so: the model's follow-up queries may drop the "latest".
  const freshQuestion = isTimeSensitive(ask.query);

  let capped = false;
  let snippetFallback = false;
  // Assigned inside closures, so TypeScript must not narrow these to their initial null.
  let sources = null as Source[] | null;
  let filter = null as CitationFilter | null;
  let ttftMs = null as number | null;
  /** Exactly what the client was sent, which is what the thread saves. */
  let answerText = '';

  // ---------------------------------------------------------------- answer output

  /** Fixes the source list and emits it. After this, no tool may run. */
  const beginAnswer = (list: Source[]) => {
    if (sources) return;
    sources = list;
    filter = new CitationFilter(new Set(list.map((s) => s.n)));
    sse.send('sources', list);
  };

  const pageSources = (): Source[] =>
    pages.map((p) => ({ n: p.n, kind: 'web', title: p.title, url: p.url, snippet: p.snippet }));

  const emit = (text: string) => {
    if (!text) return;
    ttftMs ??= Date.now() - started;
    answerText += text;
    sse.send('token', { text });
  };
  const write = (delta: string) => emit(filter!.push(delta));

  // ---------------------------------------------------------------- tools

  let step = 0;
  let fetchAttempts = 0;

  // The thrash guard (rule A3). Steps are reserved in the order calls are made, and the
  // streak is counted in that same order, so it covers the up-front reads too.
  let streakTool = null as ToolName | null;
  let streak = 0;
  /** The next step, for a call that will run. */
  const reserve = (tool: ToolName): number => {
    streak = tool === streakTool ? streak + 1 : 1;
    streakTool = tool;
    return ++step;
  };

  const trace = (
    s: number,
    tool: ToolName,
    input: Record<string, unknown>,
    t0: number,
    outcome: { ok: true; reason: string } | { ok: false; reason: string; error: string }
  ) => {
    const ms = Date.now() - t0;
    calls.push({ step: s, call: outcome.ok ? { name: tool, ok: true, ms } : { name: tool, ok: false, error: outcome.error, ms } });
    sse.send('trace', { step: s, tool, input, ok: outcome.ok, ms, reason: outcome.reason, ...(outcome.ok ? {} : { error: outcome.error }) });
  };

  /** A tool signal that dies with the run, with its batch (see settleBatch), or after the per-call timeout. */
  const toolSignal = (batch: AbortSignal | undefined, timeoutMs = env.fetchTimeoutMs) =>
    AbortSignal.any([abort.signal, ...(batch ? [batch] : []), AbortSignal.timeout(timeoutMs)]);

  /**
   * Why a call was stopped from outside, as its step's error; null when it failed on its own.
   * A call that was stopped is still traced, so the run log shows every call that was made.
   */
  const cutShort = (batch: AbortSignal | undefined): string | null => {
    if (abort.signal.aborted) {
      const r: unknown = abort.signal.reason;
      return r instanceof WallClockCap
        ? `cancelled at the ${env.maxWallClockSec} s wall-clock cap`
        : `cancelled: ${(r as Error | undefined)?.message || 'the run was aborted'}`;
    }
    return batch?.aborted ? String((batch.reason as Error).message) : null;
  };

  /**
   * Runs one batch of parallel tool calls. The first call to throw cancels the rest of the
   * batch, and every call settles, tracing its step, before that first error goes on to end
   * the run. A plain Promise.all would record the run while its siblings were still in flight.
   */
  async function settleBatch<T>(calls: ((batch: AbortSignal) => Promise<T>)[]): Promise<T[]> {
    const batch = new AbortController();
    let failure = null as { reason: unknown } | null;
    const settled = await Promise.allSettled(
      calls.map((call) =>
        call(batch.signal).catch((reason: unknown) => {
          if (!failure) {
            failure = { reason };
            batch.abort(new Error(`cancelled because a parallel call failed: ${(reason as Error).message || 'unknown error'}`));
          }
          throw reason;
        })
      )
    );
    if (failure) throw failure.reason;
    return settled.map((r) => (r as PromiseFulfilledResult<T>).value);
  }

  async function runWebSearch(s: number, raw: unknown, batch?: AbortSignal): Promise<{ content: string; results: SearchResult[] }> {
    const t0 = Date.now();
    const parsed = WebSearchInput.safeParse(raw);
    if (!parsed.success) {
      const error = `invalid input: ${parsed.error.issues[0]?.message ?? 'bad query'}`;
      trace(s, 'web_search', asInput(raw), t0, { ok: false, reason: 'rejected before searching', error });
      return { content: `Error: ${error}`, results: [] };
    }
    const { query, reason } = parsed.data;
    searchQueries.push(query);
    let results: SearchResult[];
    let cacheNote: string;
    try {
      const bypass = freshQuestion || isTimeSensitive(query);
      cache.searches++;
      const r = await cachedSearch(query, toolSignal(batch), { bypass, log, requestId: ask.requestId });
      results = r.results;
      if (r.from === 'lru' || r.from === 'mongo') cache.hits++;
      else usage.searches++;
      if (r.cacheError) cache.readErrors++;
      cacheNote =
        r.from === 'bypass'
          ? 'time-sensitive, cache bypassed'
          : r.cacheError
            ? `${r.cacheError}, searched live`
            : r.from === 'miss'
              ? 'cache miss'
              : `cache hit (${r.from})`;
    } catch (err) {
      const cancelled = cutShort(batch);
      if (cancelled) {
        trace(s, 'web_search', { query }, t0, { ok: false, reason: reason ?? 'search', error: cancelled });
        throw err;
      }
      // The provider is down: trace it, then end the run. There is nothing to answer from.
      const e = err instanceof SearchProviderError ? err : new SearchProviderError(`${env.searchProvider}: ${(err as Error).message}`);
      trace(s, 'web_search', { query }, t0, { ok: false, reason: reason ?? 'search', error: e.message });
      throw e;
    }
    for (const r of results) if (!searchResults.has(normUrl(r.url))) searchResults.set(normUrl(r.url), r);
    trace(s, 'web_search', { query }, t0, {
      ok: true,
      reason: `${reason ?? 'search'} → ${results.length ? `${results.length} results` : 'no results'} · ${cacheNote}`
    });
    if (!results.length) return { content: 'No results.', results };
    const content = results
      .map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.snippet.replace(/\s+/g, ' ').slice(0, 300)}`)
      .join('\n');
    return { content, results };
  }

  async function runFetchPage(
    s: number,
    raw: unknown,
    timeoutMs = env.fetchTimeoutMs,
    batch?: AbortSignal
  ): Promise<{ content: string; isError: boolean }> {
    const t0 = Date.now();
    const parsed = FetchPageInput.safeParse(raw);
    const fail = (input: Record<string, unknown>, reason: string, error: string) => {
      // DESIGN: with nothing read yet, a failed read is a step toward the snippet fallback. Say so.
      const note = pages.length ? '' : '; no page read yet, so the answer falls back to search snippets if none succeeds';
      trace(s, 'fetch_page', input, t0, { ok: false, reason: `${reason}${note}`, error });
      return { content: `Error: ${error}. This page cannot be cited.`, isError: true };
    };
    if (!parsed.success) return fail(asInput(raw), 'rejected before fetching', 'invalid input: url must be a full url');
    const { url, reason = 'read' } = parsed.data;

    const known = pageByUrl.get(normUrl(url));
    if (known) {
      trace(s, 'fetch_page', { url }, t0, { ok: true, reason: `${reason} → already read as [${known.n}]` });
      return { content: `Already read as source [${known.n}].`, isError: false };
    }
    const hit = searchResults.get(normUrl(url));
    // Only what search surfaced in this request may be read: a guessed url is how an
    // invented page gets into an answer.
    if (!hit) return fail({ url }, reason, 'url was not returned by web_search in this request');

    let text: string;
    let fetchedTitle: string | undefined;
    fetchAttempts++;
    try {
      if (env.searchProvider === 'tavily') usage.extracts++;
      ({ text, title: fetchedTitle } = await fetchPage(hit.url, toolSignal(batch, timeoutMs)));
    } catch (err) {
      const cancelled = cutShort(batch);
      if (cancelled) {
        trace(s, 'fetch_page', { url }, t0, { ok: false, reason, error: cancelled });
        throw err;
      }
      const timedOut = err instanceof DOMException && err.name === 'TimeoutError';
      const error = timedOut ? `timed out after ${timeoutMs}ms` : err instanceof FetchError ? err.message : (err as Error).message;
      return fail({ url }, reason, error || 'fetch failed');
    }

    const sel = selectPassages(text, [ask.query, ...searchQueries].join(' '), PAGE_WORDS);
    if (!sel) return fail({ url }, reason, 'page had no readable passages');

    const page: Page = { n: pages.length + 1, url: hit.url, title: hit.title || fetchedTitle || hostOf(hit.url), snippet: sel.snippet };
    pages.push(page);
    pageByUrl.set(normUrl(url), page);
    trace(s, 'fetch_page', { url }, t0, {
      ok: true,
      reason: `${reason} → source [${page.n}], ${sel.passages.length} of ${sel.totalPassages} passages kept`
    });
    return {
      content: `Source [${page.n}]: ${page.title}\nURL: ${page.url}\n\n${sel.passages.join('\n\n')}`,
      isError: false
    };
  }

  /** Always the run's first step. A failure is a failed step, not a failed run: the answer goes on without memory, and says so in the trace. */
  async function runRecall(s: number, query: string, batch?: AbortSignal): Promise<{ memories: RecalledMemory[]; error?: string }> {
    const t0 = Date.now();
    try {
      const memories = await recallMemories(ask.userId, query, toolSignal(batch));
      trace(s, 'recall_memory', { query }, t0, {
        ok: true,
        reason: memories.length
          ? `${memories.length} memor${memories.length === 1 ? 'y' : 'ies'}: ${memories.map((m) => m.id).join(', ')}`
          : 'no saved memories'
      });
      return { memories };
    } catch (err) {
      const cancelled = cutShort(batch);
      if (cancelled) {
        trace(s, 'recall_memory', { query }, t0, { ok: false, reason: 'recall stopped before it finished', error: cancelled });
        throw err;
      }
      const error = (err as Error).message || 'recall failed';
      log.error({ requestId: ask.requestId, err }, 'recall_memory failed; answering without memory');
      trace(s, 'recall_memory', { query }, t0, { ok: false, reason: 'answering without saved memories', error });
      return { memories: [], error };
    }
  }

  async function runSaveMemory(s: number, raw: unknown, batch?: AbortSignal): Promise<{ content: string; isError: boolean }> {
    const t0 = Date.now();
    const parsed = SaveMemoryInput.safeParse(raw);
    if (!parsed.success) {
      const error = `invalid input: ${parsed.error.issues[0]?.message ?? 'bad text'}`;
      trace(s, 'save_memory', asInput(raw), t0, { ok: false, reason: 'rejected before saving', error });
      return { content: `Error: ${error}`, isError: true };
    }
    const { text, reason = 'remember' } = parsed.data;
    try {
      const r = await saveMemory(ask.userId, text, ask.threadId, toolSignal(batch));
      if (r.saved) {
        trace(s, 'save_memory', { text }, t0, { ok: true, reason: `${reason} → saved as ${r.id}` });
        return { content: `Saved (${r.id}). It applies to this user's future conversations.`, isError: false };
      }
      trace(s, 'save_memory', { text }, t0, { ok: true, reason: `${reason} → already remembered as ${r.duplicateOf.id}, not saved again` });
      return { content: `Already remembered: "${r.duplicateOf.text}". Nothing new was saved.`, isError: false };
    } catch (err) {
      const cancelled = cutShort(batch);
      if (cancelled) {
        trace(s, 'save_memory', { text }, t0, { ok: false, reason, error: cancelled });
        throw err;
      }
      const error = (err as Error).message || 'save failed';
      log.error({ requestId: ask.requestId, err }, 'save_memory failed');
      trace(s, 'save_memory', { text }, t0, { ok: false, reason, error });
      return { content: `Error: ${error}. Nothing was saved; tell the user it could not be remembered.`, isError: true };
    }
  }

  /** Runs one turn's tool calls in parallel. Slots are reserved in the model's order, before any await. */
  async function runTools(uses: Anthropic.ToolUseBlock[]): Promise<Anthropic.ToolResultBlockParam[]> {
    return settleBatch(
      uses.map((tu) => async (batch: AbortSignal): Promise<Anthropic.ToolResultBlockParam> => {
        const result = (content: string, isError = false): Anthropic.ToolResultBlockParam => ({
          type: 'tool_result',
          tool_use_id: tu.id,
          content,
          ...(isError ? { is_error: true } : {})
        });
        if (!QUICK_TOOL_NAMES.has(tu.name)) {
          log.warn({ requestId: ask.requestId, tool: tu.name }, 'quick run asked for a tool outside the quick toolbelt; refused');
          return result(`Error: ${tu.name} is not available on a quick search.`, true);
        }
        if (step >= env.maxToolCalls) {
          capped = true;
          return result(`Error: not run, the tool-call cap of ${env.maxToolCalls} is reached.`, true);
        }
        const tool = tu.name as ToolName;
        if (tool === streakTool && streak >= env.maxConsecutiveSameTool) {
          // A trace step, so it takes a slot under the tool-call cap like any other: the
          // model cannot dodge the cap by retrying into the guard.
          const s = ++step;
          const error =
            `refused: this would be ${tool} call ${streak + 1} in a row, and the limit is ` +
            `${env.maxConsecutiveSameTool} consecutive calls to one tool`;
          refused.push({ step: s, tool });
          sse.send('trace', { step: s, tool, input: asInput(tu.input), ok: false, ms: 0, reason: 'thrash guard, not run', error });
          return result(
            `Error: not run. ${error}. Answer now from what you already have, or use a different tool.`,
            true
          );
        }
        const s = reserve(tool);
        if (tu.name === 'web_search') return result((await runWebSearch(s, tu.input, batch)).content);
        const r =
          tu.name === 'save_memory'
            ? await runSaveMemory(s, tu.input, batch)
            : await runFetchPage(s, tu.input, env.fetchTimeoutMs, batch);
        return result(r.content, r.isError);
      })
    );
  }

  // ---------------------------------------------------------------- one model turn

  /**
   * One model turn, in one of two modes:
   *   auto    the model picks: more tools, or the answer.
   *   answer  tools are off. Used when the budget is spent or for the snippet fallback.
   *
   * `held` is text from a turn that ended with nothing citable, for the caller to decide on.
   */
  async function turn(mode: 'auto' | 'answer'): Promise<{ msg: Anthropic.Message; held: string }> {
    const stream = llm().messages.stream(
      {
        model: env.llmModel,
        max_tokens: MAX_TOKENS_PER_TURN,
        system: systemPrompt(),
        tools: QUICK_TOOLS,
        tool_choice: { type: mode === 'answer' ? 'none' : 'auto' },
        thinking: { type: 'adaptive' },
        output_config: { effort: 'low' },
        messages
      },
      { signal: abort.signal }
    );
    const forceAnswer = mode === 'answer';

    // Text is committed to the client only once it can only be the answer: when tools are
    // off, or when there are pages to cite, no tool call has appeared in this turn, and the
    // text already cites or is too long to be a preamble. Otherwise it is held, and
    // discarded if the turn turns out to be a tool call.
    let held = '';
    let sawToolUse = false;
    for await (const ev of stream) {
      if (ev.type === 'content_block_start' && ev.content_block.type === 'tool_use') sawToolUse = true;
      if (ev.type !== 'content_block_delta' || ev.delta.type !== 'text_delta') continue;
      if (sources) write(ev.delta.text);
      else {
        held += ev.delta.text;
        const answering = forceAnswer || /\[\d/.test(held) || held.length >= COMMIT_AFTER_CHARS;
        if (!sawToolUse && (forceAnswer || pages.length > 0) && answering) {
          beginAnswer(pageSources());
          write(held);
          held = '';
        }
      }
    }
    const msg = await stream.finalMessage();
    // A short answer that never reached the threshold is still an answer if no tool followed.
    if (!sources && held && pages.length > 0 && !msg.content.some((b) => b.type === 'tool_use')) {
      beginAnswer(pageSources());
      write(held);
      held = '';
    }
    usage.in += msg.usage.input_tokens + (msg.usage.cache_creation_input_tokens ?? 0) + (msg.usage.cache_read_input_tokens ?? 0);
    usage.out += msg.usage.output_tokens;
    if (msg.stop_reason === 'refusal') throw new ModelRefusal('llm provider: the model declined to answer (refusal)');
    return { msg, held };
  }

  // ---------------------------------------------------------------- the loop

  let terminated: Terminated = 'done';
  try {
    // ---- step one, no model: search the question as asked, read the top results. A
    // follow-up that points back ("does it…") is searched with the question before it.
    // Recall runs alongside it: neither needs the other, so it costs no time to first token.
    const withContext = firstSearchQuery(ask);
    const recallStep = reserve('recall_memory');
    const searchStep = reserve('web_search');
    const [recalled, first] = (await settleBatch<unknown>([
      (batch) => runRecall(recallStep, withContext ?? ask.query, batch),
      (batch) =>
        runWebSearch(
          searchStep,
          {
            query: withContext ?? ask.query.slice(0, 400),
            reason: withContext ? 'the follow-up, with the earlier question it refers back to' : 'the question as asked'
          },
          batch
        )
    ])) as [Awaited<ReturnType<typeof runRecall>>, Awaited<ReturnType<typeof runWebSearch>>];
    const top = first.results.slice(0, Math.max(0, Math.min(PREFETCH_PAGES, env.maxToolCalls - step)));
    // Reserved synchronously per result, so steps are numbered in rank order. These count
    // toward the thrash guard's streak like any fetch_page the model makes.
    const reads = await settleBatch(
      top.map((r) => {
        const s = reserve('fetch_page');
        return (batch: AbortSignal) =>
          runFetchPage(s, { url: r.url, reason: 'top result, read up front' }, env.prefetchTimeoutMs, batch);
      })
    );
    const read = reads.filter((r) => !r.isError).map((r) => r.content);
    const unread = top.flatMap((r, i) => (reads[i]?.isError ? [`${r.url} (${reads[i]!.content.replace(/^Error: /, '')})`] : []));
    messages.push({
      role: 'user',
      content: [
        ask.query,
        '---',
        memoryNote(recalled),
        `The question was searched ${withContext ? 'together with the earlier question it refers back to' : 'as asked'} (${step} of ${env.maxToolCalls} tool calls used, ${Math.max(0, env.maxToolCalls - step)} left).`,
        `Search results:\n${first.content}`,
        read.length ? `Pages read, cite them by their source number:\n\n${read.join('\n\n---\n\n')}` : '',
        unread.length ? `Not read in time (fetch_page can retry with a longer timeout):\n${unread.join('\n')}` : '',
        // Searched on its own words: right for a new topic, wrong for a follow-up the heuristic missed.
        ask.history.length && !withContext
          ? 'If this question continues the conversation and these results miss what it is about, search again with that spelled out.'
          : '',
        'If the pages answer the question, answer now. If not, search again or read more results.'
      ]
        .filter(Boolean)
        .join('\n\n')
    });

    let mode: 'auto' | 'answer' = 'auto';
    for (;;) {
      const { msg, held } = await turn(mode);
      if (sources) {
        if (msg.content.some((b) => b.type === 'tool_use')) {
          log.warn({ requestId: ask.requestId }, 'model called a tool after starting its answer; the call was not run');
        }
        break;
      }

      const uses = msg.content.filter((b): b is Anthropic.ToolUseBlock => b.type === 'tool_use');
      if (uses.length) {
        messages.push({ role: 'assistant', content: msg.content });
        const results = await runTools(uses);
        const content: Anthropic.ContentBlockParam[] = [...results];
        const timeLow = Date.now() > deadline - ANSWER_RESERVE_MS;
        mode = 'auto';
        if (capped || timeLow) {
          capped = true;
          mode = 'answer';
          content.push({
            type: 'text',
            text: `The research budget for this search is spent (${timeLow ? 'time' : `${env.maxToolCalls} tool calls`}). Write the answer now from the pages already read, and say in one sentence that the research stopped early.`
          });
        }
        messages.push({ role: 'user', content });
        continue;
      }

      // The model answered with nothing citable in hand.
      if (searchResults.size && !snippetFallback) {
        // Every page read failed, or the cap left no call to read one with. Answer from
        // the search snippets, numbered as sources. Out of calls is a cap, not a finish.
        if (!fetchAttempts && step >= env.maxToolCalls) capped = true;
        snippetFallback = true;
        const fallback = [...searchResults.values()].slice(0, 5);
        const list: Source[] = fallback
          .filter((r) => r.snippet)
          .map((r, i) => ({ n: i + 1, kind: 'web', title: r.title, url: r.url, snippet: r.snippet }));
        log.warn({ requestId: ask.requestId, sources: list.length }, 'no page could be read; answering from search snippets');
        messages.push({ role: 'assistant', content: msg.content });
        messages.push({
          role: 'user',
          content:
            `${capped ? 'The tool-call budget ran out before any page was read' : 'No page could be read'}, so answer from these search snippets instead, and say in one sentence that the answer rests on search snippets rather than full pages${capped ? ' because the research stopped early' : ''}. Cite them by these numbers:\n\n` +
            (list.map((s) => `[${s.n}] ${s.title}\n${s.url}\n${s.snippet}`).join('\n\n') || '(none of the results had a snippet)')
        });
        beginAnswer(list);
        mode = 'answer';
        continue;
      }
      // Nothing was retrieved at all: the answer says so and cites nothing.
      beginAnswer([]);
      write(held);
      break;
    }
    emit(filter?.flush() ?? '');
    if (capped) terminated = 'cap';
  } catch (err) {
    const reason: unknown = abort.signal.aborted ? abort.signal.reason : err;
    if (reason instanceof WallClockCap) {
      // Out of time mid-turn. Keep whatever answer exists, and say plainly that it stopped.
      terminated = 'cap';
      capped = true;
      beginAnswer(sources ?? pageSources());
      emit(filter?.flush() ?? '');
      emit(`${answerText ? '\n\n' : ''}The search stopped at its ${env.maxWallClockSec} s limit before the answer was finished.`);
    } else {
      terminated = 'error';
      const gone = reason instanceof ClientGone;
      const upstream = err instanceof Anthropic.APIError || err instanceof SearchProviderError || err instanceof ModelRefusal;
      const status = upstream || gone ? 502 : 500;
      const message = gone
        ? 'client disconnected'
        : err instanceof Anthropic.APIError
          ? `llm provider: ${err.message}`
          : (err as Error).message;
      clearTimeout(hardStop);
      await finish(status, { error: message }).catch(recordFailed);
      if (!gone) {
        log.error({ requestId: ask.requestId, err }, 'ask failed');
        sse.fail(status, message, ask.requestId);
      }
      return;
    }
  }

  clearTimeout(hardStop);
  const done = measure();
  // Saved before `done` goes out: an answer the thread lost would be invisible to the next
  // follow-up, so a failed save ends the run as an error instead of a quiet success.
  try {
    await ask.saveAnswer({ answerId, content: answerText, sources: sources ?? [], done });
  } catch (err) {
    terminated = 'error';
    const message = 'saving the answer to the thread failed';
    log.error({ requestId: ask.requestId, err }, message);
    await finish(502, { error: `${message}: ${(err as Error).message}` }).catch(recordFailed);
    sse.fail(502, message, ask.requestId);
    return;
  }
  // Recorded before `done` as well: a client that has seen `done` can see the answer in
  // /stats, and an answer with no run log is one the gates never grade, so it fails loud.
  try {
    await finish(200, {}, done);
  } catch (err) {
    recordFailed(err);
    sse.fail(502, 'recording the run log failed', ask.requestId);
    return;
  }
  sse.send('done', done);
  sse.end();

  // ---------------------------------------------------------------- accounting

  function recordFailed(err: unknown) {
    log.error({ requestId: ask.requestId, err }, 'recording the run log failed');
  }

  function measure(): DoneEvent {
    const latencyMs = Date.now() - started;
    const costUsd =
      (usage.in * env.llmInputUsdPerMtok + usage.out * env.llmOutputUsdPerMtok) / 1e6 +
      usage.searches * env.searchUsdPerCall +
      // Tavily bills extract at one credit per five pages, a fifth of a search.
      usage.extracts * (env.searchUsdPerCall / 5);
    return {
      answerId,
      latencyMs,
      ttftMs: ttftMs ?? latencyMs,
      model: env.llmModel,
      tokens: { in: usage.in, out: usage.out },
      costUsd: Math.round(costUsd * 1e6) / 1e6,
      // Only when every search in the request hit. No search at all is not a hit.
      searchCached: cache.searches > 0 && cache.hits === cache.searches,
      terminated,
      depth: 'quick',
      subQuestions: 0
    };
  }

  /** The answer's log line, then its run log and request record. `status` is what the answer ended with. */
  async function finish(status: number, extra: { error?: string } = {}, done = measure()): Promise<void> {
    const { latencyMs } = done;
    // The run log in the quality kit's shape: the same object is logged and stored.
    const runLog: RunLog = {
      tokens: usage.in + usage.out,
      wallClockSec: latencyMs / 1000,
      costUsd: done.costUsd,
      terminated,
      depth: 'quick',
      toolCalls: [...calls].sort((a, b) => a.step - b.step).map((c) => c.call)
    };
    log.info(
      {
        requestId: ask.requestId,
        userId: ask.userId,
        threadId: ask.threadId,
        answerId,
        toolCalls: runLog.toolCalls.length,
        refused,
        terminated,
        tokens: done.tokens,
        costUsd: done.costUsd,
        searchCached: done.searchCached,
        searchCache: cache,
        ttftMs: done.ttftMs,
        latencyMs,
        depth: 'quick',
        sources: sources?.length ?? 0,
        snippetFallback,
        droppedCitations: filter?.dropped ?? [],
        ...extra,
        runLog
      },
      'answer'
    );
    await recordAnswer({
      requestId: ask.requestId,
      userId: ask.userId,
      threadId: ask.threadId,
      answerId,
      query: ask.query,
      route: '/threads/:threadId/ask',
      status,
      run: runLog,
      tokensIn: done.tokens.in,
      tokensOut: done.tokens.out,
      ttftMs: done.ttftMs,
      searchCached: done.searchCached
    });
  }
}
