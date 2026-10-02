/**
 * The QUICK gear: plan → choose tool → observe → repeat → answer, with web_search and
 * fetch_page, streamed as trace* → sources → token* → done.
 *
 * How grounding is kept, in order:
 *   1. Only a page fetch_page actually read becomes a source. It is numbered at the moment
 *      it is read, and the model sees that number next to the passages it may cite.
 *   2. The source's snippet is cut from the fetched text (passages.ts). The model never
 *      writes it. It must also appear in the page's plain HTML, fetched alongside the read
 *      (verify.ts); a page whose text is not there is dropped, not cited.
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
 * Documents (retrieval.ts): docs mode runs search_documents in place of the first web
 * search and has no web tools; auto with a Space runs both searches in that first step and
 * gives the model both toolsets, so it decides; auto without a Space is web only. A chunk
 * is a source the moment search_documents returns it, numbered in the same sequence as
 * pages, with its snippet cut from the chunk's own text.
 *
 * The final answer is the model's last turn, streamed as it is written. Text the model
 * writes before a tool call (a preamble) is held back until the turn shows whether it is
 * an answer, and discarded if it is not.
 */
import Anthropic from '@anthropic-ai/sdk';
import type { Response } from 'express';
import type { Logger } from 'pino';
import { z } from 'zod';
import {
  newId,
  type AskMode,
  type DoneEvent,
  type Locator,
  type RunLog,
  type Source,
  type Terminated,
  type ToolName
} from '@lumina/contract';
import { cachedSearch, isTimeSensitive } from './cache.js';
import { env, secrets } from './env.js';
import { MAX_MEMORY_CHARS, recallMemories, saveMemory, type RecalledMemory } from './memory.js';
import { selectPassages, unwrap } from './passages.js';
import { addMessageUsage, costUsd, emptyUsage, inputTokens, llmCostUsd, toolCostUsd } from './pricing.js';
import { DocumentSearchError, searchDocuments, type SpaceContents } from './retrieval.js';
import { recordAnswer } from './runlog.js';
import { FetchError, SearchProviderError, hostOf, type SearchResult } from './search.js';
import { CitationFilter, SseStream } from './sse.js';
import type { SavedAnswer } from './threads.js';
import { chooseSnippet, choiceNote, readWithPlain, type PlainPage } from './verify.js';

/**
 * Words of each fetched page the model reads: its most relevant passages. Every word is input
 * on the path to the first token, and an answer runs about 150 words.
 */
const PAGE_WORDS = 500;
/** Research stops starting new turns this long before the wall-clock cap, to leave room to answer. */
const ANSWER_RESERVE_MS = 15_000;
/** Thinking counts against this too. It bounds the cost of any one turn. */
const MAX_TOKENS_PER_TURN = 2048;
/** How many of the first search's results are read before the model is called. */
const PREFETCH_PAGES = 3;
/**
 * The up-front reads stop waiting once this many pages are in and citable, giving the rest
 * PREFETCH_GRACE_MS more. Waiting for the slowest of three put that one publisher on the path
 * to the first token; a read it stops is traced as failed, and fetch_page can retry it.
 */
const PREFETCH_ENOUGH = 2;
const PREFETCH_GRACE_MS = 200;
/**
 * With pages in hand the model may answer at once, or open with a preamble and then call a
 * tool. Its text is committed as the answer once it cites a source or runs this long.
 */
const COMMIT_AFTER_CHARS = 160;
/**
 * The cost guard's estimates (extraRoundFits). Output per call: a quick answer runs 100 to
 * 600 tokens, a tool call 200 to 250. Tool results: a search's five results, a page's
 * PAGE_WORDS, search_documents' top five chunks.
 */
const EST_OUTPUT_TOKENS = 600;
const EST_RESULT_TOKENS: Record<string, number> = { web_search: 600, fetch_page: 800, search_documents: 1500, save_memory: 50 };

// ---------------------------------------------------------------- tools

/**
 * The quick toolbelt, cut down per run to what the mode may search (toolbelt below).
 * plan_research is never in it, and dispatch refuses anything not in the run's belt: a
 * prompt asking the model not to escalate is a suggestion, this is the gate.
 */
const WEB_TOOLS: Anthropic.Tool[] = [
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
  }
];

const SEARCH_DOCUMENTS_TOOL: Anthropic.Tool = {
  name: 'search_documents',
  description:
    "Search the documents in the user's selected Space. Returns the best-matching passages, each with its source " +
    'number for citing and where it sits in its document (page, heading or line). Only passages this tool returned ' +
    'can be cited as documents.',
  input_schema: {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'What to look for, in words the documents are likely to use.' },
      reason: { type: 'string', description: 'A short phrase: why this search.' }
    },
    required: ['query'],
    additionalProperties: false
  },
  eager_input_streaming: true
};

const MEMORY_TOOLS: Anthropic.Tool[] = [
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

/**
 * What a run searches (DESIGN.md, Responsibilities): docs mode only the Space; auto with a
 * Space both, and the model decides which the question needs; everything else the web.
 */
export type Retrieval = 'web' | 'docs' | 'both';

export function retrievalFor(mode: AskMode, hasSpace: boolean): Retrieval {
  if (mode === 'docs') return 'docs';
  return mode === 'auto' && hasSpace ? 'both' : 'web';
}

function toolbelt(retrieval: Retrieval): Anthropic.Tool[] {
  return [
    ...(retrieval === 'docs' ? [] : WEB_TOOLS),
    ...(retrieval === 'web' ? [] : [SEARCH_DOCUMENTS_TOOL]),
    ...MEMORY_TOOLS
  ];
}

// With eager input streaming the API no longer validates tool inputs, so we do.
const WebSearchInput = z.object({ query: z.string().trim().min(1).max(400), reason: z.string().optional() });
const FetchPageInput = z.object({ url: z.string().url(), reason: z.string().optional() });
const SearchDocumentsInput = z.object({ query: z.string().trim().min(1).max(400), reason: z.string().optional() });
const SaveMemoryInput = z.object({ text: z.string().trim().min(1).max(MAX_MEMORY_CHARS), reason: z.string().optional() });

/** The Space's documents listed in the prompt, at most. */
const PROMPT_DOCUMENTS = 50;

/** Where a chunk sits in its document, as the model and the trace read it. */
export function where(l: Locator): string {
  if (l.page !== undefined) return `p. ${l.page}`;
  if (l.heading !== undefined) return `"${l.heading}"${l.line !== undefined ? `, line ${l.line}` : ''}`;
  return `line ${l.line}`;
}

/**
 * DESIGN.md: when a Space is selected, the prompt names its documents, so a question about
 * what they cover goes to search_documents instead of the web.
 */
function spaceNote(space: SpaceContents): string {
  const listed = space.indexed.slice(0, PROMPT_DOCUMENTS).map((d) => `- ${d.title}${d.pages ? ` (${d.pages} pages)` : ''}`);
  const more = space.indexed.length - listed.length;
  return [
    `The user's selected Space is "${space.name}". Its searchable documents:`,
    listed.length ? listed.join('\n') : '- (none indexed yet)',
    more > 0 ? `- …and ${more} more` : '',
    space.indexing.length ? `Still being indexed, not searchable yet: ${space.indexing.join(', ')}.` : '',
    'A question about what these documents cover is answered from them, with search_documents.'
  ]
    .filter(Boolean)
    .join('\n');
}

function researchNote(retrieval: Retrieval): string {
  if (retrieval === 'docs') {
    return `- This search is over the user's documents only; there is no web search. Before your first turn the question has already been searched in them once, as asked, and the best-matching passages come with the question.
- If those passages answer the question, answer straight away. If not, call search_documents again with different wording.`;
  }
  if (retrieval === 'both') {
    return `- Before your first turn the question has already been searched once, as asked, both on the web (the top results read) and in the user's documents. Both sets of passages come with the question.
- Decide what the question needs: the documents, the web, or both when it asks how the documents relate to what is on the web. Use only the passages that answer it.
- If what you have answers the question, answer straight away. If not, use search_documents, web_search and fetch_page to look further. Make independent calls in parallel, in the same turn.`;
  }
  return `- Before your first turn the question has already been searched once, as asked, and the top results read. Their passages come with the question.
- If those passages answer the question, answer straight away. If not, use web_search and fetch_page to look further. Make independent calls in parallel, in the same turn.`;
}

function systemPrompt(retrieval: Retrieval, space: SpaceContents | null): string {
  const today = new Date().toISOString().slice(0, 10);
  const from = retrieval === 'docs' ? "the user's documents" : retrieval === 'both' ? "web pages and the user's documents" : 'web pages';
  return `You are LUMINA, a search assistant that answers only from ${from} it has actually read. Today is ${today}.
${space && retrieval !== 'web' ? `\n${spaceNote(space)}\n` : ''}
How to research:
${researchNote(retrieval)}
- A run has at most ${env.maxToolCalls} tool calls, including the ones already made for you; the question says how many are left.
- At most ${env.maxConsecutiveSameTool} calls in a row may use the same tool, counting the reads already made for you. A call past that is refused: answer from what you have, or use a different tool.
- Write no text before or between tool calls. The only text you write is the final answer.

Memory:
- What you know about the user from earlier conversations comes with the question, under "About this user". Follow their stated preferences in your answer. Memories are not sources: never cite them.
- When the user states a lasting preference or fact about themselves ("remember that…", "I always want…", "I work in…"), call save_memory with it as one short sentence, then briefly confirm. Do not save facts from pages, or details that only matter for this one question.

How to answer:
- Put the direct answer in the first sentence, then only what the question needs. Keep it under about 150 words unless the question asks for depth; no headings for a short answer.
- Write plain text, not Markdown. The answer is shown exactly as written, so Markdown symbols appear as literal characters: no asterisks or underscores for bold or italics, no # headings, no tables, no backticks. For a list, put each item on its own line starting with "- ".
- Each page and each document passage you read has a source number. Cite a claim by putting that number in square brackets right after it, like this [2]. One number per bracket: [1][3], never [1, 3].
- Cite only the source numbers given with what you read. Web search results are not sources.
- Make claims only from the passages you read. If they do not answer the question, say that plainly instead of guessing.

Earlier turns:
- If this is a follow-up, the conversation so far comes before the question. Use it to work out what the question refers to.
- Earlier answers had their citation numbers removed, and the sources behind them cannot be cited now. Cite only what was read for this question.`;
}

let anthropic: Anthropic | null = null;
export const llm = () => (anthropic ??= new Anthropic({ apiKey: secrets.anthropic, maxRetries: 1 }));

// ---------------------------------------------------------------- the run

export type QuickAsk = {
  requestId: string;
  userId: string;
  threadId: string;
  query: string;
  mode: AskMode;
  /** The selected Space and what is in it; null when none is selected or the mode is web. */
  space: SpaceContents | null;
  /** Earlier turns of the thread, already trimmed, as alternating user/assistant messages. */
  history: Anthropic.MessageParam[];
  /** How many leading messages of `history` every later ask renders the same: the cacheable part. */
  stableHistory: number;
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

export const normUrl = (u: string) => u.replace(/#.*$/, '').replace(/\/+$/, '');

/** The recalled memories as the model sees them, most relevant first. Empty when there are none. */
export function memoryNote({ memories, error }: { memories: RecalledMemory[]; error?: string }): string {
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

/** "Remember that…", "From now on…", "Going forward…", "Keep in mind…", "Don't forget…", "Note that…". */
const REMEMBER_OPENER =
  /^\s*(?:please\s+)?(?:remember\b|from now on\b|going forward\b|keep in mind\b|don'?t forget\b|do not forget\b|note that\b|for (?:all )?future (?:answers|questions|conversations)\b)/i;

/**
 * Whether a message only asks to remember something: it opens like a memory request and
 * asks no question. Such a message skips the up-front search, which would read three pages
 * about the preference itself. The model keeps its search tools, so a message that also
 * wants something looked up can still have it.
 */
export function onlyAsksToRemember(query: string): boolean {
  return REMEMBER_OPENER.test(query) && !query.includes('?');
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
  const space = ask.space;
  const retrieval = retrievalFor(ask.mode, Boolean(space));
  const tools = toolbelt(retrieval);
  const allowed = new Set(tools.map((t) => t.name));
  const system = systemPrompt(retrieval, space);

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
  /** Everything citable, web pages and document chunks, numbered in the order it was read. */
  const retrieved: Source[] = [];
  const pageByUrl = new Map<string, Page>();
  /** Pages read whose text is not in their plain HTML, by normalized url, with why. Never sources. */
  const notCitable = new Map<string, string>();
  const sourceByChunk = new Map<string, Source>();
  // Pushed as calls finish; the run log lists them by step, the order they were made.
  const calls: { step: number; call: RunLog['toolCalls'][number] }[] = [];
  /** Calls the thrash guard refused. Traced, but not in the run log, since they never ran. */
  const refused: { step: number; tool: ToolName }[] = [];
  // `searches` is provider calls, the ones that cost money; a cache hit is not one.
  const usage = emptyUsage();
  /** Set when the cost guard stopped a round of tools, with what it projected. */
  let costStop = null as { projectedUsd: number; tools: string[] } | null;
  /** Where the time before the first token went, in ms from the start of the run. Logged with the answer. */
  const timing = {
    firstStepMs: null as number | null,
    readsMs: null as number | null,
    turns: [] as {
      mode: 'auto' | 'answer';
      startMs: number;
      firstThinkingMs: number | null;
      firstTextMs: number | null;
      endMs: number;
      in: number;
      cacheRead: number;
      cacheWrite: number;
      out: number;
      stop: string | null;
    }[]
  };
  const since = () => Date.now() - started;
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

  const retrievedSources = (): Source[] => [...retrieved];

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
  async function settleBatch<T>(calls: ((batch: AbortSignal) => Promise<T>)[], parent?: AbortSignal): Promise<T[]> {
    const batch = new AbortController();
    const signal = parent ? AbortSignal.any([parent, batch.signal]) : batch.signal;
    let failure = null as { reason: unknown } | null;
    const settled = await Promise.allSettled(
      calls.map((call) =>
        call(signal).catch((reason: unknown) => {
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
      const note = retrieved.length ? '' : '; no page read yet, so the answer falls back to search snippets if none succeeds';
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

    // Read before and found not to be on the page as served: reading it again changes nothing.
    const dropped = notCitable.get(normUrl(url));
    if (dropped) return fail({ url }, reason, dropped);

    let text: string;
    let fetchedTitle: string | undefined;
    let plain: PlainPage;
    fetchAttempts++;
    try {
      if (env.searchProvider === 'tavily') usage.extracts++;
      // The plain HTML comes back alongside the read, under the same timeout (verify.ts).
      ({ text, title: fetchedTitle, plain } = await readWithPlain(hit.url, toolSignal(batch, timeoutMs)));
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
    // Stopped from outside after the read but before its plain-HTML check finished: the
    // check never ran, so the page is not kept as though it had been unverifiable.
    const stopped = cutShort(batch);
    if (stopped) {
      trace(s, 'fetch_page', { url }, t0, { ok: false, reason, error: stopped });
      throw (abort.signal.aborted ? abort.signal.reason : batch!.reason) as Error;
    }

    const sel = selectPassages(text, [ask.query, ...searchQueries].join(' '), PAGE_WORDS);
    if (!sel) return fail({ url }, reason, 'page had no readable passages');
    const choice = chooseSnippet(sel.candidates, plain);
    if (!choice.keep) {
      notCitable.set(normUrl(url), choice.reason);
      return fail({ url }, reason, choice.reason);
    }

    const page: Page = { n: retrieved.length + 1, url: hit.url, title: hit.title || fetchedTitle || hostOf(hit.url), snippet: choice.snippet };
    retrieved.push({ n: page.n, kind: 'web', title: page.title, url: page.url, snippet: page.snippet });
    pageByUrl.set(normUrl(url), page);
    trace(s, 'fetch_page', { url }, t0, {
      ok: true,
      reason: `${reason} → source [${page.n}], ${sel.passages.length} of ${sel.totalPassages} passages kept · ${choiceNote(choice)}`
    });
    return {
      content: `Source [${page.n}]: ${page.title}\nURL: ${page.url}\n\n${sel.passages.join('\n\n')}`,
      isError: false
    };
  }

  /**
   * Hybrid search over the selected Space. Each chunk becomes a source when it is first
   * returned, its snippet cut from the chunk's own text exactly as a web page's is cut from
   * the page; the model reads the whole chunk. A failure ends the run, like a dead search
   * provider: the alternative is answering a document question without the documents.
   */
  async function runSearchDocuments(
    s: number,
    raw: unknown,
    reasonDefault: string,
    batch?: AbortSignal
  ): Promise<{ content: string; isError: boolean; found: number }> {
    const t0 = Date.now();
    const parsed = SearchDocumentsInput.safeParse(raw);
    if (!parsed.success || !space) {
      const error = !space ? 'no Space is selected' : `invalid input: ${parsed.error!.issues[0]?.message ?? 'bad query'}`;
      trace(s, 'search_documents', asInput(raw), t0, { ok: false, reason: 'rejected before searching', error });
      return { content: `Error: ${error}`, isError: true, found: 0 };
    }
    const { query, reason = reasonDefault } = parsed.data;
    const input = { query, spaceId: space.id };
    let result: Awaited<ReturnType<typeof searchDocuments>>;
    try {
      result = await searchDocuments(ask.userId, space.id, query, toolSignal(batch));
    } catch (err) {
      const cancelled = cutShort(batch);
      if (cancelled) {
        trace(s, 'search_documents', input, t0, { ok: false, reason, error: cancelled });
        throw err;
      }
      const e = err instanceof DocumentSearchError ? err : new DocumentSearchError(`document search: ${(err as Error).message}`);
      log.error({ requestId: ask.requestId, err: e.cause ?? err }, 'search_documents failed');
      trace(s, 'search_documents', input, t0, { ok: false, reason, error: e.message });
      throw e;
    }

    const blocks = result.chunks.map((c) => {
      let src = sourceByChunk.get(c.id);
      if (!src) {
        const snippet = selectPassages(unwrap(c.text), `${ask.query} ${query}`, PAGE_WORDS)?.snippet ?? c.text.replace(/\s+/g, ' ').trim();
        src = { n: retrieved.length + 1, kind: 'doc', docId: c.docId, title: c.title, locator: c.locator, snippet };
        retrieved.push(src);
        sourceByChunk.set(c.id, src);
      }
      return { n: src.n, text: `Source [${src.n}]: ${c.title}, ${where(c.locator)}\n\n${c.text}` };
    });
    const fusedFrom = `RRF over ${result.vectorHits} vector + ${result.textHits} BM25 hits`;
    const droppedNote = result.dropped ? `, ${result.dropped} from documents not yet indexed dropped` : '';
    trace(s, 'search_documents', input, t0, {
      ok: true,
      reason: blocks.length
        ? `${reason} → ${blocks.length} passages, sources ${blocks.map((b) => `[${b.n}]`).join('')} · ${fusedFrom}${droppedNote}`
        : `${reason} → no matching passages · ${fusedFrom}${droppedNote}`
    });
    if (!blocks.length) {
      const empty = space.indexed.length ? 'No passage in the Space\'s documents matched.' : 'The Space has no indexed documents yet.';
      return { content: empty, isError: false, found: 0 };
    }
    return { content: blocks.map((b) => b.text).join('\n\n---\n\n'), isError: false, found: blocks.length };
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
        if (!allowed.has(tu.name)) {
          log.warn({ requestId: ask.requestId, tool: tu.name, retrieval }, 'quick run asked for a tool outside its toolbelt; refused');
          return result(`Error: ${tu.name} is not available on this search.`, true);
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
            : tu.name === 'search_documents'
              ? await runSearchDocuments(s, tu.input, 'search the documents', batch)
              : await runFetchPage(s, tu.input, env.fetchTimeoutMs, batch);
        return result(r.content, r.isError);
      })
    );
  }

  /**
   * The up-front reads of the first search's top results, in parallel. Steps are reserved in
   * rank order before any await. Once PREFETCH_ENOUGH pages are in and citable, the rest get
   * PREFETCH_GRACE_MS, then are stopped: each is traced as a failed step with why, and the
   * model is told it can retry. Anything else that stops them (the run, a failed sibling in
   * `parent`) still ends the run.
   */
  async function prefetch(top: SearchResult[], parent: AbortSignal): Promise<{ content: string; isError: boolean }[]> {
    const enough = new AbortController();
    const signal = AbortSignal.any([parent, enough.signal]);
    let kept = 0;
    let grace: NodeJS.Timeout | undefined;
    const stopRest = () =>
      enough.abort(new Error(`not waited for: ${PREFETCH_ENOUGH} pages were already read when it had not finished`));
    try {
      return await Promise.all(
        top.map((r) => {
          const s = reserve('fetch_page');
          return runFetchPage(s, { url: r.url, reason: 'top result, read up front' }, env.prefetchTimeoutMs, signal).then(
            (res) => {
              if (!res.isError && ++kept === PREFETCH_ENOUGH && kept < top.length) grace = setTimeout(stopRest, PREFETCH_GRACE_MS);
              return res;
            },
            (err: unknown) => {
              // Stopped because enough pages were in: a failed read, already traced. Anything else ends the run.
              if (parent.aborted || !enough.signal.aborted) throw err;
              return { content: `Error: ${(enough.signal.reason as Error).message}`, isError: true };
            }
          );
        })
      );
    } finally {
      clearTimeout(grace);
    }
  }

  // ---------------------------------------------------------------- prompt caching

  /**
   * Cache breakpoints, at most three, each where the prefix before it is reused by a later
   * request (Sonnet 5 caches a prefix of 1,024 tokens or more):
   *   1. the system prompt, which caches the tools with it: the same for every quick run in
   *      this mode, so after the first run in five minutes every call reads it at 0.1x;
   *   2. the end of the thread history that later asks render byte for byte the same
   *      (threads.ts `stable`), so each follow-up reads the thread so far and writes only
   *      the turn that just became stable;
   *   3. the end of the conversation, only from a run's second model call on, when a call
   *      in auto mode may still be followed by another.
   * The first call's own question message (search results and pages) is not cached: it is
   * reused only if the run needs a second call, about 3% of quick runs, and writing it costs
   * 1.25x on every run. Measured on Sonnet 5, a forced-answer call (tool_choice none) still
   * read 1 and 2; the cost guard prices it uncached anyway, so its estimate errs high.
   */
  const cachedSystem: Anthropic.TextBlockParam[] = [{ type: 'text', text: system, cache_control: { type: 'ephemeral' } }];

  function withBreakpoints(tail: boolean): Anthropic.MessageParam[] {
    const marks = new Set<number>();
    if (ask.stableHistory > 0) marks.add(ask.stableHistory - 1);
    if (tail) marks.add(messages.length - 1);
    return messages.map((m, i) => {
      if (!marks.has(i)) return m;
      const blocks = typeof m.content === 'string' ? [{ type: 'text' as const, text: m.content }] : m.content;
      const last = blocks.at(-1);
      if (!last || last.type === 'thinking' || last.type === 'redacted_thinking') return m;
      return { ...m, content: [...blocks.slice(0, -1), { ...last, cache_control: { type: 'ephemeral' } }] } as Anthropic.MessageParam;
    });
  }

  // ---------------------------------------------------------------- cost guard

  /**
   * Before a model call beyond the first, decides whether the round of tools the model just
   * asked for, the call that reads their results, and one more call to answer after that
   * (reserved, since that call may ask for tools again) can all fit inside the per-answer
   * budget. If not, the tools are not run and the model answers from what it already has.
   * Estimates err high: searches priced as cache misses, tool results at their usual size,
   * every call's output at EST_OUTPUT_TOKENS.
   */
  function extraRoundFits(uses: Anthropic.ToolUseBlock[], last: Anthropic.Message): { fits: boolean; projectedUsd: number } {
    const count = (name: string) => uses.filter((u) => u.name === name).length;
    const toolsUsd = toolCostUsd({
      searches: count('web_search'),
      extracts: env.searchProvider === 'tavily' ? count('fetch_page') : 0
    });
    const lastIn = last.usage.input_tokens + (last.usage.cache_creation_input_tokens ?? 0) + (last.usage.cache_read_input_tokens ?? 0);
    const cached = (last.usage.cache_read_input_tokens ?? 0) + (last.usage.cache_creation_input_tokens ?? 0);
    const nextIn = lastIn + last.usage.output_tokens + uses.reduce((n, u) => n + (EST_RESULT_TOKENS[u.name] ?? 1000), 0);
    // The next call reads what is cached and writes the rest (breakpoint 3); the reserved
    // answer call reads only the tools and system prompt, so it is priced uncached.
    const nextUsd = llmCostUsd({ in: 0, cacheRead: cached, cacheWrite: nextIn - cached, out: EST_OUTPUT_TOKENS });
    const answerUsd = llmCostUsd({ in: nextIn + EST_OUTPUT_TOKENS, cacheRead: 0, cacheWrite: 0, out: EST_OUTPUT_TOKENS });
    const projectedUsd = costUsd(usage) + toolsUsd + nextUsd + answerUsd;
    return { fits: projectedUsd <= env.maxCostPerAnswerUsd, projectedUsd: Math.round(projectedUsd * 1e6) / 1e6 };
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
    const startMs = since();
    let firstThinkingMs = null as number | null;
    let firstTextMs = null as number | null;
    const stream = llm().messages.stream(
      {
        model: env.llmModel,
        max_tokens: MAX_TOKENS_PER_TURN,
        system: cachedSystem,
        tools,
        tool_choice: { type: mode === 'answer' ? 'none' : 'auto' },
        thinking: { type: 'adaptive' },
        output_config: { effort: 'low' },
        // The tail is cached only once a run is already on a later call and may make another.
        messages: withBreakpoints(mode === 'auto' && timing.turns.length > 0)
      },
      { signal: abort.signal }
    );
    const forceAnswer = mode === 'answer';

    // Text is committed to the client only once it can only be the answer: when tools are
    // off, or when there is something to cite, no tool call has appeared in this turn, and
    // the text already cites or is too long to be a preamble. Otherwise it is held, and
    // discarded if the turn turns out to be a tool call.
    let held = '';
    let sawToolUse = false;
    for await (const ev of stream) {
      if (ev.type === 'content_block_start' && ev.content_block.type === 'tool_use') sawToolUse = true;
      if (ev.type === 'content_block_start' && ev.content_block.type === 'thinking') firstThinkingMs ??= since();
      if (ev.type !== 'content_block_delta' || ev.delta.type !== 'text_delta') continue;
      firstTextMs ??= since();
      if (sources) write(ev.delta.text);
      else {
        held += ev.delta.text;
        const answering = forceAnswer || /\[\d/.test(held) || held.length >= COMMIT_AFTER_CHARS;
        if (!sawToolUse && (forceAnswer || retrieved.length > 0) && answering) {
          beginAnswer(retrievedSources());
          write(held);
          held = '';
        }
      }
    }
    const msg = await stream.finalMessage();
    // A short answer that never reached the threshold is still an answer if no tool followed.
    if (!sources && held && retrieved.length > 0 && !msg.content.some((b) => b.type === 'tool_use')) {
      beginAnswer(retrievedSources());
      write(held);
      held = '';
    }
    addMessageUsage(usage, msg.usage);
    timing.turns.push({
      mode,
      startMs,
      firstThinkingMs,
      firstTextMs,
      endMs: since(),
      in: msg.usage.input_tokens,
      cacheRead: msg.usage.cache_read_input_tokens ?? 0,
      cacheWrite: msg.usage.cache_creation_input_tokens ?? 0,
      out: msg.usage.output_tokens,
      stop: msg.stop_reason
    });
    if (msg.stop_reason === 'refusal') throw new ModelRefusal('llm provider: the model declined to answer (refusal)');
    return { msg, held };
  }

  // ---------------------------------------------------------------- the loop

  let terminated: Terminated = 'done';
  try {
    // ---- step one, no model: search the question as asked, read the top results. A
    // follow-up that points back ("does it…") is searched with the question before it.
    // Docs mode searches the Space instead of the web; auto with a Space searches both, and
    // the model decides what to use. Recall runs alongside, and the reads start the moment
    // the web search returns, without waiting on recall or the document search: the first
    // token waits only for the slowest of the three branches.
    const withContext = firstSearchQuery(ask);
    const firstQuery = withContext ?? ask.query.slice(0, 400);
    const asAsked = withContext ? 'the follow-up, with the earlier question it refers back to' : 'the question as asked';
    // A message that only asks to remember something is not searched up front, on the web or
    // in the Space: the model saves it and confirms, and keeps its tools in case it asks more.
    const rememberOnly = onlyAsksToRemember(ask.query);
    const recallStep = reserve('recall_memory');
    const webStep = retrieval === 'docs' || rememberOnly ? null : reserve('web_search');
    const docsStep = retrieval === 'web' || rememberOnly ? null : reserve('search_documents');
    const docsReason =
      retrieval === 'docs'
        ? `docs mode: ${asAsked}, in the Space only`
        : `auto mode with a Space selected: ${asAsked}, in the documents alongside the web`;
    type FirstWeb = { found: Awaited<ReturnType<typeof runWebSearch>>; top: SearchResult[]; reads: { content: string; isError: boolean }[] };
    const [recalled, web, docs] = (await settleBatch<unknown>([
      (batch) => runRecall(recallStep, withContext ?? ask.query, batch),
      async (batch): Promise<FirstWeb | null> => {
        if (webStep === null) return null;
        const found = await runWebSearch(webStep, { query: firstQuery, reason: asAsked }, batch);
        timing.firstStepMs = since();
        // Reserved per result in prefetch, in rank order. These count toward the thrash
        // guard's streak like any fetch_page the model makes.
        const top = found.results.slice(0, Math.max(0, Math.min(PREFETCH_PAGES, env.maxToolCalls - step)));
        const reads = await prefetch(top, batch);
        timing.readsMs = since();
        return { found, top, reads };
      },
      async (batch) => (docsStep === null ? null : runSearchDocuments(docsStep, { query: firstQuery }, docsReason, batch))
    ])) as [Awaited<ReturnType<typeof runRecall>>, FirstWeb | null, Awaited<ReturnType<typeof runSearchDocuments>> | null];
    const first = web?.found ?? null;
    const top = web?.top ?? [];
    const reads = web?.reads ?? [];
    const read = reads.filter((r) => !r.isError).map((r) => r.content);
    const unread = top.flatMap((r, i) => (reads[i]?.isError ? [`${r.url} (${reads[i]!.content.replace(/^Error: /, '')})`] : []));
    const left = `${step} of ${env.maxToolCalls} tool calls used, ${Math.max(0, env.maxToolCalls - step)} left`;
    messages.push({
      role: 'user',
      content: rememberOnly
        ? [
            ask.query,
            '---',
            memoryNote(recalled),
            `Nothing was searched up front: this message reads as only asking you to remember something (${left}).`,
            'Save what it asks you to remember with save_memory, then confirm in a sentence. If it also asks for something to be looked up, search for that before answering.'
          ]
            .filter(Boolean)
            .join('\n\n')
        : [
            ask.query,
            '---',
            memoryNote(recalled),
            `The question was searched ${retrieval === 'docs' ? "in the user's documents " : retrieval === 'both' ? "on the web and in the user's documents " : ''}${withContext ? 'together with the earlier question it refers back to' : 'as asked'} (${left}).`,
            docs
              ? docs.found
                ? `Passages from the documents in "${space!.name}", cite them by their source number:\n\n${docs.content}`
                : `Document search: ${docs.content}`
              : '',
            first ? `Web search results:\n${first.content}` : '',
            read.length ? `Web pages read, cite them by their source number:\n\n${read.join('\n\n---\n\n')}` : '',
            unread.length ? `Not read in time (fetch_page can retry with a longer timeout):\n${unread.join('\n')}` : '',
            // Searched on its own words: right for a new topic, wrong for a follow-up the heuristic missed.
            ask.history.length && !withContext
              ? 'If this question continues the conversation and these results miss what it is about, search again with that spelled out.'
              : '',
            retrieval === 'both'
              ? 'Use the documents, the web, or both, whichever the question needs. If what you have answers it, answer now. If not, search again or read more.'
              : retrieval === 'docs'
                ? 'If the passages answer the question, answer now. If not, search the documents again with different wording.'
                : 'If the pages answer the question, answer now. If not, search again or read more results.'
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
        // The cost guard, before another model call is committed to.
        const guard = extraRoundFits(uses, msg);
        let results: Anthropic.ToolResultBlockParam[];
        if (guard.fits) results = await runTools(uses);
        else {
          capped = true;
          costStop = { projectedUsd: guard.projectedUsd, tools: uses.map((u) => u.name) };
          log.info({ requestId: ask.requestId, ...costStop, budgetUsd: env.maxCostPerAnswerUsd }, 'cost guard: answering without another round of tools');
          results = uses.map((u) => ({
            type: 'tool_result',
            tool_use_id: u.id,
            content: `Error: not run. Another round of research would take this answer past its $${env.maxCostPerAnswerUsd} cost budget.`,
            is_error: true
          }));
        }
        const content: Anthropic.ContentBlockParam[] = [...results];
        const timeLow = Date.now() > deadline - ANSWER_RESERVE_MS;
        mode = 'auto';
        if (capped || timeLow) {
          capped = true;
          mode = 'answer';
          const spent = timeLow ? 'time' : costStop ? 'cost' : `${env.maxToolCalls} tool calls`;
          content.push({
            type: 'text',
            text: `The research budget for this search is spent (${spent}). Write the answer now from what was already read, and say in one sentence that the research stopped early.`
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
      beginAnswer(sources ?? retrievedSources());
      emit(filter?.flush() ?? '');
      emit(`${answerText ? '\n\n' : ''}The search stopped at its ${env.maxWallClockSec} s limit before the answer was finished.`);
    } else {
      terminated = 'error';
      const gone = reason instanceof ClientGone;
      const upstream =
        err instanceof Anthropic.APIError ||
        err instanceof SearchProviderError ||
        err instanceof DocumentSearchError ||
        err instanceof ModelRefusal;
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
    return {
      answerId,
      latencyMs,
      ttftMs: ttftMs ?? latencyMs,
      model: env.llmModel,
      // Every input token, cached or not; cache reads and writes differ only in price.
      tokens: { in: inputTokens(usage), out: usage.out },
      costUsd: Math.round(costUsd(usage) * 1e6) / 1e6,
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
      tokens: inputTokens(usage) + usage.out,
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
        mode: ask.mode,
        retrieval,
        ...(space ? { spaceId: space.id } : {}),
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
        promptCache: { read: usage.cacheRead, write: usage.cacheWrite, uncached: usage.in },
        costStop,
        timing,
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
