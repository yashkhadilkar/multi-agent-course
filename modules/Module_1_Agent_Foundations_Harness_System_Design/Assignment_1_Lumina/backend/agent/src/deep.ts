/**
 * The DEEP gear: plan → research each sub-question → merge → one structured answer,
 * streamed as plan → trace* → sources → token* → done.
 *
 * The LLM is called twice (DESIGN.md, Trade-offs): once to plan, once to write. Between the
 * two, nothing is model-driven:
 *   1. plan_research and recall_memory run together. The plan event is sent the moment the
 *      plan exists, before any retrieval starts; recall's trace step is held until after it.
 *   2. Each sub-question searches with its own text (the web, the Space's documents, or both,
 *      by the same mode rules as quick) and reads its top results directly, a few
 *      sub-questions at a time.
 *   3. Sub-questions share one set of claimed urls: a result another sub-question already
 *      read is skipped for the next unclaimed one, so related sub-questions fan out to
 *      different pages instead of collapsing onto the same two.
 *   4. Everything read is merged into one numbering: web by url, documents by docId plus
 *      page (or heading / line), renumbered from 1 in plan order. Every source and every
 *      fan-out trace step carries the sub-question it served.
 *   5. One synthesis call writes a direct answer, a section per sub-question, then what is
 *      still unknown, under the same grounding rules as quick: snippets cut from the read
 *      text and found in the page's plain HTML (verify.ts), the source list final before the
 *      first token, unknown [n] dropped as it streams.
 *
 * Budget: MAX_TOOL_CALLS_DEEP calls on one shared counter. Planning and recall take two;
 * the rest are split evenly across the sub-questions. When the counter (or the research
 * deadline) runs out, no new call starts, the answer is written from what was collected,
 * and the run ends as `cap`.
 */
import Anthropic from '@anthropic-ai/sdk';
import type { Response } from 'express';
import type { Logger } from 'pino';
import {
  newId,
  type DoneEvent,
  type Locator,
  type RunLog,
  type Source,
  type SubQuestion,
  type Terminated,
  type ToolName,
  type TraceEvent
} from '@lumina/contract';
import { cachedSearch, isTimeSensitive } from './cache.js';
import { env } from './env.js';
import { recallMemories, type RecalledMemory } from './memory.js';
import { selectPassages, unwrap } from './passages.js';
import { llm, memoryNote, normUrl, retrievalFor, where, type QuickAsk, type Retrieval } from './quick.js';
import { DocumentSearchError, searchDocuments } from './retrieval.js';
import { recordAnswer } from './runlog.js';
import { FetchError, SearchProviderError, hostOf, type SearchResult } from './search.js';
import { CitationFilter, SseStream } from './sse.js';
import { chooseSnippet, choiceNote, readWithPlain, type PlainPage } from './verify.js';

/** Words of each page the synthesis reads. Smaller than quick's: a deep answer reads three or four times the pages. */
const PAGE_WORDS = 450;
/** plan_research and recall_memory, taken off the budget before it is split. */
const UP_FRONT_CALLS = 2;
/** No new research call starts this long before the wall-clock cap, so the answer still gets written. */
const ANSWER_RESERVE_MS = 60_000;
const PLAN_MAX_TOKENS = 600;
/**
 * The deep answer's length, in words: the prompt aims each section at a share of
 * ANSWER_TARGET_WORDS and states ANSWER_MAX_WORDS as the ceiling.
 */
const ANSWER_TARGET_WORDS = 520;
const ANSWER_MAX_WORDS = 650;
/** The direct answer and "What is still unknown" each get this many words; the sections share the rest. */
const OPENING_WORDS = 60;
const UNKNOWN_WORDS = 60;
/**
 * Output tokens per word of answer, citation markup and line breaks included. Measured at
 * about 2.55 on a deep answer dense with prices and figures (414 words, 1,057 tokens);
 * prose runs nearer 1.3. Sized for the dense case, with margin, so an answer inside the
 * word ceiling is never cut off. The synthesis call runs without thinking, so this limit is
 * spent on answer text alone. Hitting it ends the run as `cap`, with the answer marked as
 * cut off.
 */
const TOKENS_PER_WORD = 2.8;
const ANSWER_MAX_TOKENS = Math.ceil(ANSWER_MAX_WORDS * TOKENS_PER_WORD);
/** Search snippets per sub-question used as sources when not a single page could be read. */
const FALLBACK_PER_SUB = 2;

export type DeepAsk = QuickAsk;

// ---------------------------------------------------------------- plan_research

/**
 * The planner replies in a compact line format rather than JSON: on a plan of four to six
 * sub-questions, JSON's repeated keys and quoting were a third of the output tokens, and
 * output tokens are what the 4 s plan target is spent on. Parsed strictly (parsePlan).
 */
const PLAN_LINE = /^\s*(\d{1,2})[.)]\s*(.+?)\s*\|\|\s*(.+?)\s*$/;
const PLAN_REASON = /^\s*PLAN:\s*(.+?)\s*$/i;

/** The plan, or a PlanError saying what was wrong with it. Never a guessed plan. */
function parsePlan(text: string): { reason: string; questions: { question: string; reason: string }[] } {
  let reason = '';
  const questions: { question: string; reason: string }[] = [];
  for (const line of text.split('\n')) {
    const r = PLAN_REASON.exec(line);
    if (r) reason = r[1]!;
    const q = PLAN_LINE.exec(line);
    if (q && q[2] && q[3]) questions.push({ question: q[2].slice(0, 400), reason: q[3] });
  }
  if (!questions.length) throw new PlanError(`plan_research: the planner's reply had no sub-questions: ${JSON.stringify(text.slice(0, 200))}`);
  return { reason: reason || 'decomposed into sub-questions', questions };
}

function planSystem(): string {
  const today = new Date().toISOString().slice(0, 10);
  return `You plan research for LUMINA, a search assistant that answers from web pages and documents it reads. Today is ${today}.

Split the user's latest question into ${env.deepSubQuestionsMin} to ${env.deepSubQuestionsMax} sub-questions that, answered together, answer it fully.
- Each sub-question covers a different part: a different facet, side of a comparison, cost line, failure mode, or what changes at scale. No two should be answerable by the same search, and none restates the whole question.
- Write each as a standalone web search question of at most 10 words that names its subject. No "it", "they" or "this": if the question is a follow-up, resolve what it refers to from the conversation.
- Use 4 sub-questions. Use 3 only when the question has exactly three parts, and 5 or 6 only when it names that many separate parts.
- Each reason is at most 6 words: why this part matters to the question.
- The plan streams to the user before research starts, so keep it terse.

Reply with nothing but the plan, in exactly this form:
PLAN: <at most 10 words: how you split the question>
1. <sub-question> || <reason>
2. <sub-question> || <reason>
(and so on)`;
}

// ---------------------------------------------------------------- synthesis

function answerSystem(retrieval: Retrieval, subQuestions: number): string {
  const perSection = Math.floor((ANSWER_TARGET_WORDS - OPENING_WORDS - UNKNOWN_WORDS) / subQuestions);
  const today = new Date().toISOString().slice(0, 10);
  const from = retrieval === 'docs' ? "the user's documents" : retrieval === 'both' ? "the web and the user's documents" : 'the web';
  return `You are LUMINA in deep research mode. The user's question was split into sub-questions, each researched on ${from}, and you write the one answer from the passages that were read. Today is ${today}.

Structure, in plain text with no Markdown (no #, no **, no tables):
1. First, with no heading, the direct answer to the whole question in two or three sentences, at most ${OPENING_WORDS} words.
2. Then one section per sub-question, in plan order, each at most ${perSection} words. Each starts with a short heading of a few words on its own line, followed by a short paragraph or a few lines starting with "- ". Separate sections with a blank line.
3. Last, a section headed "What is still unknown", at most ${UNKNOWN_WORDS} words: what the sources did not settle, disagreed on, or did not cover, specific to this question. If a sub-question had nothing read for it, say so here.

Citations:
- Each source has a number. Cite a claim by putting its number in square brackets right after it, like this [2]. One number per bracket: [1][3], never [1, 3].
- Cite only the numbers given with the passages. A source found for one sub-question may support another section too.
- Make claims only from the passages. Where they do not answer something, say so plainly instead of guessing.

Length: about ${ANSWER_TARGET_WORDS} words in all, and never more than ${ANSWER_MAX_WORDS}. Stay inside each part's word limit: keep the findings that answer the sub-question and drop background, history and repetition. No padding, no restating the question, no closing summary.

Memory: what you know about the user comes with the question under "About this user". Follow their stated preferences. Memories are not sources: never cite them.

Earlier turns: if this is a follow-up, the conversation so far comes before the question. Earlier answers had their citation numbers removed, and the sources behind them cannot be cited now.`;
}

// ---------------------------------------------------------------- the run

/** Why the run was stopped from outside. */
class WallClockCap extends Error {}
class ClientGone extends Error {}
class ModelRefusal extends Error {}
/** The planner answered, but not with a usable plan. An upstream failure like any other. */
class PlanError extends Error {}

type WebRead = { kind: 'web'; url: string; title: string; snippet: string; passages: string[] };
type DocRead = { kind: 'doc'; key: string; docId: string; title: string; locator: Locator; snippet: string; text: string };
type Read = WebRead | DocRead;

/** What one sub-question turned up, in the order it will be numbered. */
type Research = { sub: SubQuestion; reads: Read[]; results: SearchResult[] };

const locatorKey = (l: Locator) => (l.page !== undefined ? `p${l.page}` : l.heading !== undefined ? `h${l.heading}` : `l${l.line}`);

export async function runDeep(ask: DeepAsk, res: Response, log: Logger): Promise<void> {
  const started = Date.now();
  const capMs = env.maxWallClockSecDeep * 1000;
  const researchDeadline = started + capMs - ANSWER_RESERVE_MS;
  const sse = new SseStream(res);
  const answerId = newId('ans');
  const space = ask.space;
  const retrieval = retrievalFor(ask.mode, Boolean(space));

  const abort = new AbortController();
  const hardStop = setTimeout(() => abort.abort(new WallClockCap()), capMs);
  res.on('close', () => {
    if (!res.writableFinished) abort.abort(new ClientGone('client disconnected'));
  });

  /** Pushed as calls finish. `sub` is 0 for the steps that serve the whole question. */
  const calls: { step: number; sub: number; call: RunLog['toolCalls'][number] }[] = [];
  const usage = { in: 0, out: 0, searches: 0, extracts: 0 };
  const cache = { searches: 0, hits: 0, readErrors: 0 };
  const freshQuestion = isTimeSensitive(ask.query);

  let plan = null as SubQuestion[] | null;
  let perSub = 0;
  let capped = false;
  let snippetFallback = false;
  let sources = null as Source[] | null;
  let filter = null as CitationFilter | null;
  let ttftMs = null as number | null;
  let answerText = '';
  let pagesRead = 0;
  const research: Research[] = [];
  /** Normalized urls some sub-question has already taken to read. Claimed synchronously, so two cannot take the same one. */
  const claimed = new Set<string>();

  // ---------------------------------------------------------------- output

  const beginAnswer = (list: Source[]) => {
    if (sources) return;
    sources = list;
    filter = new CitationFilter(new Set(list.map((s) => s.n)));
    sse.send('sources', list);
  };
  const emit = (text: string) => {
    if (!text) return;
    ttftMs ??= Date.now() - started;
    answerText += text;
    sse.send('token', { text });
  };
  const write = (delta: string) => emit(filter!.push(delta));

  // Trace steps finished before the plan is out are held, so `plan` is the first event.
  let planSent = false;
  const heldTraces: TraceEvent[] = [];

  const trace = (
    s: number,
    sub: number,
    tool: ToolName,
    input: Record<string, unknown>,
    t0: number,
    outcome: { ok: true; reason: string } | { ok: false; reason: string; error: string }
  ) => {
    const ms = Date.now() - t0;
    calls.push({ step: s, sub, call: outcome.ok ? { name: tool, ok: true, ms } : { name: tool, ok: false, error: outcome.error, ms } });
    const ev = {
      step: s,
      tool,
      input,
      ok: outcome.ok,
      ms,
      reason: outcome.reason,
      ...(outcome.ok ? {} : { error: outcome.error }),
      ...(sub ? { subQuestion: sub } : {})
    };
    if (planSent) sse.send('trace', ev);
    else heldTraces.push(ev);
  };

  // ---------------------------------------------------------------- the budget

  /** The one shared counter. Null when it is spent or research time is up: no new call starts. */
  let step = 0;
  const take = (): number | null => {
    if (step >= env.maxToolCallsDeep || Date.now() > researchDeadline) {
      capped = true;
      return null;
    }
    return ++step;
  };

  const toolSignal = (batch: AbortSignal | undefined, timeoutMs: number) =>
    AbortSignal.any([abort.signal, ...(batch ? [batch] : []), AbortSignal.timeout(timeoutMs)]);

  /** Why a call was stopped from outside, as its step's error; null when it failed on its own. */
  const cutShort = (batch: AbortSignal | undefined): string | null => {
    if (abort.signal.aborted) {
      const r: unknown = abort.signal.reason;
      return r instanceof WallClockCap
        ? `cancelled at the ${env.maxWallClockSecDeep} s wall-clock cap`
        : `cancelled: ${(r as Error | undefined)?.message || 'the run was aborted'}`;
    }
    return batch?.aborted ? String((batch.reason as Error).message) : null;
  };

  /**
   * Runs `items` through `fn`, at most `limit` at once. The first to throw cancels the rest,
   * and every one settles (tracing its step) before that first error ends the run.
   */
  async function settle<I, T>(items: I[], limit: number, fn: (item: I, batch: AbortSignal) => Promise<T>, parent?: AbortSignal): Promise<T[]> {
    const batch = new AbortController();
    const signal = parent ? AbortSignal.any([parent, batch.signal]) : batch.signal;
    let failure = null as { reason: unknown } | null;
    const out: T[] = new Array(items.length);
    let next = 0;
    const worker = async () => {
      while (next < items.length && !failure && !signal.aborted) {
        const i = next++;
        try {
          out[i] = await fn(items[i]!, signal);
        } catch (reason) {
          if (!failure) {
            failure = { reason };
            batch.abort(new Error(`cancelled because a parallel call failed: ${(reason as Error).message || 'unknown error'}`));
          }
        }
      }
    };
    await Promise.all(Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, worker));
    if (failure) throw failure.reason;
    signal.throwIfAborted();
    return out;
  }

  // ---------------------------------------------------------------- tools

  async function planResearch(s: number): Promise<{ subQuestions: SubQuestion[]; reason: string }> {
    const t0 = Date.now();
    const input = { query: ask.query };
    try {
      const msg = await llm().messages.create(
        {
          model: env.llmModel,
          max_tokens: PLAN_MAX_TOKENS,
          system: planSystem(),
          thinking: { type: 'disabled' },
          output_config: { effort: 'low' },
          messages: [...ask.history, { role: 'user', content: `Plan the research for this question:\n\n${ask.query}` }]
        },
        { signal: abort.signal }
      );
      usage.in += msg.usage.input_tokens + (msg.usage.cache_creation_input_tokens ?? 0) + (msg.usage.cache_read_input_tokens ?? 0);
      usage.out += msg.usage.output_tokens;
      if (msg.stop_reason === 'refusal') throw new ModelRefusal('llm provider: the model declined to plan (refusal)');
      const parsed = parsePlan(msg.content.flatMap((b) => (b.type === 'text' ? [b.text] : [])).join(''));
      const raw = parsed.questions;
      if (raw.length < env.deepSubQuestionsMin) {
        throw new PlanError(`plan_research: the planner returned ${raw.length} sub-questions, fewer than ${env.deepSubQuestionsMin}`);
      }
      const kept = raw.slice(0, env.deepSubQuestionsMax);
      const subQuestions = kept.map((q, i) => ({ i: i + 1, question: q.question, reason: q.reason }));
      const trimmed = raw.length > kept.length ? `, ${raw.length - kept.length} past the maximum of ${env.deepSubQuestionsMax} dropped` : '';
      trace(s, 0, 'plan_research', input, t0, { ok: true, reason: `${subQuestions.length} sub-questions${trimmed}: ${parsed.reason}` });
      return { subQuestions, reason: parsed.reason };
    } catch (err) {
      const cancelled = cutShort(undefined);
      const error =
        cancelled ?? (err instanceof Anthropic.APIError ? `llm provider: ${err.message}` : (err as Error).message || 'planning failed');
      trace(s, 0, 'plan_research', input, t0, { ok: false, reason: 'decompose the question', error });
      throw err;
    }
  }

  /** A failure is a failed step, not a failed run: the answer goes on without memory. */
  async function runRecall(s: number): Promise<{ memories: RecalledMemory[]; error?: string }> {
    const t0 = Date.now();
    const input = { query: ask.query };
    try {
      const memories = await recallMemories(ask.userId, ask.query, toolSignal(undefined, env.fetchTimeoutMs));
      trace(s, 0, 'recall_memory', input, t0, {
        ok: true,
        reason: memories.length
          ? `${memories.length} memor${memories.length === 1 ? 'y' : 'ies'}: ${memories.map((m) => m.id).join(', ')}`
          : 'no saved memories'
      });
      return { memories };
    } catch (err) {
      const cancelled = cutShort(undefined);
      if (cancelled) {
        trace(s, 0, 'recall_memory', input, t0, { ok: false, reason: 'recall stopped before it finished', error: cancelled });
        throw err;
      }
      const error = (err as Error).message || 'recall failed';
      log.error({ requestId: ask.requestId, err }, 'recall_memory failed; answering without memory');
      trace(s, 0, 'recall_memory', input, t0, { ok: false, reason: 'answering without saved memories', error });
      return { memories: [], error };
    }
  }

  async function runWebSearch(s: number, sub: SubQuestion, batch: AbortSignal): Promise<SearchResult[]> {
    const t0 = Date.now();
    const query = sub.question;
    const input = { query };
    let results: SearchResult[];
    let cacheNote: string;
    try {
      cache.searches++;
      const r = await cachedSearch(query, toolSignal(batch, env.fetchTimeoutMs), {
        bypass: freshQuestion || isTimeSensitive(query),
        log,
        requestId: ask.requestId
      });
      results = r.results;
      if (r.from === 'lru' || r.from === 'mongo') cache.hits++;
      else usage.searches++;
      if (r.cacheError) cache.readErrors++;
      cacheNote =
        r.from === 'bypass' ? 'time-sensitive, cache bypassed' : r.cacheError ? `${r.cacheError}, searched live` : r.from === 'miss' ? 'cache miss' : `cache hit (${r.from})`;
    } catch (err) {
      const cancelled = cutShort(batch);
      if (cancelled) {
        trace(s, sub.i, 'web_search', input, t0, { ok: false, reason: 'search the sub-question', error: cancelled });
        throw err;
      }
      // The provider is down: trace it, then end the run, as quick does.
      const e = err instanceof SearchProviderError ? err : new SearchProviderError(`${env.searchProvider}: ${(err as Error).message}`);
      trace(s, sub.i, 'web_search', input, t0, { ok: false, reason: 'search the sub-question', error: e.message });
      throw e;
    }
    trace(s, sub.i, 'web_search', input, t0, {
      ok: true,
      reason: `search the sub-question → ${results.length ? `${results.length} results` : 'no results'} · ${cacheNote}`
    });
    return results;
  }

  async function runFetch(s: number, sub: SubQuestion, hit: SearchResult, rank: number, batch: AbortSignal): Promise<WebRead | null> {
    const t0 = Date.now();
    const input = { url: hit.url };
    const reason = `result ${rank} of the sub-question's search, read directly`;
    const fail = (error: string) => {
      // As in quick: with nothing read anywhere yet, say where a run of failures leads.
      const note = pagesRead ? '' : '; no page read yet, so the answer falls back to search snippets if none succeeds';
      trace(s, sub.i, 'fetch_page', input, t0, { ok: false, reason: `${reason}${note}`, error });
      return null;
    };
    let text: string;
    let fetchedTitle: string | undefined;
    let plain: PlainPage;
    try {
      if (env.searchProvider === 'tavily') usage.extracts++;
      // The plain HTML comes back alongside the read, under the same timeout (verify.ts).
      ({ text, title: fetchedTitle, plain } = await readWithPlain(hit.url, toolSignal(batch, env.deepFetchTimeoutMs)));
    } catch (err) {
      const cancelled = cutShort(batch);
      if (cancelled) {
        trace(s, sub.i, 'fetch_page', input, t0, { ok: false, reason, error: cancelled });
        throw err;
      }
      const timedOut = err instanceof DOMException && err.name === 'TimeoutError';
      return fail(timedOut ? `timed out after ${env.deepFetchTimeoutMs}ms` : err instanceof FetchError ? err.message : (err as Error).message || 'fetch failed');
    }
    const sel = selectPassages(text, `${sub.question} ${ask.query}`, PAGE_WORDS);
    if (!sel) return fail('page had no readable passages');
    const choice = chooseSnippet(sel.candidates, plain);
    if (!choice.keep) return fail(choice.reason);
    pagesRead++;
    trace(s, sub.i, 'fetch_page', input, t0, {
      ok: true,
      reason: `${reason} → ${sel.passages.length} of ${sel.totalPassages} passages kept · ${choiceNote(choice)}`
    });
    return { kind: 'web', url: hit.url, title: hit.title || fetchedTitle || hostOf(hit.url), snippet: choice.snippet, passages: sel.passages };
  }

  /** Hybrid search over the Space. A failure ends the run, as in quick. */
  async function runSearchDocuments(s: number, sub: SubQuestion, batch: AbortSignal): Promise<DocRead[]> {
    const t0 = Date.now();
    const query = sub.question;
    const input = { query, spaceId: space!.id };
    const reason = retrieval === 'docs' ? "search the sub-question in the Space's documents" : 'search the sub-question in the documents alongside the web';
    let result: Awaited<ReturnType<typeof searchDocuments>>;
    try {
      result = await searchDocuments(ask.userId, space!.id, query, toolSignal(batch, env.fetchTimeoutMs));
    } catch (err) {
      const cancelled = cutShort(batch);
      if (cancelled) {
        trace(s, sub.i, 'search_documents', input, t0, { ok: false, reason, error: cancelled });
        throw err;
      }
      const e = err instanceof DocumentSearchError ? err : new DocumentSearchError(`document search: ${(err as Error).message}`);
      log.error({ requestId: ask.requestId, err: e.cause ?? err }, 'search_documents failed');
      trace(s, sub.i, 'search_documents', input, t0, { ok: false, reason, error: e.message });
      throw e;
    }
    const reads: DocRead[] = result.chunks.map((c) => ({
      kind: 'doc',
      key: `${c.docId}|${locatorKey(c.locator)}`,
      docId: c.docId,
      title: c.title,
      locator: c.locator,
      snippet: selectPassages(unwrap(c.text), `${sub.question} ${ask.query}`, PAGE_WORDS)?.snippet ?? c.text.replace(/\s+/g, ' ').trim(),
      text: c.text
    }));
    const fusedFrom = `RRF over ${result.vectorHits} vector + ${result.textHits} BM25 hits`;
    const droppedNote = result.dropped ? `, ${result.dropped} from documents not yet indexed dropped` : '';
    trace(s, sub.i, 'search_documents', input, t0, {
      ok: true,
      reason: reads.length
        ? `${reason} → ${reads.length} passages (${reads.map((r) => `${r.title} ${where(r.locator)}`).join('; ')}) · ${fusedFrom}${droppedNote}`
        : `${reason} → no matching passages · ${fusedFrom}${droppedNote}`
    });
    return reads;
  }

  /**
   * One sub-question: search with its own text, then read its top unclaimed results
   * directly, all inside its share of the budget. No LLM call. Fetches stop at the thrash
   * guard's length (rule A3), since a sub-question's reads sit together in the run log.
   */
  async function researchOne(sub: SubQuestion, batch: AbortSignal): Promise<Research> {
    let left = perSub;
    const spend = (): number | null => {
      if (left <= 0) return null;
      const s = take();
      if (s !== null) left--;
      return s;
    };
    type Search = { tool: 'web' | 'docs'; s: number };
    const searches: Search[] = [];
    if (retrieval !== 'docs') {
      const s = spend();
      if (s !== null) searches.push({ tool: 'web', s });
    }
    if (retrieval !== 'web') {
      const s = spend();
      if (s !== null) searches.push({ tool: 'docs', s });
    }
    const found = await settle(searches, searches.length, (x, b) =>
      x.tool === 'web' ? runWebSearch(x.s, sub, b).then((r) => ({ web: r, docs: [] as DocRead[] })) : runSearchDocuments(x.s, sub, b).then((d) => ({ web: [] as SearchResult[], docs: d })),
      batch
    );
    const results = found.flatMap((f) => f.web);
    const docs = found.flatMap((f) => f.docs);

    // Claim the top unclaimed results, in rank order, before any await: no other
    // sub-question can take one between the check and the claim.
    const want = Math.min(left, env.maxConsecutiveSameTool);
    const picks: { hit: SearchResult; rank: number; s: number }[] = [];
    for (const [i, hit] of results.entries()) {
      if (picks.length >= want) break;
      const key = normUrl(hit.url);
      if (claimed.has(key)) continue;
      const s = spend();
      if (s === null) break;
      claimed.add(key);
      picks.push({ hit, rank: i + 1, s });
    }
    const pages = await settle(picks, picks.length, (p, b) => runFetch(p.s, sub, p.hit, p.rank, b), batch);
    return { sub, results, reads: [...pages.filter((p): p is WebRead => p !== null), ...docs] };
  }

  // ---------------------------------------------------------------- merge

  /** Text the synthesis reads per source number. */
  const material = new Map<number, string[]>();
  /** Source numbers each sub-question turned up, including ones another found first. */
  const found = new Map<number, number[]>();

  function merge(): Source[] {
    const list: Source[] = [];
    const byKey = new Map<string, Source>();
    material.clear();
    found.clear();
    for (const r of [...research].sort((a, b) => a.sub.i - b.sub.i)) {
      const refs: number[] = [];
      for (const read of r.reads) {
        const key = read.kind === 'web' ? `u:${normUrl(read.url)}` : `d:${read.key}`;
        let src = byKey.get(key);
        if (!src) {
          src =
            read.kind === 'web'
              ? { n: list.length + 1, kind: 'web', title: read.title, url: read.url, snippet: read.snippet, subQuestion: r.sub.i }
              : { n: list.length + 1, kind: 'doc', title: read.title, docId: read.docId, locator: read.locator, snippet: read.snippet, subQuestion: r.sub.i };
          list.push(src);
          byKey.set(key, src);
          material.set(src.n, []);
        }
        // Several chunks of one page are one source; the synthesis reads all of them.
        const text = read.kind === 'web' ? read.passages.join('\n\n') : read.text;
        const m = material.get(src.n)!;
        if (!m.includes(text)) m.push(text);
        if (!refs.includes(src.n)) refs.push(src.n);
      }
      found.set(r.sub.i, refs);
    }
    if (list.length || !research.some((r) => r.results.length)) return list;

    // Not one page could be read: fall back to the search snippets, and say so.
    snippetFallback = true;
    const seen = new Set<string>();
    for (const r of [...research].sort((a, b) => a.sub.i - b.sub.i)) {
      const refs: number[] = [];
      for (const hit of r.results.filter((h) => h.snippet && !seen.has(normUrl(h.url))).slice(0, FALLBACK_PER_SUB)) {
        seen.add(normUrl(hit.url));
        const src: Source = { n: list.length + 1, kind: 'web', title: hit.title, url: hit.url, snippet: hit.snippet, subQuestion: r.sub.i };
        list.push(src);
        material.set(src.n, [hit.snippet]);
        refs.push(src.n);
      }
      found.set(r.sub.i, refs);
    }
    log.warn({ requestId: ask.requestId, sources: list.length }, 'deep: no page could be read; answering from search snippets');
    return list;
  }

  function synthesisPrompt(list: Source[], recalled: { memories: RecalledMemory[]; error?: string }): string {
    const subs = plan!.map((q) => {
      const refs = found.get(q.i) ?? [];
      const done = research.some((r) => r.sub.i === q.i);
      return `${q.i}. ${q.question}\n   Sources: ${refs.length ? refs.map((n) => `[${n}]`).join(' ') : done ? 'nothing could be read' : 'not researched, the budget ran out first'}`;
    });
    const blocks = list.map((s) => {
      const head = s.kind === 'web' ? `[${s.n}] ${s.title}\n${s.url}` : `[${s.n}] ${s.title}, ${where(s.locator!)}`;
      return `${head}\n\n${(material.get(s.n) ?? []).join('\n\n')}`;
    });
    return [
      ask.query,
      '---',
      memoryNote(recalled),
      `The research plan, with the sources read for each sub-question:\n${subs.join('\n')}`,
      snippetFallback
        ? 'No page could be read, so these sources are search-result snippets, not full pages. Say so in one sentence in the answer.'
        : '',
      capped ? 'The research budget ran out before everything planned was read. Say so under "What is still unknown".' : '',
      list.length ? `Sources:\n\n${blocks.join('\n\n---\n\n')}` : 'Nothing could be retrieved for any sub-question. Say so plainly and cite nothing.',
      'Write the answer now: the direct answer first, then a section per sub-question in plan order, then "What is still unknown".'
    ]
      .filter(Boolean)
      .join('\n\n');
  }

  // ---------------------------------------------------------------- the run

  let terminated: Terminated = 'done';
  try {
    const planStep = take()!;
    const recallStep = take()!;
    let recalled = { memories: [] as RecalledMemory[] } as { memories: RecalledMemory[]; error?: string };
    let planned = null as { subQuestions: SubQuestion[]; reason: string } | null;
    await settle([0, 1], 2, async (which) => {
      if (which === 0) planned = await planResearch(planStep);
      else recalled = await runRecall(recallStep);
    });
    plan = planned!.subQuestions;
    perSub = Math.floor((env.maxToolCallsDeep - UP_FRONT_CALLS) / plan.length);

    // The plan is deep search's first paint, and it goes out before any retrieval starts.
    sse.send('plan', { subQuestions: plan, reason: planned!.reason });
    planSent = true;
    for (const ev of heldTraces.sort((a, b) => a.step - b.step)) sse.send('trace', ev);

    await settle(plan, env.deepConcurrency, async (sub, batch) => {
      research.push(await researchOne(sub, batch));
    });

    beginAnswer(merge());
    const stream = llm().messages.stream(
      {
        model: env.llmModel,
        max_tokens: ANSWER_MAX_TOKENS,
        system: answerSystem(retrieval, plan.length),
        // No thinking: the plan and the passages already do the reasoning's work, and with
        // thinking on, its tokens would share ANSWER_MAX_TOKENS with the answer text.
        thinking: { type: 'disabled' },
        output_config: { effort: 'low' },
        messages: [...ask.history, { role: 'user', content: synthesisPrompt(sources!, recalled) }]
      },
      { signal: abort.signal }
    );
    for await (const ev of stream) {
      if (ev.type === 'content_block_delta' && ev.delta.type === 'text_delta') write(ev.delta.text);
    }
    const msg = await stream.finalMessage();
    usage.in += msg.usage.input_tokens + (msg.usage.cache_creation_input_tokens ?? 0) + (msg.usage.cache_read_input_tokens ?? 0);
    usage.out += msg.usage.output_tokens;
    if (msg.stop_reason === 'refusal') throw new ModelRefusal('llm provider: the model declined to answer (refusal)');
    emit(filter!.flush());
    if (msg.stop_reason === 'max_tokens') {
      // Cut off mid-answer is not a finished answer.
      capped = true;
      emit('\n\nThe answer was cut off at its length limit.');
    }
    if (capped) terminated = 'cap';
  } catch (err) {
    const reason: unknown = abort.signal.aborted ? abort.signal.reason : err;
    if (reason instanceof WallClockCap) {
      terminated = 'cap';
      capped = true;
      beginAnswer(plan ? merge() : []);
      emit(filter?.flush() ?? '');
      emit(`${answerText ? '\n\n' : ''}The deep search stopped at its ${env.maxWallClockSecDeep} s limit before the answer was finished.`);
    } else {
      terminated = 'error';
      const gone = reason instanceof ClientGone;
      const upstream =
        err instanceof Anthropic.APIError ||
        err instanceof SearchProviderError ||
        err instanceof DocumentSearchError ||
        err instanceof ModelRefusal ||
        err instanceof PlanError;
      const status = upstream || gone ? 502 : 500;
      const message = gone ? 'client disconnected' : err instanceof Anthropic.APIError ? `llm provider: ${err.message}` : (err as Error).message;
      clearTimeout(hardStop);
      await finish(status, { error: message }).catch(recordFailed);
      if (!gone) {
        log.error({ requestId: ask.requestId, err }, 'deep ask failed');
        sse.fail(status, message, ask.requestId);
      }
      return;
    }
  }

  clearTimeout(hardStop);
  const done = measure();
  try {
    await ask.saveAnswer({ answerId, content: answerText, sources: sources ?? [], done, ...(plan ? { subQuestions: plan } : {}) });
  } catch (err) {
    terminated = 'error';
    const message = 'saving the answer to the thread failed';
    log.error({ requestId: ask.requestId, err }, message);
    await finish(502, { error: `${message}: ${(err as Error).message}` }).catch(recordFailed);
    sse.fail(502, message, ask.requestId);
    return;
  }
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
      usage.extracts * (env.searchUsdPerCall / 5);
    return {
      answerId,
      latencyMs,
      ttftMs: ttftMs ?? latencyMs,
      model: env.llmModel,
      tokens: { in: usage.in, out: usage.out },
      costUsd: Math.round(costUsd * 1e6) / 1e6,
      searchCached: cache.searches > 0 && cache.hits === cache.searches,
      terminated,
      depth: 'deep',
      subQuestions: plan?.length ?? 0
    };
  }

  async function finish(status: number, extra: { error?: string } = {}, done = measure()): Promise<void> {
    const { latencyMs } = done;
    // Sub-questions run in parallel, so their steps interleave in time. The run log lists
    // the whole-question steps first, then each sub-question's steps together, in step
    // order within each: every sub-question's trajectory reads as it ran.
    const runLog: RunLog = {
      tokens: usage.in + usage.out,
      wallClockSec: latencyMs / 1000,
      costUsd: done.costUsd,
      terminated,
      depth: 'deep',
      toolCalls: [...calls].sort((a, b) => a.sub - b.sub || a.step - b.step).map((c) => c.call)
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
        terminated,
        tokens: done.tokens,
        costUsd: done.costUsd,
        searchCached: done.searchCached,
        searchCache: cache,
        ttftMs: done.ttftMs,
        latencyMs,
        depth: 'deep',
        subQuestions: plan?.length ?? 0,
        callsPerSubQuestion: perSub,
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
