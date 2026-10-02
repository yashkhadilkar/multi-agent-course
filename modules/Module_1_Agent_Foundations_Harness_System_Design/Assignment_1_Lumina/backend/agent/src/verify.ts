/**
 * Checking a web source's snippet against the page as a plain GET serves it.
 *
 * The reader (Tavily's extract) sees a page as rendered: text a script writes, a video's
 * transcript. A citation is checked against the plain HTML, where that text may not exist,
 * so a snippet cut from it reads as ungrounded although it was copied word for word from
 * what we read. So each page is also fetched plainly, in parallel with the read and under
 * the same signal, and the snippet is the first candidate that appears there:
 *
 *   plain fetch fails (403, 429, timeout, network)  → keep the best snippet: nobody can check it
 *   plain fetch works, a candidate appears in it     → that candidate is the snippet
 *   plain fetch works, no candidate appears          → drop the source: its text is not on the page
 *
 * "Appears" is the grounding check's own definition: the same normalization, the same tag
 * stripping, and a run of MATCH_TOKENS consecutive tokens. Kept in sync by hand with
 * benchmark/lib.mjs (normalize, snippetIsGrounded) and benchmark/bench.mjs (stripHtml), which
 * do not ship with the service.
 *
 * Document chunks need none of this: their snippets are cut from the chunk's own text, and
 * that text is what a document citation is checked against.
 */
import { fetchPage, type FetchedPage } from './search.js';
import { env } from './env.js';

/** Consecutive normalized tokens a snippet must share with the page (the grounding check's window). */
const MATCH_TOKENS = 12;
/** A page larger than this is checked on its first MAX_PLAIN_CHARS characters. */
const MAX_PLAIN_CHARS = 5 * 1024 * 1024;

/** benchmark/lib.mjs normalize: lowercase, curly quotes to straight, everything else not [a-z0-9'] to one space. */
export const normalizeText = (s: string): string =>
  s
    .toLowerCase()
    .replace(/[‘’“”]/g, "'")
    .replace(/[^a-z0-9']+/g, ' ')
    .trim();

/** benchmark/bench.mjs stripHtml: scripts and styles out, tags and entities to spaces. */
export const stripHtml = (html: string): string =>
  html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&[a-z#0-9]+;/gi, ' ');

/** benchmark/lib.mjs snippetIsGrounded, against an already normalized page. */
export function appearsIn(snippet: string, normalizedPage: string): boolean {
  const need = normalizeText(snippet).split(' ').filter(Boolean);
  if (!need.length || !normalizedPage) return false;
  if (need.length <= MATCH_TOKENS) return normalizedPage.includes(need.join(' '));
  for (let i = 0; i + MATCH_TOKENS <= need.length; i++) {
    if (normalizedPage.includes(need.slice(i, i + MATCH_TOKENS).join(' '))) return true;
  }
  return false;
}

/** The plain page, stripped and normalized, or why there is none. Never throws. */
export type PlainPage = { ok: true; text: string } | { ok: false; reason: string };

const plainOf = (html: string): PlainPage => ({ ok: true, text: normalizeText(stripHtml(html.slice(0, MAX_PLAIN_CHARS))) });

async function plainFetch(url: string, signal: AbortSignal): Promise<PlainPage> {
  try {
    const res = await fetch(url, {
      redirect: 'follow',
      headers: { 'user-agent': 'Mozilla/5.0 (compatible; lumina/0.1)', accept: 'text/html,text/plain,*/*' },
      signal
    });
    if (!res.ok) {
      void res.body?.cancel().catch(() => {});
      return { ok: false, reason: `plain fetch ${res.status}` };
    }
    return plainOf(await res.text());
  } catch (err) {
    const timedOut = signal.aborted && signal.reason instanceof DOMException && signal.reason.name === 'TimeoutError';
    return { ok: false, reason: timedOut ? 'plain fetch timed out' : `plain fetch failed: ${(err as Error).message || 'network error'}` };
  }
}

/**
 * Reads a page and fetches its plain HTML at the same time, under one signal: the read's
 * timeout is the plain fetch's too, so this takes no longer than the slower of the two
 * within it. A failed plain fetch never fails the read. A reader that already made a plain
 * GET (SerpApi mode) is checked against that response instead of fetching twice.
 */
export async function readWithPlain(url: string, signal: AbortSignal): Promise<FetchedPage & { plain: PlainPage }> {
  if (env.searchProvider !== 'tavily') {
    const page = await fetchPage(url, signal);
    return { ...page, plain: page.html !== undefined ? plainOf(page.html) : { ok: false, reason: 'no plain HTML from the reader' } };
  }
  const [page, plain] = await Promise.all([fetchPage(url, signal), plainFetch(url, signal)]);
  return { ...page, plain };
}

export type SnippetChoice =
  /** A candidate appears in the plain HTML. `rank` is its place among the candidates, 1 = best. */
  | { keep: true; snippet: string; checked: true; rank: number }
  /** No plain HTML to check against; the best candidate stands. */
  | { keep: true; snippet: string; checked: false; reason: string }
  /** The plain HTML was read and none of the candidates is in it. */
  | { keep: false; reason: string };

/**
 * Only candidates of at least MATCH_TOKENS tokens are tried when the page has any: a shorter
 * one matches as a whole string, so a page title or a caption would pass the check without
 * being a passage a claim could rest on.
 */
export function chooseSnippet(candidates: string[], plain: PlainPage): SnippetChoice {
  if (!plain.ok) return { keep: true, snippet: candidates[0]!, checked: false, reason: plain.reason };
  const long = candidates.filter((c) => normalizeText(c).split(' ').length >= MATCH_TOKENS);
  const tried = long.length ? long : candidates;
  const hit = tried.find((c) => appearsIn(c, plain.text));
  const i = hit === undefined ? -1 : candidates.indexOf(hit);
  if (i >= 0) return { keep: true, snippet: candidates[i]!, checked: true, rank: i + 1 };
  return {
    keep: false,
    reason: `none of ${tried.length} candidate snippets appears in the page's plain HTML (text rendered by script, or not on the page as served); not citable`
  };
}

/** How a kept choice reads in a trace step's reason. */
export const choiceNote = (c: Extract<SnippetChoice, { keep: true }>): string =>
  c.checked ? `snippet found in the plain HTML${c.rank > 1 ? ` (candidate ${c.rank})` : ''}` : `snippet unchecked: ${c.reason}`;
