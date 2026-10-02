/**
 * The two retrieval primitives behind web_search and fetch_page, one implementation per
 * SEARCH_PROVIDER so switching providers is an env change, not a code change.
 *
 *   tavily   search → POST /search     read → POST /extract (plain text)
 *   serpapi  search → GET /search.json read → fetch the page, @mozilla/readability + jsdom
 *
 * Failures throw. A provider that is down must look different from a query with no
 * results, so "no results" is an empty array and everything else is an exception.
 */
import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';
import { Agent, fetch as undiciFetch } from 'undici';
import { env, secrets } from './env.js';

export type SearchResult = { title: string; url: string; snippet: string };
/**
 * `html` is the page exactly as a plain GET served it, when the reader made one (SerpApi
 * mode). verify.ts checks snippets against it; Tavily's extract has none, so there the plain
 * fetch runs alongside it.
 */
export type FetchedPage = { text: string; title?: string; html?: string };

/** The search provider itself failed. Ends the run: the answer would have nothing under it. */
export class SearchProviderError extends Error {
  override name = 'SearchProviderError';
}

/** One page could not be read. The run carries on with the others. */
export class FetchError extends Error {
  override name = 'FetchError';
}

const MAX_RESULTS = 5;
const MAX_HTML_BYTES = 3 * 1024 * 1024;

export async function webSearch(query: string, signal: AbortSignal): Promise<SearchResult[]> {
  return env.searchProvider === 'tavily' ? tavilySearch(query, signal) : serpapiSearch(query, signal);
}

export async function fetchPage(url: string, signal: AbortSignal): Promise<FetchedPage> {
  return env.searchProvider === 'tavily' ? tavilyExtract(url, signal) : readPage(url, signal);
}

// ---------------------------------------------------------------- tavily

/**
 * Tavily's own connection pool. The global fetch drops an idle connection after 4 s, so the
 * first search of each new question paid a fresh TLS handshake; this one keeps it for
 * TAVILY_KEEPALIVE_MS. undici's fetch, not the global one, so the pool and the fetch that
 * uses it are always the same undici version.
 */
const tavilyPool = new Agent({ keepAliveTimeout: env.tavilyKeepAliveMs, keepAliveMaxTimeout: env.tavilyKeepAliveMs });

async function tavily(path: string, body: Record<string, unknown>, signal: AbortSignal): Promise<unknown> {
  if (!secrets.tavily) throw new SearchProviderError('TAVILY_API_KEY is not set');
  let res: Awaited<ReturnType<typeof undiciFetch>>;
  try {
    res = await undiciFetch(`https://api.tavily.com${path}`, {
      method: 'POST',
      headers: { authorization: `Bearer ${secrets.tavily}`, 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal,
      dispatcher: tavilyPool
    });
  } catch (err) {
    if (signal.aborted) throw err;
    throw new SearchProviderError(`tavily ${path}: ${(err as Error).message}`);
  }
  if (!res.ok) {
    const detail = (await res.text().catch(() => '')).slice(0, 200);
    throw new SearchProviderError(`tavily ${path} ${res.status}${detail ? `: ${detail}` : ''}`);
  }
  return res.json();
}

/**
 * Everything that affects Tavily's latency is pinned rather than left to its defaults:
 * the depth from TAVILY_SEARCH_DEPTH (basic unless set), no generated answer, no raw page
 * content (fetch_page reads pages), no images, and no auto_parameters (which may pick a
 * slower depth by itself). Measured over 20 bench questions: Tavily's own response_time
 * was p50 530 ms for fast against 1060 ms for basic, and the network adds ~0.1 s.
 */
const TAVILY_SEARCH = {
  search_depth: env.tavilySearchDepth,
  topic: 'general',
  max_results: MAX_RESULTS,
  include_answer: false,
  include_raw_content: false,
  include_images: false,
  auto_parameters: false
} as const;

async function tavilySearch(query: string, signal: AbortSignal): Promise<SearchResult[]> {
  const json = (await tavily('/search', { query, ...TAVILY_SEARCH }, signal)) as {
    results?: { title?: string; url?: string; content?: string }[];
  };
  return (json.results ?? []).flatMap((r) =>
    r.url ? [{ title: r.title?.trim() || hostOf(r.url), url: r.url, snippet: r.content?.trim() ?? '' }] : []
  );
}

async function tavilyExtract(url: string, signal: AbortSignal): Promise<FetchedPage> {
  let json: { results?: { raw_content?: string }[]; failed_results?: { error?: string }[] };
  try {
    json = (await tavily('/extract', { urls: [url], extract_depth: 'basic', format: 'text' }, signal)) as typeof json;
  } catch (err) {
    // Extract failing is one unreadable page, not a dead search provider.
    if (err instanceof SearchProviderError) throw new FetchError(err.message);
    throw err;
  }
  const text = json.results?.[0]?.raw_content?.trim();
  if (text) return { text };
  throw new FetchError(json.failed_results?.[0]?.error || 'tavily extract returned no content');
}

// ---------------------------------------------------------------- serpapi

async function serpapiSearch(query: string, signal: AbortSignal): Promise<SearchResult[]> {
  if (!secrets.serpapi) throw new SearchProviderError('SERPAPI_API_KEY is not set');
  const qs = new URLSearchParams({ engine: 'google', q: query, num: String(MAX_RESULTS), api_key: secrets.serpapi });
  let res: Response;
  try {
    res = await fetch(`https://serpapi.com/search.json?${qs}`, { signal });
  } catch (err) {
    if (signal.aborted) throw err;
    throw new SearchProviderError(`serpapi: ${(err as Error).message}`);
  }
  const json = (await res.json().catch(() => ({}))) as {
    error?: string;
    organic_results?: { title?: string; link?: string; snippet?: string }[];
  };
  // SerpApi reports an empty result page as an `error` string on a 200. That is an
  // answer ("nothing found"), not an outage, and must not end the run as one.
  if (res.ok && json.error && /hasn't returned any results/i.test(json.error)) return [];
  if (!res.ok || json.error) throw new SearchProviderError(`serpapi ${res.status}: ${json.error ?? 'request failed'}`);
  return (json.organic_results ?? [])
    .slice(0, MAX_RESULTS)
    .flatMap((r) => (r.link ? [{ title: r.title?.trim() || hostOf(r.link), url: r.link, snippet: r.snippet?.trim() ?? '' }] : []));
}

/** Fetch the page ourselves and keep the article text, one block per line. */
async function readPage(url: string, signal: AbortSignal): Promise<FetchedPage> {
  let res: Response;
  try {
    res = await fetch(url, {
      redirect: 'follow',
      headers: { 'user-agent': 'Mozilla/5.0 (compatible; lumina/0.1)', accept: 'text/html,text/plain' },
      signal
    });
  } catch (err) {
    if (signal.aborted && !(signal.reason instanceof DOMException && signal.reason.name === 'TimeoutError')) throw err;
    throw new FetchError(signal.aborted ? `timed out after ${env.fetchTimeoutMs}ms` : (err as Error).message);
  }
  if (!res.ok) throw new FetchError(`${res.status} from publisher`);
  const type = res.headers.get('content-type') ?? '';
  if (type.startsWith('text/plain')) {
    const plain = (await res.text()).slice(0, MAX_HTML_BYTES);
    return { text: plain, html: plain };
  }
  if (!type.includes('html')) throw new FetchError(`unsupported content-type ${type || 'unknown'}`);

  const html = (await res.text()).slice(0, MAX_HTML_BYTES);
  const article = new Readability(new JSDOM(html, { url }).window.document).parse();
  if (!article?.content) throw new FetchError('no readable article text');
  // Readability's textContent runs paragraphs together; break after each block first so
  // passages stay paragraph-shaped.
  const blocks = article.content.replace(/<\/(p|div|li|h[1-6]|pre|blockquote|tr|section|article)>/gi, '\n$&');
  const text = new JSDOM(blocks).window.document.body.textContent?.trim() ?? '';
  if (!text) throw new FetchError('no readable article text');
  return { text, title: article.title?.trim() || undefined, html };
}

export function hostOf(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}
