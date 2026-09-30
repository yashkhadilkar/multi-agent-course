import type { Response } from 'express';
import type { z } from 'zod';
import {
  DoneEvent,
  PlanEvent,
  SourcesEvent,
  StreamErrorEvent,
  TokenEvent,
  TraceEvent,
  type SseEventName
} from '@lumina/contract';

const SCHEMAS = {
  plan: PlanEvent,
  trace: TraceEvent,
  sources: SourcesEvent,
  token: TokenEvent,
  done: DoneEvent,
  error: StreamErrorEvent
} satisfies Record<SseEventName, z.ZodTypeAny>;

type Payload<E extends SseEventName> = z.input<(typeof SCHEMAS)[E]>;

/**
 * One answer's event stream. Headers go out with the first event, not before, so a
 * failure that happens before anything was streamed can still be a real HTTP status
 * (502) instead of a 200 carrying an error event.
 *
 * Every event is validated against the contract before it is written: an event the UI
 * cannot parse is our bug, and it should throw here rather than render as nothing there.
 */
export class SseStream {
  private opened = false;

  constructor(private readonly res: Response) {}

  get started(): boolean {
    return this.opened;
  }

  /** The client went away; writing further is pointless. */
  get closed(): boolean {
    return this.res.destroyed || this.res.writableEnded;
  }

  send<E extends SseEventName>(event: E, data: Payload<E>): void {
    const body = SCHEMAS[event].parse(data);
    if (this.closed) return;
    this.open();
    // No compression on this service, so each write leaves as its own chunk.
    this.res.write(`event: ${event}\ndata: ${JSON.stringify(body)}\n\n`);
  }

  /** Ends the request with an error: a status if nothing was streamed, else an error event. */
  fail(status: number, error: string, requestId: string): void {
    if (this.closed) return;
    if (this.opened) {
      this.send('error', { status, error });
      this.res.end();
    } else {
      this.res.status(status).json({ error, status, requestId });
    }
  }

  end(): void {
    if (!this.closed) this.res.end();
  }

  private open(): void {
    if (this.opened) return;
    this.opened = true;
    this.res.status(200);
    this.res.setHeader('content-type', 'text/event-stream; charset=utf-8');
    this.res.setHeader('cache-control', 'no-cache, no-transform');
    this.res.setHeader('connection', 'keep-alive');
    this.res.setHeader('x-accel-buffering', 'no');
    this.res.flushHeaders();
    this.res.socket?.setNoDelay(true);
  }
}

/**
 * Drops every [n] that has no matching source, as the text streams. A citation can be
 * split across two deltas ("[1" then "2]"), so a tail that could still become one is held
 * back until the next delta decides it. "[1, 3]" is rewritten to "[1][3]", the form the
 * contract's citationNumbers() counts.
 */
export class CitationFilter {
  private held = '';
  readonly dropped: number[] = [];

  constructor(private readonly valid: ReadonlySet<number>) {}

  push(delta: string): string {
    const s = this.held + delta;
    const tail = /\[[\d,\s]{0,16}$/.exec(s);
    const cut = tail ? tail.index : s.length;
    this.held = s.slice(cut);
    return this.clean(s.slice(0, cut));
  }

  flush(): string {
    const s = this.held;
    this.held = '';
    return this.clean(s);
  }

  private clean(s: string): string {
    return s.replace(/(\s?)\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]/g, (_m, space: string, list: string) => {
      const ns = list.split(',').map((x) => Number(x.trim()));
      const keep = ns.filter((n) => this.valid.has(n));
      this.dropped.push(...ns.filter((n) => !this.valid.has(n)));
      return keep.length ? space + keep.map((n) => `[${n}]`).join('') : '';
    });
  }
}
