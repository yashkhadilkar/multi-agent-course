# Learner Progress

<!-- Claude reads this at the start of each session and updates it at the end.
     Learners: you don't need to touch this — Claude maintains it. -->

## Learner profile
- Name: [unset]
- Preferred learning style: [unset — set during /start: Socratic | Lecture+checkpoints | Build-along]
- Started: [date]
- Last session: 2026-09-30

## Module status

| Module | Status | Notes / weak spots |
|--------|--------|--------------------|
| 01 — Agent Foundations, Agent Harness & System Design | in progress | Building LUMINA. Done: quick loop, search cache, threads, memory, run logs, Spaces + jobs worker (GridFS → parse → chunk → embed → read-your-write probe → indexed; heartbeat, sweeper, resume, 3 attempts). Not yet: search_documents / hybrid RRF, deep search, gateway routes. |
| 02 — Skills & Subagents: Product Architecture & Coordination | not started | |
| 03 — Production Agentic RAG & AI Systems | not started | |
| 04 — Multi-Agent Systems & Orchestration | not started | |
| 05 — Real-Time Voice Agents & Conversational Systems | not started | |
| 06 — Leading AI Systems Across Teams | not started | |
| 07 — Demo Day (EPYHIA) | not started | |

Status values: not started · in progress · completed · needs review

## Weak spots to revisit
- [none yet]

## Next step
- LUMINA: hybrid retrieval (`search_documents`: $vectorSearch + $search fused with RRF, page-locator citations), then the gateway proxy so the UI can reach Spaces.
