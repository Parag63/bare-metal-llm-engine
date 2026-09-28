# brain/ — Persistent Project Context

> **What is this?** This folder is the *externalized memory* of the project. It stores
> conclusions, decisions, current state, and architectural rules in a format that any
> AI coding assistant (Claude, Gemini, Cursor, Copilot) — or a new human contributor —
> can read to get up to speed in minutes instead of hours.
>
> **Rule:** If you discover something important during development, write it down here.
> A closed chat window forgets everything; this folder does not.

## Quick orientation

| File | Purpose |
|---|---|
| [`PROJECT.md`](PROJECT.md) | **Start here.** Project identity, tech stack, constraints, coding style |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | High-level architecture, module map, data flow |
| [`DECISIONS.md`](DECISIONS.md) | Consolidated decision log — *why* things are the way they are |
| [`STATUS.md`](STATUS.md) | Current completion state, what works, what's next |
| [`CONVENTIONS.md`](CONVENTIONS.md) | Coding conventions, naming rules, patterns to follow |

## How to use this

### For AI agents
Read `PROJECT.md` first — it tells you what the project is, what the rules are, and
what you must never do. Then read `STATUS.md` to understand what's done and what's in
progress. Consult the other files as needed.

### For human developers
Browse in reading order: PROJECT → ARCHITECTURE → STATUS → CONVENTIONS. The DECISIONS
file is a reference you consult when you need to know *why* something works a specific
way.

### Maintenance rules
1. **Keep files focused.** Each file answers one kind of question.
2. **Update STATUS.md** whenever a milestone is completed.
3. **Never delete a past decision.** If a decision is superseded, mark it as such and
   add the new decision. The history of *what you believed and when* is part of the
   record.
4. **Markdown only.** AI agents parse Markdown natively. No PDFs, no Word docs, no
   wikis behind a login.
