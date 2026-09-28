# Add a New ADR (Architectural Decision Record)

> **When to use:** When making a significant architectural decision that affects
> multiple parts of the codebase, constrains future choices, or is likely to be
> questioned later.
>
> **Prerequisites:** None
>
> **Time estimate:** 15–30 minutes

## When to write an ADR

Write an ADR when:
- You choose between two or more viable approaches and the choice affects the API
- You add a constraint that future code must follow
- You reject a seemingly obvious approach for a non-obvious reason
- Someone (or an AI agent) might later ask "why didn't you just...?"

Don't write an ADR for:
- Routine implementation choices (which loop to use, variable names)
- Decisions that are easily reversible with no downstream impact

## Steps

### 1. Determine the next ADR number

```bash
ls docs/adr/
```

Current ADRs: 0001 through 0005. The next one would be `0006`.

### 2. Create the ADR file

Create `docs/adr/0006-<short-title>.md` using this template:

```markdown
# ADR 0006 — [Title]

**Status:** Accepted
**Date:** YYYY-MM-DD
**Author:** Parag Das

## Context

[What is the situation? What problem are you solving? What constraints exist?]

## Decision

[What did you decide to do?]

## Consequences

### Positive
- [What gets better]

### Negative
- [What gets harder or more complex]

### Neutral
- [Trade-offs that are neither clearly positive nor negative]

## Alternatives considered

### [Alternative 1]
[Why it was rejected]

### [Alternative 2]
[Why it was rejected]
```

### 3. Add the ADR to the index

Update the ADR table in:
- `docs/README.md` — the docs index
- `brain/DECISIONS.md` — the brain's decision log
- `README.md` — the project root README (if it references ADRs)

### 4. Commit

```bash
git add docs/adr/0006-<short-title>.md docs/README.md brain/DECISIONS.md
git commit -m "ADR 0006: <short-title>"
```

## Conventions

- **Never modify a past ADR's substance.** If a decision is superseded, add a new ADR
  that references the old one, and mark the old one's status as `Superseded by ADR-XXXX`.
- **Keep titles short and descriptive.** The title should make sense in a table without
  needing to open the file.
- **The "Alternatives considered" section is the most valuable part.** It answers
  "why didn't you just..." before the question is asked.
