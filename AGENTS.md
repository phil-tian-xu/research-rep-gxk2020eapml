# Research Replication Project
## Background and Purpose
Phil is a Finance PhD student at CUHK working on empirical asset pricing, financial econometrics, machine learning, and quantitative finance.

This repository is for replicating a research paper: understanding its methods, implementing them, comparing results, and explaining differences.

Prioritize correctness, reproducibility, and clear code.

## Project-Specific Instructions
At the start of every turn involving this project, before answering or working, read `PROJECT_AGENTS.md` in the repository root. Read it again even if you read it in an earlier turn.

If it is missing or empty, continue normally.

That file provides the every detail related to the paper replication. Follow it over the general defaults below where they conflict. Follow the user’s explicit instructions over either file.

## Repository Structure
- `core/`: All research logic, including data preparation, models, and performance measures.
- `utils/`: General helpers unrelated to research, such as paths and file saving.
- `config/`: Settings for repeatable runs. Simple runs do not require a config file.
- `data/`: Source and processed datasets.
- `results/`: Generated outputs.
- `docs/`: The paper, research notes, and documented decisions.
- `materials/`: Exploratory files outside the reproducible workflow. Do not read or search this folder unless the user asks.
- `tests/`: Tests for important reusable logic.
- `main.py`: Workflow entry point, when needed.

Keep the structure simple. Add folders only when useful.

## SCRP Computing Cluster、
Phil uses CUHK Economics SCRP for workloads exceeding local resources.

Before planning or modifying cluster execution, read SCRP_AGENTS.md. Use the official SCRP documentation for cluster usage, and current account command output for actual quotas and availability.

## Research Standards
- Understand the relevant paper methods before implementing them. Separate what the paper specifies from our choices and unresolved questions.
- Preserve raw data. Make data processing and experiments reproducible, and use only information available at the relevant time.
- Validate important calculations and results. Investigate differences from the paper rather than changing methods just to match its numbers.
- Document meaningful assumptions, decisions, and deviations in `docs/`. Discuss unclear choices with the user when they could materially affect the research results.


## Coding Style
- Write simple, readable Python. Follow the existing project conventions and environment.
- Keep research logic in `core/` and general helpers in `utils/`. Avoid unnecessary abstractions, unrelated changes, and premature optimization.
- Use resources sensibly without sacrificing correctness. Test important reusable logic.
- Do not commit credentials, private data, or large generated files.

## Markdown Formatting
- Do not hard-wrap Markdown prose to a fixed line length (such as 80 or 120 characters). Keep each paragraph and each list item on a single source line, and let the Markdown reader wrap text for display.
- Use line breaks for document structure and meaningful formatting, including headings, separate paragraphs, list items, tables, and code blocks. Do not reflow existing documents solely to apply this preference unless Phil asks.

## Working Style
- Understand existing work before changing it. Stay within the requested scope.
- Use judgment for routine implementation details. Ask when missing information affects the research direction or interpretation.
- Explain important changes, checks, and limitations clearly. Do not claim results or successful checks without verifying them.
- Keep project-specific instructions and research notes current as the work develops.
- By default, provide drafts, code, and proposed changes in chat for the user to apply. Create, edit, or delete local files only when the user explicitly asks you to do so.
