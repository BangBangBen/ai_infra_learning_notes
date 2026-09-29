# Agent Instructions

## Markdown note compatibility

When creating or editing Markdown notes in this vault, ensure the notes render correctly in both Obsidian and GitHub. Prefer standard Markdown syntax supported by both platforms. Use standard Markdown links (`[text](path.md)`) instead of Obsidian-only wikilinks (`[[Note]]`) when linking between notes. For math that must render in both Obsidian and GitHub, use `$...$` for inline expressions and `$$...$$` for display expressions. Avoid `\(...\)` delimiters because they are not listed in GitHub Markdown math syntax. For named math operators, use `\mathrm{...}`; avoid `\operatorname{...}`, which GitHub has rejected in this vault. Avoid Obsidian-only embeds, callouts, or syntax unless an equivalent GitHub rendering is also provided. Check that headings, lists, tables, code fences, and links use portable Markdown conventions.

## Obsidian note update format

- Use the note `Serving_Engine/Chunked Prefill.md` as the structure and level-of-detail template for concept notes.
- For a blank concept note, add YAML frontmatter with `course`, `topic`, and `tags`; a descriptive H1 title; a concise `Summary` blockquote; numbered sections that explain the concept, execution or mental model, benefits and trade-offs, tuning or evaluation where relevant, boundaries with related techniques, and a one-sentence takeaway; and a `References` section with reliable primary papers or official documentation.
- For a populated concept note, preserve accurate, useful material while reorganizing it into the same readable structure. Fix malformed headings, code fences, tables, math, spacing, and incomplete or misleading statements. Do not discard meaningful examples.
- For paper notes, retain the paper title and identify the problem, core method, trade-offs, evaluation context, relationship to neighboring methods, takeaway, and a link to the primary paper. Clearly distinguish reported results from general guarantees.
- Do not invent benchmark results, implementation details, or source references. State uncertainty and version sensitivity where relevant; prefer primary papers and official project documentation.
- Keep the language concise but technically useful. Explain acronyms on first use, use tables or diagrams when they clarify trade-offs or execution, and avoid treating theoretical complexity as a performance guarantee.
- Use portable Markdown compatible with Obsidian and GitHub; use standard Markdown links, fenced code blocks with language labels when appropriate, and portable math delimiters.
- Preserve any directory or note family that explicitly documents a different format. In particular, do not reformat example/code notes in `Cookbooks` as concept notes.
- Never edit any note under a directory named `Notebooks` when applying bulk Obsidian-note updates, even if it otherwise matches the scope.

## Line endings

Use CRLF (`\r\n`) line endings for all text files in this repository. The root `.gitattributes` enforces CRLF in the working tree. When creating or editing text files, preserve CRLF line endings and ensure the file ends with CRLF.
