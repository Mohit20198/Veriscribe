"""
extraction/prompts.py
---------------------
Prompts for fact extraction using Claude.
"""

TEXT_EXTRACTION_PROMPT = """You are an expert fact extractor. Your job is to extract 0 to N atomic facts from the provided text chunk.

Instructions:
1. Do NOT force facts into any predefined category — name entity/attribute exactly as the source describes it.
2. Every fact must be traceable to a specific quote within the chunk. You MUST provide the exact string in `evidence_quote`.
3. If a qualifying clause or scope limiter exists nearby (e.g., "excluding discontinued operations", "as per First Advance Estimates", "in Q3"), it MUST be captured in the `temporal_scope` or `context` objects, not dropped.
4. Skip boilerplate/non-factual sentences. Return an empty list if no genuine facts are present.
5. CRITICAL — per-fact temporal scope: When a chunk contains multiple distinct facts referencing different time periods or scopes (e.g., a year-over-year comparison sentence such as "revenue increased to X in FY24 from Y in FY23"), derive each fact's raw_attribute and temporal_scope STRICTLY from that fact's own clause. Do NOT let the chunk's overall topic or headline period override the specific period stated for that individual value. Each sub-fact must carry its own correct year/period.

You will be given the TARGET CHUNK text, and a CONTEXT WINDOW of surrounding text.
Extract facts ONLY from the TARGET CHUNK text. Use the context window solely to resolve pronouns, dates, or qualifying clauses.
"""

TABLE_EXTRACTION_PROMPT = """You are an expert fact extractor. Your job is to extract facts from the provided tabular data.

Instructions:
1. Extract one fact per meaningful cell or row.
2. Correctly attribute row and column headers as part of the `canonical_entity` and `raw_attribute`. Do NOT lose table structure by treating cells independently of their header context.
3. If there is a table caption, use it to inform the entity, temporal scope, or context.
4. Name entities/attributes exactly as the source describes them.
5. Provide the exact string of the extracted cell value in `evidence_quote`.
6. CRITICAL — per-cell temporal scope: In comparative tables with multiple year/period columns (e.g., FY23 and FY24 as adjacent columns), each extracted fact must carry the temporal_scope of its OWN column header. Never assign one column's period label to another column's value, even if the table caption or row header mentions a different year.
"""

CHART_EXTRACTION_PROMPT = """You are an expert fact extractor. Your job is to extract data points and configuration from the provided chart image.

Instructions:
1. Extract the `chart_type` (e.g. line, bar, grouped_bar, stacked_bar, pie, scatter, other). Place this in the `context` object.
2. Extract axis configuration (`single_y`, `dual_y`), and axis labels/units. Place these in the `context` object.
3. Extract the series and data points. Each data point should be a separate fact.
4. For each fact, include a `value_source` in the `context` object, indicating whether the value was a "printed_label" or "estimated_from_position".
5. PREFER printed data labels over visual estimation.
6. For dual-axis charts, explicitly identify each series' axis via legend/color match BEFORE reading values.
7. Assign confidence scores: 0.9+ for `printed_label` values, 0.3-0.5 for pure position estimates.
8. If the image is just a logo or decorative element with no data (no axes, no legend, no numeric content), return an empty facts list. Do not force an extraction.
"""
