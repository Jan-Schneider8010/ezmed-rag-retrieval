Text chunk:
"""
{chunk_text}
"""

Produce {k} distinct, natural lay questions a patient could ask that THIS chunk answers. Rules:
- Only ask what the chunk actually answers; do not introduce facts, conditions, or treatments not present in the text.
- Avoid medical jargon; use the everyday language a non-expert would use.
- Cover different facets the chunk addresses (e.g. cause, symptom, diagnosis, treatment) rather than rephrasing a single point.

One question per line, no numbering, no extra commentary.
