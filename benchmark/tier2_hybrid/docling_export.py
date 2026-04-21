"""
Shared helper: serialize a DoclingDocument to markdown with HTML tables.

OmniDocBench ground-truth tables are stored as HTML, so we replace
Docling's default pipe-table rendering with each table's HTML export.
This keeps the evaluator comparing semantic content rather than
syntactic table formatting.
"""

from __future__ import annotations

import re

# A markdown pipe-table block: one or more consecutive lines where each
# line starts (possibly after leading whitespace) with a '|' character.
_MD_TABLE_BLOCK = re.compile(
    r"(?:^[ \t]*\|.*(?:\r?\n|$))+",
    re.MULTILINE,
)


def markdown_with_html_tables(doc) -> str:
    """
    Export *doc* to markdown, then replace each markdown pipe-table
    block with the corresponding HTML <table> produced by
    TableItem.export_to_html.

    Docling's iterate_items visits items in reading order, so the i-th
    TableItem maps to the i-th pipe-table block in the markdown output.
    Non-table content (headings, paragraphs, lists, math, images) is
    passed through unchanged.
    """
    from docling_core.types.doc.document import TableItem

    markdown = doc.export_to_markdown()

    html_tables = []
    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            try:
                html = item.export_to_html(doc=doc, add_caption=False)
            except Exception:
                html = ""
            html_tables.append(html.strip() if html else "")

    if not html_tables:
        return markdown

    iter_html = iter(html_tables)

    def _sub(match: "re.Match[str]") -> str:
        try:
            replacement = next(iter_html)
        except StopIteration:
            return match.group(0)
        if not replacement:
            return match.group(0)
        return replacement + "\n"

    return _MD_TABLE_BLOCK.sub(_sub, markdown)
