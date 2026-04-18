"""
Markdown post-processing rules for the Docling extraction pipeline.

All rules here are document-agnostic structural fixes for common
layout-model mistakes: over-segmented list items, text blocks
misclassified as tables, orphaned quotation marks, justified-text
spacing, and excessive blank lines.

Margin line numbers and page headers/footers are handled upstream
in the assembler via glyph-level margin detection and cross-page
chrome detection -- no hardcoded patterns live in this module.

Rules are applied in a fixed order and can be toggled individually
via the ``rules`` parameter on ``postprocess()``.
"""

import re
from typing import Optional, Set


# =========================================================================
# Individual rules
# =========================================================================

def collapse_spurious_list_items(text: str) -> str:
    """
    Merge consecutive markdown list items that are clearly body text
    broken across layout-model boxes.

    A ``- `` line whose text starts with a lowercase letter is almost
    certainly a wrapped paragraph continuation, not a real list item.
    """
    lines = text.split("\n")
    merged = []
    _CONTINUATION = re.compile(r"^- [a-z]")

    for line in lines:
        if (_CONTINUATION.match(line)
                and merged
                and merged[-1].startswith("- ")):
            merged[-1] = merged[-1].rstrip() + " " + line[2:]
        else:
            merged.append(line)

    return "\n".join(merged)


def unwrap_single_cell_tables(text: str) -> str:
    """
    Convert markdown tables that have only one meaningful data column
    back into plain text.

    Detects tables where every row has exactly two columns and the
    first column is always a bare number (a misclassified line-number
    column).  The structural shape check is document-agnostic.
    """
    result_lines = []
    table_buf = []
    in_table = False

    for line in text.split("\n"):
        is_table_row = line.startswith("|") and line.endswith("|")
        is_separator = bool(re.match(r"^\|[-:|]+\|$", line.strip()))

        if is_table_row or is_separator:
            in_table = True
            table_buf.append(line)
        else:
            if in_table:
                converted = _try_unwrap_table(table_buf)
                result_lines.append(converted)
                table_buf = []
                in_table = False
            result_lines.append(line)

    if table_buf:
        result_lines.append(_try_unwrap_table(table_buf))

    return "\n".join(result_lines)


def _try_unwrap_table(rows: list) -> str:
    """
    If every data row has exactly 2 cells and the first cell is a bare
    number, unwrap to plain text.  Otherwise return the original table.
    """
    data_rows = [r for r in rows
                 if not re.match(r"^\|[-:|]+\|$", r.strip())]
    if not data_rows:
        return "\n".join(rows)

    unwrapped = []
    all_numeric_first_col = True

    for row in data_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) == 2 and re.match(r"^\d{1,4}$", cells[0].strip()):
            unwrapped.append(cells[1].strip())
        elif len(cells) == 1:
            unwrapped.append(cells[0].strip())
        else:
            all_numeric_first_col = False
            break

    if all_numeric_first_col and unwrapped:
        return " ".join(unwrapped)

    return "\n".join(rows)


def merge_standalone_quotes(text: str) -> str:
    """
    Merge paragraphs that consist of a single quotation mark into the
    adjacent paragraph.  An opening or closing quote on its own line
    looks broken when rendered as a separate block.
    """
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in ('"', '\u201c', '\u201d'):
            if result and result[-1].strip():
                result[-1] = result[-1].rstrip() + " " + stripped
            elif i + 1 < len(lines) and lines[i + 1].strip():
                lines[i + 1] = stripped + " " + lines[i + 1].lstrip()
            else:
                result.append(lines[i])
            i += 1
            continue
        result.append(lines[i])
        i += 1
    return "\n".join(result)


def collapse_multiple_spacing(text: str) -> str:
    """
    Replace runs of 2+ spaces within a line with a single space.
    The PDF text layer often has wide inter-word gaps from justified
    text.
    """
    return re.sub(r"(?m)  +", " ", text)


def collapse_blank_lines(text: str) -> str:
    """
    Reduce runs of 3+ blank lines to at most 2 (one visual paragraph
    break).
    """
    return re.sub(r"\n{4,}", "\n\n\n", text)


# =========================================================================
# Orchestrator
# =========================================================================

ALL_RULES = {
    "spurious_list_items",
    "single_cell_tables",
    "standalone_quotes",
    "multiple_spacing",
    "blank_lines",
}

_RULE_FNS = {
    "spurious_list_items": collapse_spurious_list_items,
    "single_cell_tables": unwrap_single_cell_tables,
    "standalone_quotes": merge_standalone_quotes,
    "multiple_spacing": collapse_multiple_spacing,
    "blank_lines": collapse_blank_lines,
}

_RULE_ORDER = [
    "single_cell_tables",
    "spurious_list_items",
    "standalone_quotes",
    "multiple_spacing",
    "blank_lines",
]


def postprocess(
    text: str,
    rules: Optional[Set[str]] = None,
) -> str:
    """
    Apply post-processing rules to assembled markdown.

    Args:
        text:  Markdown string from the assembler.
        rules: Set of rule names to apply.  Defaults to all rules.
               Pass a subset to selectively enable rules.

    Returns:
        Cleaned markdown string.
    """
    if rules is None:
        rules = ALL_RULES

    for name in _RULE_ORDER:
        if name in rules:
            text = _RULE_FNS[name](text)

    return text.strip() + "\n"
