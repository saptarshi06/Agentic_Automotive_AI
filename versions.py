# extractor/versions.py
import os, json
from pathlib import Path
from azure.ai.documentintelligence.models import AnalyzeResult

def build_v1(result: AnalyzeResult, figures_data: dict, filename: str) -> dict:
    """Raw blocks: text + diagram PNG refs + tables."""
    blocks = []
    base = Path(filename).stem

    # Text paragraphs in reading order
    if result.paragraphs:
        sorted_paras = sorted(
            result.paragraphs,
            key=lambda p: (p.spans[0].offset if p.spans else 0)
        )
        for para in sorted_paras:
            blocks.append({"type": "text", "content": para.content})

    # Figures — save PNG files, add ref
    for fig_id, png_bytes in figures_data.items():
        page_no = "unknown"
        if result.figures:
            for fig in result.figures:
                if fig.id == fig_id and fig.bounding_regions:
                    page_no = fig.bounding_regions[0].page_number
        ref_name = f"diagram_{base}_page{page_no}_{fig_id}.png"
        with open(f"outputs/figures/{ref_name}", "wb") as f:
            f.write(png_bytes)
        blocks.append({"type": "diagram", "ref": ref_name})

    # Tables
    if result.tables:
        for table in result.tables:
            rows = {}
            for cell in table.cells:
                r = cell.row_index
                rows.setdefault(r, {})[cell.column_index] = cell.content
            table_rows = [list(row.values()) for row in sorted(rows.items())]
            blocks.append({"type": "table", "rows": table_rows})

    return {"type": "v1_raw_blocks", "source": filename, "blocks": blocks}


def build_v2(result: AnalyzeResult, filename: str) -> dict:
    """Text + diagram descriptions (caption text) + structured table rows."""
    blocks = []

    if result.paragraphs:
        for para in sorted(result.paragraphs, key=lambda p: (p.spans[0].offset if p.spans else 0)):
            blocks.append({"type": "text", "content": para.content})

    # Figures: use caption content as description if available
    if result.figures:
        for fig in result.figures:
            desc = fig.caption.content if fig.caption else f"Figure on page {fig.bounding_regions[0].page_number if fig.bounding_regions else '?'}"
            blocks.append({"type": "diagram_description", "description": desc})

    if result.tables:
        for table in result.tables:
            # Build header from first row
            header_row = {c.column_index: c.content for c in table.cells if c.row_index == 0}
            headers = [header_row.get(i, f"col{i}") for i in range(table.column_count)]
            data_rows = []
            for row_idx in range(1, table.row_count):
                row_cells = {c.column_index: c.content for c in table.cells if c.row_index == row_idx}
                data_rows.append({headers[i]: row_cells.get(i, "") for i in range(table.column_count)})
            blocks.append({"type": "table", "headers": headers, "rows": data_rows})

    return {"type": "v2_described", "source": filename, "blocks": blocks}


def build_v3(result: AnalyzeResult, filename: str) -> dict:
    """Document structure: header, TOC, footer, company, confidential flag."""
    header_texts, footer_texts, toc_items, body_texts = [], [], [], []

    if result.paragraphs:
        for para in result.paragraphs:
            role = para.role or "body"
            if role in ("pageHeader", "sectionHeading"):
                header_texts.append(para.content)
            elif role == "pageFooter":
                footer_texts.append(para.content)
            elif role == "title":
                toc_items.append(para.content)
            else:
                body_texts.append(para.content)

    full_text = result.content or ""
    confidential = any(w in full_text.upper() for w in ["CONFIDENTIAL", "PROPRIETARY", "INTERNAL ONLY"])

    # Try to detect company name from first sectionHeading or title
    company = toc_items[0] if toc_items else "Unknown"

    return {
        "type": "v3_structure",
        "source": filename,
        "header": " | ".join(header_texts),
        "table_of_contents": toc_items,
        "footer": " | ".join(footer_texts),
        "company": company,
        "confidential": confidential,
        "page_count": len(result.pages) if result.pages else 0,
    }


def build_v4(result: AnalyzeResult, filename: str) -> dict:
    """Cross-references and summary from key-value pairs."""
    import re

    full_text = result.content or ""

    # Find cross-references like "refer section 3.2", "see section 4", "per clause 2.1"
    xref_pattern = re.compile(
        r'(refer(?:ence)?|see|per|as per|refer to)\s+(section|clause|appendix|figure|table)\s+([\d.]+)',
        re.IGNORECASE
    )
    cross_refs = [
        {"text": m.group(0), "target": f"{m.group(2)} {m.group(3)}"}
        for m in xref_pattern.finditer(full_text)
    ]

    # Key-value pairs from Azure DI
    kv_pairs = {}
    if result.key_value_pairs:
        for kv in result.key_value_pairs:
            if kv.key and kv.value:
                kv_pairs[kv.key.content] = kv.value.content

    # Summary: first 3 non-empty paragraphs
    summary_parts = []
    if result.paragraphs:
        for para in result.paragraphs[:5]:
            if para.content and len(para.content) > 30:
                summary_parts.append(para.content)
            if len(summary_parts) == 3:
                break

    return {
        "type": "v4_semantic",
        "source": filename,
        "cross_references": cross_refs,
        "key_value_pairs": kv_pairs,
        "summary": " ".join(summary_parts),
    }