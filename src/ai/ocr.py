"""
OCR invoice scanning.

Primary path uses Claude vision to read a supplier invoice image and return
structured line items (product, qty, unit price, total) as JSON — far more
robust on real-world invoices than classic OCR. A pytesseract path provides an
offline fallback for raw text extraction when the API isn't configured.
"""
from __future__ import annotations

import json
import re

from src.ai import claude_client

EXTRACTION_PROMPT = """You are an invoice parser. Read this supplier invoice image and
extract the data as STRICT JSON with this schema:
{
  "supplier_name": string|null,
  "invoice_number": string|null,
  "invoice_date": string|null,
  "currency": string|null,
  "line_items": [
    {"description": string, "quantity": number, "unit_price": number, "total": number}
  ],
  "subtotal": number|null,
  "tax": number|null,
  "grand_total": number|null
}
Return ONLY the JSON object, no commentary. Use null for anything you cannot read."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    # strip markdown fences if present
    text = re.sub(r"^```(json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"_raw": text, "_parse_error": True}


def scan_invoice(image_bytes: bytes, media_type: str = "image/png") -> dict:
    """Return structured invoice data. Falls back to plain OCR text if needed."""
    if claude_client.available():
        try:
            raw = claude_client.vision(EXTRACTION_PROMPT, image_bytes, media_type=media_type)
            data = _extract_json(raw)
            data["_engine"] = "claude_vision"
            return data
        except Exception as e:  # pragma: no cover
            return {"_error": str(e), "_engine": "claude_vision_failed", **_tesseract_fallback(image_bytes)}
    return _tesseract_fallback(image_bytes)


def _tesseract_fallback(image_bytes: bytes) -> dict:
    try:
        import io

        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return {"_engine": "pytesseract", "raw_text": text, "line_items": _parse_text_lines(text)}
    except Exception as e:
        return {
            "_engine": "unavailable",
            "_error": f"No OCR backend available ({e}). Set ANTHROPIC_API_KEY or install tesseract.",
            "line_items": [],
        }


def _parse_text_lines(text: str) -> list[dict]:
    """Very small heuristic line parser for the tesseract fallback."""
    items = []
    for line in text.splitlines():
        nums = re.findall(r"\d+\.\d+|\d+", line)
        if len(nums) >= 2 and any(c.isalpha() for c in line):
            desc = re.sub(r"[\d\.\,]+", "", line).strip(" -|:")
            if desc:
                items.append(
                    {
                        "description": desc,
                        "quantity": float(nums[0]),
                        "unit_price": float(nums[-2]) if len(nums) >= 2 else None,
                        "total": float(nums[-1]),
                    }
                )
    return items
