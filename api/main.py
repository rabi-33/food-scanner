from datetime import date, datetime, timezone
from pathlib import Path
import os
import re
from uuid import uuid4

import cv2
import numpy as np
import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model" / "ocr_models"
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}\b|"
    r"\b\d{1,2}[/-]\d{4}\b|"
    r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s*\d{2,4}\b",
    re.IGNORECASE,
)
MONTHS = {
    name: number
    for number, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        start=1,
    )
}

app = FastAPI(title="Food Inspector Scanner API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
ocr = None
inspection_memory = []


def get_ocr():
    global ocr
    if ocr is None:
        ocr = PaddleOCR(
            text_detection_model_dir=str(MODEL_ROOT / "PP-OCRv6_medium_det"),
            text_recognition_model_dir=str(MODEL_ROOT / "PP-OCRv6_medium_rec"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return ocr


def ocr_text(result):
    lines = []
    for item in result:
        data = item.json if hasattr(item, "json") else {}
        if callable(data):
            data = data()
        if isinstance(data, str):
            import json
            data = json.loads(data)
        payload = data.get("res", data) if isinstance(data, dict) else {}
        lines.extend(payload.get("rec_texts", []))
    return lines


def find_barcodes(image):
    detector = cv2.barcode_BarcodeDetector()
    _, values, types, _ = detector.detectAndDecodeWithType(image)
    return [
        {"value": value, "type": code_type}
        for value, code_type in zip(values or [], types or [])
        if value
    ]


def date_status(date_texts):
    if not date_texts:
        return {"status": "NEEDS_REVIEW", "expiry_date": None}

    parsed = []
    for text in date_texts:
        normalized = text.upper().replace(" ", "")
        month_match = re.search(r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*(\d{2,4})", normalized)
        if month_match:
            year = int(month_match.group(2))
            year += 2000 if year < 100 else 0
            parsed.append(date(year, MONTHS[month_match.group(1)], 1))
            continue
        numeric = re.search(r"(\d{1,2})[/-](\d{2,4})$", normalized)
        if numeric:
            month, year = int(numeric.group(1)), int(numeric.group(2))
            year += 2000 if year < 100 else 0
            if 1 <= month <= 12:
                parsed.append(date(year, month, 1))

    if not parsed:
        return {"status": "NEEDS_REVIEW", "expiry_date": None}
    expiry = min(parsed)
    return {
        "status": "EXPIRED" if expiry < date.today().replace(day=1) else "FRESH",
        "expiry_date": expiry.isoformat(),
    }


async def save_inspection(record):
    inspection_memory.insert(0, record)
    del inspection_memory[100:]
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        return
    async with httpx.AsyncClient(timeout=8) as client:
        response = await client.post(
            f"{supabase_url.rstrip('/')}/rest/v1/inspections",
            headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}", "Prefer": "return=minimal"},
            json=record,
        )
        response.raise_for_status()


async def load_inspections(limit=100):
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        return inspection_memory[:limit]
    async with httpx.AsyncClient(timeout=8) as client:
        response = await client.get(
            f"{supabase_url.rstrip('/')}/rest/v1/inspections",
            headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
            params={"select": "*", "order": "created_at.desc", "limit": limit},
        )
        response.raise_for_status()
        return response.json()


@app.get("/health")
def health():
    return {"status": "ok", "service": "food-inspector-scanner"}


@app.get("/")
def root():
    return {
        "service": "food-inspector-scanner",
        "status": "online",
        "health": "/health",
        "api_docs": "/docs",
        "scan_endpoint": "POST /scan",
    }


@app.post("/scan")
async def scan(image: UploadFile = File(...)):
    content = await image.read()
    pixels = np.frombuffer(content, dtype=np.uint8)
    decoded = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")

    lines = ocr_text(get_ocr().predict(decoded))
    text = " ".join(lines)
    date_texts = DATE_PATTERN.findall(text)
    result = date_status(date_texts)
    result.update({
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": image.filename,
        "ocr_text": lines,
        "barcodes": find_barcodes(decoded),
        "date_candidates": date_texts,
    })
    try:
        await save_inspection(result)
    except httpx.HTTPError:
        result["storage_warning"] = "Scan completed, but cloud history could not be saved."
    return result


@app.get("/inspections")
async def inspections(limit: int = 100):
    return {"items": await load_inspections(max(1, min(limit, 100)))}


@app.get("/inspections/{inspection_id}/report")
async def inspection_report(inspection_id: str):
    records = await load_inspections(100)
    record = next((item for item in records if item.get("id") == inspection_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="Inspection not found")
    return {
        "document_type": "Food inspection report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inspection": record,
    }


@app.get("/products/{barcode}")
async def product_lookup(barcode: str):
    """Public product metadata fallback until an authorized FSSAI feed is available."""
    async with httpx.AsyncClient(timeout=8) as client:
        response = await client.get(
            f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json",
            params={"fields": "code,product_name,brands,categories,quantity"},
        )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Product database unavailable")
    payload = response.json()
    if payload.get("status") != 1:
        raise HTTPException(status_code=404, detail="Barcode not found")
    return {"source": "openfoodfacts", "product": payload.get("product", {})}