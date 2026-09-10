from datetime import date
from pathlib import Path
import re

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
        "filename": image.filename,
        "ocr_text": lines,
        "barcodes": find_barcodes(decoded),
        "date_candidates": date_texts,
    })
    return result


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