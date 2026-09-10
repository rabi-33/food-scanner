import argparse
import re
from pathlib import Path

import cv2
from paddleocr import PaddleOCR


DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}\b|"
    r"\b\d{1,2}[/-]\d{4}\b|"
    r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s*\d{2,4}\b",
    re.IGNORECASE,
)


def extract_ocr_text(result):
    text_lines = []
    for item in result:
        data = item.json if hasattr(item, "json") else {}
        if callable(data):
            data = data()
        if isinstance(data, str):
            import json
            data = json.loads(data)
        result_data = data.get("res", data) if isinstance(data, dict) else {}
        text_lines.extend(result_data.get("rec_texts", []))
    return text_lines


def detect_barcode(image):
    detector = cv2.barcode_BarcodeDetector()
    _, decoded_info, decoded_type, _ = detector.detectAndDecodeWithType(image)
    return [
        {"value": value, "type": code_type}
        for value, code_type in zip(decoded_info or [], decoded_type or [])
        if value
    ]


def main():
    parser = argparse.ArgumentParser(description="Test expiry OCR and barcode scanning.")
    parser.add_argument("image", type=Path, help="Path to a product image")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    model_root = Path("data/ocr_models")
    ocr = PaddleOCR(
        lang="en",
        text_detection_model_dir=str(model_root / "PP-OCRv6_medium_det"),
        text_recognition_model_dir=str(model_root / "PP-OCRv6_medium_rec"),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    result = ocr.predict(image)
    text_lines = extract_ocr_text(result)
    all_text = " ".join(text_lines)

    print(f"Image: {args.image}")
    print("\nOCR text:")
    for line in text_lines:
        print(f"  {line}")
    print("\nPossible expiry/date values:")
    dates = DATE_PATTERN.findall(all_text)
    print("  " + (", ".join(dates) if dates else "none found"))
    print("\nBarcodes:")
    barcodes = detect_barcode(image)
    if barcodes:
        for barcode in barcodes:
            print(f"  {barcode['type']}: {barcode['value']}")
    else:
        print("  none found")


if __name__ == "__main__":
    main()