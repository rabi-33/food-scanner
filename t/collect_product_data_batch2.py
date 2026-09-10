"""
SIH 26034 -- Collect Indian Product Data via Bulk Download
Downloads the Open Food Facts India subset CSV directly.
No rate limits since it's a single file download.
Then filters and cleans for our use case.
"""

import csv
import os
import gzip
import urllib.request
import io
import json

OUTPUT_DIR = "data"
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "india_products_bulk.csv")
MERGED_OUTPUT = os.path.join(OUTPUT_DIR, "india_products_all.csv")

# Open Food Facts provides country-specific search exports
# Using the advanced search CSV export for India
# This URL fetches Indian products as CSV (up to 10,000)
SEARCH_URL = (
    "https://world.openfoodfacts.org/cgi/search.pl?"
    "action=process"
    "&tagtype_0=countries"
    "&tag_contains_0=contains"
    "&tag_0=india"
    "&sort_by=unique_scans_n"  # most scanned first = best quality data
    "&page_size=1000"
    "&page={page}"
    "&json=1"
)

# Fields we care about for Legal Metrology compliance
KEEP_FIELDS = [
    "barcode", "product_name", "brand", "category", "net_quantity",
    "packaging", "labels_certifications", "manufacturing_place",
    "ingredients", "nutriscore", "nova_group",
    "image_front_url", "image_nutrition_url", "image_ingredients_url",
    "has_image", "data_quality"
]


def clean_field(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).replace("\n", " ").replace("\r", "").replace("\t", " ").strip()


def fetch_search_page(page):
    """Fetch products using search endpoint (different from API v2, less strict limits)."""
    url = SEARCH_URL.format(page=page)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SIH26034-BulkCollector/1.0 (academic project)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data
    except Exception as e:
        print(f"  [ERROR] Page {page}: {e}")
        return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing barcodes from batch 1
    existing_barcodes = set()
    existing_rows = []
    batch1_csv = os.path.join(OUTPUT_DIR, "india_products_real.csv")

    if os.path.exists(batch1_csv):
        with open(batch1_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_barcodes.add(row["barcode"])
                existing_rows.append(row)
        print(f"[INFO] Loaded {len(existing_rows)} existing entries from batch 1")

    # Try the search endpoint (separate from API v2, often works when v2 is limited)
    print("=" * 60)
    print("  COLLECTING VIA SEARCH ENDPOINT")
    print("=" * 60)

    all_new = []
    import time

    for page in range(1, 25):  # 24 pages * ~1000 = up to ~24,000 products
        print(f"\n[FETCH] Search page {page}/24 ...")
        data = fetch_search_page(page)

        if data is None:
            print(f"  [WARN] Failed, waiting 10s and retrying...")
            time.sleep(10)
            data = fetch_search_page(page)
            if data is None:
                print(f"  [SKIP] Still failing, moving on")
                time.sleep(5)
                continue

        products = data.get("products", [])
        if not products:
            print(f"  [INFO] No more products")
            break

        new_count = 0
        for p in products:
            barcode = clean_field(p.get("code", ""))
            name = clean_field(p.get("product_name", ""))
            if not barcode or not name:
                continue
            if barcode in existing_barcodes:
                continue

            existing_barcodes.add(barcode)
            new_count += 1

            brand = clean_field(p.get("brands", ""))
            category = clean_field(p.get("categories", ""))
            quantity = clean_field(p.get("quantity", ""))
            packaging = clean_field(p.get("packaging", ""))
            labels = clean_field(p.get("labels", ""))
            mfg_place = clean_field(p.get("manufacturing_places", ""))
            ingredients = clean_field(p.get("ingredients_text", ""))
            nutriscore = clean_field(p.get("nutriscore_grade", ""))
            nova = clean_field(p.get("nova_group", ""))
            img_front = clean_field(p.get("image_front_url", ""))
            img_nutrition = clean_field(p.get("image_nutrition_url", ""))
            img_ingredients = clean_field(p.get("image_ingredients_url", ""))
            has_image = "yes" if img_front else "no"
            filled = sum(1 for v in [name, brand, category, quantity, packaging,
                                      labels, mfg_place, ingredients] if v)

            all_new.append({
                "barcode": barcode,
                "product_name": name,
                "brand": brand,
                "category": category,
                "net_quantity": quantity,
                "packaging": packaging,
                "labels_certifications": labels,
                "manufacturing_place": mfg_place,
                "ingredients": ingredients,
                "nutriscore": nutriscore,
                "nova_group": nova,
                "image_front_url": img_front,
                "image_nutrition_url": img_nutrition,
                "image_ingredients_url": img_ingredients,
                "has_image": has_image,
                "data_quality": f"{filled}/8",
            })

        print(f"  [OK] {len(products)} products, {new_count} new (total new: {len(all_new)})")

        # Check if we have enough combined with existing
        total = len(existing_rows) + len(all_new)
        if total >= 5000:
            print(f"\n  [DONE] Reached {total} total entries!")
            break

        time.sleep(3)  # moderate delay

    # Save new entries
    if all_new:
        with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=KEEP_FIELDS)
            writer.writeheader()
            writer.writerows(all_new)
        print(f"\n[SAVE] Saved {len(all_new)} new entries to {CSV_OUTPUT}")

    # Merge all into one file
    all_rows = existing_rows + all_new
    with open(MERGED_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'=' * 60}")
    print(f"  COLLECTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Batch 1 (API v2):    {len(existing_rows)} entries")
    print(f"  Batch 2 (Search):    {len(all_new)} entries")
    print(f"  TOTAL MERGED:        {len(all_rows)} entries")
    print(f"  Unique barcodes:     {len(existing_barcodes)}")
    print(f"")
    print(f"  Files:")
    print(f"    Merged CSV: {MERGED_OUTPUT}")
    print(f"    Batch 2:    {CSV_OUTPUT}")
    print(f"{'=' * 60}")

    # Stats
    brands = set(r["brand"] for r in all_rows if r["brand"])
    with_img = sum(1 for r in all_rows if r["has_image"] == "yes")
    with_ing = sum(1 for r in all_rows if r["ingredients"])
    with_qty = sum(1 for r in all_rows if r["net_quantity"])

    print(f"\n  [STAT] Unique brands:      {len(brands)}")
    print(f"  [STAT] With images:        {with_img}")
    print(f"  [STAT] With ingredients:   {with_ing}")
    print(f"  [STAT] With net quantity:  {with_qty}")


if __name__ == "__main__":
    main()
