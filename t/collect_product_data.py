"""
SIH 26034 -- Step 1: Collect Real Indian Product Data
Pulls real packaged commodity data from Open Food Facts API (open source database).
Collects: barcodes, product names, brands, categories, net quantity, images.
All data is REAL products sold in India.

Source: https://world.openfoodfacts.org
License: Open Database License (ODbL)
"""

import json
import csv
import os
import time
import urllib.request
import urllib.error

# ─── CONFIG ─────────────────────────────────────────────────────────────────

OUTPUT_DIR = "data"
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "india_products_real.csv")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "product_images")
RAW_JSON = os.path.join(OUTPUT_DIR, "india_products_raw.json")

# Open Food Facts API - India products
BASE_URL = "https://world.openfoodfacts.org/api/v2/search"
FIELDS = ",".join([
    "code",                    # barcode (EAN-13)
    "product_name",            # product name
    "brands",                  # brand name
    "categories",              # product category
    "quantity",                # net quantity (e.g., "500g")
    "serving_size",            # serving size
    "packaging",               # packaging material
    "labels",                  # certifications (FSSAI, organic, etc.)
    "manufacturing_places",    # where it was made
    "origins",                 # origin of ingredients
    "stores",                  # stores selling it
    "countries",               # countries sold in
    "ingredients_text",        # ingredients list
    "nutriscore_grade",        # nutrition grade
    "nova_group",              # food processing level
    "image_url",               # main product image
    "image_front_url",         # front label image
    "image_nutrition_url",     # nutrition label image
    "image_ingredients_url",   # ingredients label image
])

PAGE_SIZE = 100  # max per request
TARGET_COUNT = 5000
DELAY_SECONDS = 1.5  # be respectful to the API


def fetch_page(page_num):
    """Fetch one page of Indian products from Open Food Facts."""
    url = (
        f"{BASE_URL}?"
        f"countries_tags_contains=en:india"
        f"&fields={FIELDS}"
        f"&page_size={PAGE_SIZE}"
        f"&page={page_num}"
        f"&json=1"
    )

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SIH26034-DataCollector/1.0 (student project)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data
    except urllib.error.HTTPError as e:
        print(f"  [ERROR] HTTP {e.code} on page {page_num}")
        return None
    except urllib.error.URLError as e:
        print(f"  [ERROR] URL error on page {page_num}: {e.reason}")
        return None
    except Exception as e:
        print(f"  [ERROR] Unexpected error on page {page_num}: {e}")
        return None


def download_image(url, save_path):
    """Download an image from URL."""
    if not url or os.path.exists(save_path):
        return False
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SIH26034-DataCollector/1.0 (student project)"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(save_path, "wb") as f:
                f.write(response.read())
        return True
    except Exception:
        return False


def clean_field(value):
    """Clean a field value - remove newlines, extra spaces."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).replace("\n", " ").replace("\r", "").strip()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    all_products = []
    total_available = None

    print("=" * 60)
    print("  SIH 26034 - COLLECTING REAL INDIAN PRODUCT DATA")
    print("  Source: Open Food Facts (openfoodfacts.org)")
    print("=" * 60)

    # Fetch pages
    pages_needed = (TARGET_COUNT // PAGE_SIZE) + 1

    for page in range(1, pages_needed + 1):
        print(f"\n[FETCH] Page {page}/{pages_needed} ...")
        data = fetch_page(page)

        if data is None:
            print(f"  [WARN] Failed to fetch page {page}, retrying...")
            time.sleep(3)
            data = fetch_page(page)
            if data is None:
                print(f"  [SKIP] Skipping page {page}")
                continue

        if total_available is None:
            total_available = data.get("count", 0)
            print(f"  [INFO] Total Indian products in database: {total_available}")

        products = data.get("products", [])
        if not products:
            print(f"  [INFO] No more products. Stopping.")
            break

        all_products.extend(products)
        print(f"  [OK] Got {len(products)} products (total so far: {len(all_products)})")

        if len(all_products) >= TARGET_COUNT:
            break

        # Rate limiting - be nice to the server
        time.sleep(DELAY_SECONDS)

    # Trim to target
    all_products = all_products[:TARGET_COUNT]

    # Save raw JSON
    print(f"\n[SAVE] Saving raw JSON ({len(all_products)} products)...")
    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(all_products, f, ensure_ascii=False, indent=2)

    # Process into clean CSV
    print(f"[SAVE] Processing into clean CSV...")

    csv_fields = [
        "barcode", "product_name", "brand", "category", "net_quantity",
        "packaging", "labels_certifications", "manufacturing_place",
        "ingredients", "nutriscore", "nova_group",
        "image_front_url", "image_nutrition_url", "image_ingredients_url",
        "has_image", "data_quality"
    ]

    csv_rows = []
    images_downloaded = 0
    skipped_empty = 0

    for p in all_products:
        barcode = clean_field(p.get("code", ""))
        name = clean_field(p.get("product_name", ""))

        # Skip entries with no barcode or no name
        if not barcode or not name:
            skipped_empty += 1
            continue

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

        # Data quality score (how many fields are filled)
        filled = sum(1 for v in [name, brand, category, quantity, packaging,
                                  labels, mfg_place, ingredients] if v)
        quality = f"{filled}/8"

        csv_rows.append({
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
            "data_quality": quality,
        })

        # Download front image for first 500 products (for training)
        if img_front and images_downloaded < 500:
            ext = img_front.split(".")[-1].split("?")[0]
            if ext not in ("jpg", "png", "webp", "jpeg"):
                ext = "jpg"
            img_path = os.path.join(IMAGES_DIR, f"{barcode}.{ext}")
            if download_image(img_front, img_path):
                images_downloaded += 1
                if images_downloaded % 50 == 0:
                    print(f"  [IMG] Downloaded {images_downloaded} images...")

    # Write CSV
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total fetched from API: {len(all_products)}")
    print(f"  Clean rows saved:       {len(csv_rows)}")
    print(f"  Skipped (no name/code): {skipped_empty}")
    print(f"  Images downloaded:      {images_downloaded}")
    print(f"")
    print(f"  Files:")
    print(f"    CSV:    {CSV_OUTPUT}")
    print(f"    JSON:   {RAW_JSON}")
    print(f"    Images: {IMAGES_DIR}/")
    print(f"{'=' * 60}")

    # Quick stats
    brands = set(r["brand"] for r in csv_rows if r["brand"])
    cats = set()
    for r in csv_rows:
        if r["category"]:
            for c in r["category"].split(","):
                cats.add(c.strip())

    print(f"\n  [STAT] Unique brands: {len(brands)}")
    print(f"  [STAT] Unique categories: {len(cats)}")
    print(f"  [STAT] With images: {sum(1 for r in csv_rows if r['has_image'] == 'yes')}")
    print(f"  [STAT] With ingredients: {sum(1 for r in csv_rows if r['ingredients'])}")
    print(f"  [STAT] With net quantity: {sum(1 for r in csv_rows if r['net_quantity'])}")

    # Show top brands
    brand_counts = {}
    for r in csv_rows:
        b = r["brand"]
        if b:
            brand_counts[b] = brand_counts.get(b, 0) + 1
    top_brands = sorted(brand_counts.items(), key=lambda x: -x[1])[:15]
    print(f"\n  [STAT] Top 15 Brands:")
    for b, c in top_brands:
        print(f"    {b}: {c}")


if __name__ == "__main__":
    main()
