import requests
import gzip
import csv
import pandas as pd
import time
import io
import os

URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"

def main():
    print("[*] Connecting to Open Food Facts 5GB Bulk Database Stream...")
    start = time.time()

    try:
        response = requests.get(URL, stream=True, timeout=20, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        response.raise_for_status()
    except Exception as e:
        print(f"[!] Stream connection failed: {e}")
        return

    products = []
    target_count = 100000  # We will pull 100k rows directly out of the stream
    
    print("[*] Streaming and extracting real data on the fly (Bypassing API limits)...")
    
    try:
        with gzip.GzipFile(fileobj=response.raw) as gz:
            text_stream = io.TextIOWrapper(gz, encoding='utf-8', errors='ignore')
            reader = csv.DictReader(text_stream, delimiter='\t')
            
            for row in reader:
                name = row.get('product_name', '')
                ingredients = row.get('ingredients_text', '')
                categories = row.get('categories', '')
                
                if name and ingredients and categories:
                    cat = categories.split(',')[0].strip()
                    # Filter out junk categories
                    if len(cat) > 2 and "en:" not in cat and "fr:" not in cat:
                        products.append({
                            'product_name': name,
                            'ingredients': ingredients,
                            'category': cat
                        })
                
                if len(products) % 10000 == 0 and len(products) > 0:
                    print(f"    Streamed {len(products)} real products...")
                    
                if len(products) >= target_count:
                    break
    except Exception as e:
        print(f"[!] Stream interrupted (normal if connection closed early): {e}")

    if not products:
        print("[!] No products collected.")
        return

    df = pd.DataFrame(products)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/india_products_large.csv", index=False)
    print(f"[*] Done! Collected {len(df)} REAL products in {time.time() - start:.2f} seconds.")

if __name__ == '__main__':
    main()
