import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

API_URL = "https://world.openfoodfacts.org/cgi/search.pl"

def fetch_page(page):
    try:
        params = {
            "search_terms": "india",
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page": page,
            "page_size": 100
        }
        headers = {"User-Agent": "SIHHackathonBot/1.0"}
        response = requests.get(API_URL, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json().get('products', [])
    except Exception:
        pass
    return []

def main():
    print("[*] Launching high-speed concurrent scraper for Open Food Facts...")
    start_time = time.time()
    
    all_products = []
    # Pull 60 pages concurrently
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_page, p): p for p in range(1, 61)}
        for future in as_completed(futures):
            products = future.result()
            if products:
                all_products.extend(products)
                print(f"    Fetched batch... Total so far: {len(all_products)}")

    cleaned_data = []
    for p in all_products:
        name = p.get('product_name', '')
        ingredients = p.get('ingredients_text', '')
        categories = p.get('categories', '')
        
        if name and categories:
            # Take the main category
            main_cat = categories.split(',')[0].strip()
            if len(main_cat) > 2:
                cleaned_data.append({
                    'product_name': name,
                    'ingredients': ingredients,
                    'category': main_cat
                })

    df = pd.DataFrame(cleaned_data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/india_products_large.csv", index=False)
    print(f"[*] Done! Collected {len(df)} REAL products in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
