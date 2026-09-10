# Food Inspector Scanner

The project has three parts:

- `api/`: FastAPI scanner for Render. PaddleOCR reads expiry text, OpenCV reads barcodes, and the date rules return `FRESH`, `EXPIRED`, or `NEEDS_REVIEW`.
- `mobile/`: Expo React Native app for inspectors. Set `EXPO_PUBLIC_API_URL` to the Render API URL before starting it.
- `admin/`: lightweight Render static dashboard for image-based review.

## Local API

```powershell
python -m uvicorn api.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Test it with:

```powershell
curl.exe -F "image=@data/exp_data_food/10.jfif" http://127.0.0.1:8000/scan
```

## Mobile app

```powershell
$env:EXPO_PUBLIC_API_URL = "http://192.168.1.10:8000"
Set-Location mobile
npm start
```

Use the computer's LAN IP on a physical phone, not `127.0.0.1`.

## Render

`render.yaml` defines the scanner API and admin site. Connect this repository to Render as a Blueprint. Render will build the API with `api/requirements.txt` and publish `admin/` as a static site.

The `/products/{barcode}` endpoint currently uses Open Food Facts for public product metadata. FSSAI data should be connected through an authorized feed or a reviewed CSV import when one is available; it is not safe to claim that Open Food Facts is an FSSAI source.