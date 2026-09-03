import io
import os
import json
import requests
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from google import genai

load_dotenv()

app = FastAPI(title="Farmer Marketplace Comprehensive API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "579b464db66ec23bdd000001cdd394fe9d734a72793fe0e701468b61")

# Initialize Gemini Client
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
MANDI_API_URL = "https://api.data.gov.in/resource/9ef74138-9622-4331-8081-38e37a303e82"


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Farmer Marketplace API active (Voice, Vision Quality, Mandi Rates)"
    }


# ==========================================
# 1. VOICE TRANSCRIPTION (Sarvam AI)
# ==========================================
@app.post("/api/voice-listing")
async def process_voice_listing(
    file: UploadFile = File(...),
    language_code: str = Form("unknown")
):
    """
    Transcribes audio (supports Marathi, Kannada, Hindi, etc.) using Sarvam AI.
    """
    if not SARVAM_API_KEY:
        raise HTTPException(status_code=500, detail="Sarvam API key missing.")

    try:
        audio_bytes = await file.read()
        headers = {"api-subscription-key": SARVAM_API_KEY}
        files = {"file": (file.filename or "audio.wav", audio_bytes, file.content_type or "audio/wav")}
        data = {
            "model": "saaras:v3",
            "mode": "transcribe",
            "language_code": language_code,
            "with_timestamps": "false"
        }

        response = requests.post(SARVAM_STT_URL, headers=headers, files=files, data=data)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Sarvam AI Error: {response.text}")

        result = response.json()
        return {
            "success": True,
            "transcript": result.get("transcript"),
            "detected_language": result.get("language_code")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 2. PRODUCE IMAGE QUALITY RECOGNITION (Gemini Vision)
# ==========================================
@app.post("/api/analyze-crop-quality")
async def analyze_crop_quality(file: UploadFile = File(...)):
    """
    Analyzes an uploaded produce image to identify crop type, quality grade, 
    freshness score, and visible defects.
    """
    if not gemini_client:
        raise HTTPException(status_code=500, detail="Gemini API key is missing on backend server.")

    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        prompt = """
        You are an expert agricultural quality inspector. Analyze this produce image and return ONLY a valid JSON object without markdown formatting using the following structure:
        {
            "crop_name": "Identified Crop (e.g., Tomato, Onion)",
            "quality_grade": "Grade A / Grade B / Grade C",
            "freshness_score_percentage": 85,
            "color_consistency": "Good / Average / Poor",
            "visible_defects": ["List of defect descriptions or empty list"],
            "suggested_shelf_life_days": 5,
            "commercial_recommendation": "Brief assessment for market sale"
        }
        """

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_image, prompt]
        )

        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()
        analysis_json = json.loads(cleaned_text)

        return {
            "success": True,
            "analysis": analysis_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision Analysis Failed: {str(e)}")


# ==========================================
# 3. LIVE MANDI PRICES & COMPARISON (Data.gov.in / Agmarknet)
# ==========================================
@app.get("/api/mandi-prices")
def get_mandi_prices(
    state: str = Query(None, description="e.g. Maharashtra, Karnataka"),
    district: str = Query(None, description="e.g. Pune, Bengaluru"),
    commodity: str = Query(None, description="e.g. Tomato, Onion, Wheat"),
    limit: int = Query(10, description="Number of results to return")
):
    """
    Fetches real-time market prices from government Agmarknet / data.gov.in API.
    """
    params = {
        "api-key": DATA_GOV_API_KEY,
        "format": "json",
        "limit": limit
    }

    filters = []
    if state:
        filters.append(f"filters[state]={state}")
    if district:
        filters.append(f"filters[district]={district}")
    if commodity:
        filters.append(f"filters[commodity]={commodity}")

    query_string = "&".join(filters)
    request_url = f"{MANDI_API_URL}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"
    if query_string:
        request_url += f"&{query_string}"

    try:
        response = requests.get(request_url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Mandi API Error: {response.text}")

        data = response.json()
        records = data.get("records", [])

        formatted_records = []
        for r in records:
            formatted_records.append({
                "state": r.get("state"),
                "district": r.get("district"),
                "market": r.get("market"),
                "commodity": r.get("commodity"),
                "variety": r.get("variety"),
                "min_price_per_quintal": r.get("min_price"),
                "max_price_per_quintal": r.get("max_price"),
                "modal_price_per_quintal": r.get("modal_price"),
                "arrival_date": r.get("arrival_date")
            })

        return {
            "success": True,
            "total_records": len(formatted_records),
            "data": formatted_records
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch Mandi rates: {str(e)}")