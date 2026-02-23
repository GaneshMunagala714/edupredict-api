# EduPredict ML Backend

FastAPI + Random Forest backend for the EduPredict dashboard.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload CSV, get column info |
| POST | `/train` | Train Random Forest models |
| POST | `/predict` | Single value prediction |
| POST | `/predict/bulk` | Bulk predictions |
| GET | `/models/{model_id}` | Model info |

## Deploy to Render.com (Free) — Step by Step

### 1. Create a GitHub repo for the backend
- Go to github.com/new
- Name it `edupredict-api`
- Make it Public
- Click Create repository

### 2. Upload these files to the repo
Upload all three files:
- main.py
- requirements.txt
- render.yaml

### 3. Deploy on Render
- Go to https://render.com and sign up (free)
- Click "New +" → "Web Service"
- Connect your GitHub account
- Select the `edupredict-api` repo
- Render auto-detects render.yaml — click Deploy
- Wait ~3 minutes for build to complete
- Your API URL will be: https://edupredict-api.onrender.com

### 4. Update your frontend
In your dashboard index.html, set:
```
const API_URL = "https://edupredict-api.onrender.com";
```

### 5. Test the API
Visit: https://edupredict-api.onrender.com/health
You should see: {"status": "ok"}

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# API runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```
