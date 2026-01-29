from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import requests
import os

app = FastAPI(title="EU Law Intelligence API")

# 1. Configuraties: Haal token veilig op uit Render instellingen
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 2. Data inladen
try:
    with open('data.pkl', 'rb') as f:
        database = pickle.load(f)
        chunks = database['chunks']
        embeddings = np.array(database['embeddings']).astype('float32')
except Exception as e:
    print(f"Database fout: {e}")
    chunks, embeddings = [], None

@app.get("/")
def home():
    return {"status": "Online", "auth": "Token geconfigureerd" if HF_TOKEN else "Token ontbreekt"}

@app.get("/ask")
def ask_law(query: str):
    if not HF_TOKEN:
        return {"error": "HF_TOKEN ontbreekt in Render Environment Variables"}

    # Stap 1: Zoekvraag naar AI-motor met Token
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": query}, timeout=15)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return {"error": "AI-motor fout", "details": str(e)}

    # Stap 2: Check op 'loading'
    if isinstance(result, dict) and "error" in result:
        return {"message": "AI-motor start op...", "debug": result}

    # Stap 3: Resultaten berekenen
    query_vector = np.array(result).astype('float32')
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    top_indices = np.argsort(distances)[:3]
    
    return {
        "query": query,
        "top_answers": [chunks[i] for i in top_indices]
    }
