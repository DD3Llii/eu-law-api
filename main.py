import os
import pickle
import numpy as np
import requests
from fastapi import FastAPI

app = FastAPI()

# 1. Configuratie met de exacte Router URL
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 2. Database inladen
with open('data.pkl', 'rb') as f:
    db = pickle.load(f)
    chunks = db['chunks']
    embeddings = np.array(db['embeddings']).astype('float32')

@app.get("/")
def home():
    return {"status": "Online", "chunks": len(chunks)}

@app.get("/ask")
def ask(query: str):
    # Gebruik de meest eenvoudige payload om 400-fouten te voorkomen
    payload = {"inputs": query, "options": {"wait_for_model": True}}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return {"error": "AI-motor weigert", "details": str(e), "response": response.text if 'response' in locals() else "No Response"}

    # Controleer op laden van model
    if isinstance(result, dict) and "error" in result:
        return {"status": "loading", "message": "AI start op, probeer het over 10 seconden opnieuw."}

    # Vector verwerking en berekening
    vector = np.array(result).astype('float32')
    if vector.ndim > 1:
        vector = vector[0]

    distances = np.linalg.norm(embeddings - vector, axis=1)
    indices = np.argsort(distances)[:3]
    
    return {
        "query": query,
        "results": [chunks[i] for i in indices]
    }
