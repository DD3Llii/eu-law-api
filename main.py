import os
import pickle
import numpy as np
import requests
from fastapi import FastAPI

app = FastAPI()

# 1. De enige juiste URL voor de nieuwe Hugging Face Router
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# 2. Database inladen
try:
    with open('data.pkl', 'rb') as f:
        db = pickle.load(f)
        chunks = db['chunks']
        embeddings = np.array(db['embeddings']).astype('float32')
except Exception as e:
    chunks, embeddings = [], None

@app.get("/")
def home():
    return {"status": "Online", "database": "Geladen" if chunks else "Niet gevonden"}

@app.get("/ask")
def ask(query: str):
    if not query:
        return {"error": "Geen query gevonden"}

    # Payload zonder extra opties om 400-fouten te voorkomen
    payload = {"inputs": query}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        # Als we een 404 of 400 krijgen, willen we precies zien wat Hugging Face zegt
        if response.status_code != 200:
            return {
                "error": f"AI-motor weigert (Status {response.status_code})",
                "huggingface_says": response.text
            }
            
        result = response.json()
    except Exception as e:
        return {"error": "Verbindingsfout", "details": str(e)}

    # Stap 3: Vector berekening
    try:
        query_vector = np.array(result).astype('float32')
        # Soms stuurt HF een lijst-in-een-lijst terug [[...]]
        if query_vector.ndim > 1:
            query_vector = query_vector[0]

        distances = np.linalg.norm(embeddings - query_vector, axis=1)
        top_indices = np.argsort(distances)[:3]
        
        return {
            "query": query,
            "top_answers": [chunks[i] for i in top_indices]
        }
    except Exception as e:
        return {"error": "Data-verwerkingsfout", "details": str(e), "raw_ai_output": str(result)[:200]}
