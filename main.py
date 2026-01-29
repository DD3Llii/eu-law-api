from fastapi import FastAPI
import pickle
import numpy as np
import requests
import os

app = FastAPI()

# 1. Haal de sleutel veilig op uit de Render-instellingen
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 2. Milieudata laden
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    embeddings = np.array(data['embeddings']).astype('float32')

@app.get("/")
def home():
    return {"status": "Online", "auth": "Sleutel actief" if HF_TOKEN else "Sleutel ontbreekt"}

@app.get("/ask")
def ask_law(query: str):
    # Stap 1: Vraag stellen met de juiste autorisatie
    response = requests.post(API_URL, headers=headers, json={"inputs": query})
    result = response.json()
    
    # Stap 2: Check op fouten of laden
    if isinstance(result, dict) and "error" in result:
        return {"query": query, "top_answers": ["AI-motor start op..."], "debug": result}
    
    # Stap 3: Resultaten tonen
    query_vector = np.array(result).astype('float32')
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    top_indices = np.argsort(distances)[:3]
    
    return {"query": query, "top_answers": [chunks[i] for i in top_indices]}
