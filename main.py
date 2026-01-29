from fastapi import FastAPI
import pickle
import numpy as np
import requests
import os

app = FastAPI()

# 1. Gebruik de nieuwe verplichte Router URL
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 2. Database inladen (Milieuwet)
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    embeddings = np.array(data['embeddings']).astype('float32')

@app.get("/")
def home():
    return {"status": "Online", "engine": "Hugging Face Router"}

@app.get("/ask")
def ask_law(query: str):
    # Payload voor de nieuwe router
    payload = {"inputs": query, "options": {"wait_for_model": True}}
    
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    
    # Foutafhandeling voor de router-response
    if isinstance(result, dict) and "error" in result:
        return {"error": "Router melding", "details": result}
    
    # Vectorvergelijking
    query_vector = np.array(result).astype('float32')
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    top_indices = np.argsort(distances)[:3]
    
    return {
        "query": query,
        "top_answers": [chunks[i] for i in top_indices]
    }
