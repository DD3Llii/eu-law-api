from fastapi import FastAPI
import pickle
import numpy as np
import requests
import os
import time

app = FastAPI()

# 1. Configuraties
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 2. Database inladen
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    embeddings = np.array(data['embeddings']).astype('float32')

@app.get("/")
def home():
    return {"status": "Online", "mode": "Extended Timeout Enabled"}

@app.get("/ask")
def ask_law(query: str):
    # Payload met expliciete opdracht om te wachten op het model
    payload = {"inputs": query, "options": {"wait_for_model": True}}
    
    # We geven de AI-motor tot 60 seconden de tijd (was 15)
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.Timeout:
        return {"error": "De AI-motor doet er te lang over. Klik nogmaals op 'Test Endpoint' om de motor warm te houden."}
    except Exception as e:
        return {"error": "AI-motor fout", "details": str(e)}

    # Vectorverwerking
    query_vector = np.array(result).astype('float32')
    if len(query_vector.shape) > 1:
        query_vector = query_vector[0]

    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    top_indices = np.argsort(distances)[:3]
    
    return {
        "query": query,
        "top_answers": [chunks[i] for i in top_indices]
    }
