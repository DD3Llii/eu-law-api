from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import requests
import os

app = FastAPI(title="EU Law Intelligence API")

# 1. Configuraties en URL's
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

# 2. Data inladen met foutcontrole
try:
    with open('data.pkl', 'rb') as f:
        database = pickle.load(f)
        chunks = database['chunks']
        # Converteer embeddings direct naar float32 voor snelle Numpy berekeningen
        embeddings = np.array(database['embeddings']).astype('float32')
except FileNotFoundError:
    print("FOUT: data.pkl niet gevonden. Upload dit bestand naar de hoofdmap.")
    chunks, embeddings = [], None

@app.get("/")
def health_check():
    """Controleert of de server en de database online zijn."""
    if embeddings is None:
        return {"status": "Error", "message": "Database (data.pkl) ontbreekt."}
    return {"status": "Online", "mode": "Numpy Vector Search"}

@app.get("/ask")
def ask_law(query: str):
    """Zoekt de meest relevante wetsartikelen op basis van een zoekopdracht."""
    if not query:
        raise HTTPException(status_code=400, detail="Voer een zoekvraag in via de 'query' parameter.")

    # Stap 1: Zoekvraag omzetten naar een vector via Hugging Face
    try:
        response = requests.post(API_URL, json={"inputs": query}, timeout=10)
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return {"error": "Externe AI-motor onbereikbaar", "details": str(e)}

    # Stap 2: Afhandelen van opstartfase (Cold Start) van het AI-model
    if isinstance(result, dict) and "error" in result:
        return {
            "status": "Aan het opstarten",
            "message": "De AI-motor wordt geladen. Probeer het over 15 seconden opnieuw.",
            "raw_error": result.get("error")
        }

    # Stap 3: Vectorvergelijking uitvoeren met Numpy (Euclidean Distance)
    try:
        query_vector = np.array(result).astype('float32')
        # Bereken afstand tussen zoekvraag en alle 836 blokken
        distances = np.linalg.norm(embeddings - query_vector, axis=1)
        # Selecteer de 3 blokken met de kleinste afstand
        top_indices = np.argsort(distances)[:3]
        
        return {
            "query": query,
            "results": [chunks[i] for i in top_indices]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fout bij dataverwerking: {str(e)}")
