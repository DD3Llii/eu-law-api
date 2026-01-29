from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
import requests
import os

app = FastAPI()

# 1. Configuraties
HF_TOKEN = os.getenv("HF_TOKEN")
# We gebruiken de stabiele endpoint die 401 gaf (toen we nog geen token hadden)
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 2. Data inladen
try:
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
        chunks = data['chunks']
        embeddings = np.array(data['embeddings']).astype('float32')
except Exception as e:
    chunks, embeddings = [], None

@app.get("/")
def home():
    return {"status": "Online", "database_size": len(chunks) if chunks else 0}

@app.get("/ask")
def ask_law(query: str):
    if not query:
        return {"error": "Geen zoekopdracht ontvangen."}
    
    # Stap 1: De AI-motor aanroepen
    # We sturen de query nu op de meest universele manier mee
    payload = {"inputs": query, "options": {"wait_for_model": True}}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        
        # Als de router zegt dat we een andere URL moeten gebruiken, volgen we die automatisch
        if response.status_code == 301 or "router" in response.text:
            new_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
            response = requests.post(new_url, headers=headers, json=payload, timeout=20)

        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return {"error": "AI-motor fout", "details": str(e), "response": response.text if 'response' in locals() else "Geen verbinding"}

    # Stap 2: De vector verwerken (Hugging Face stuurt soms een geneste lijst terug)
    try:
        query_vector = np.array(result).astype('float32')
        if len(query_vector.shape) > 1:
            query_vector = query_vector[0] # Pak de eerste vector als het een lijst van lijsten is
            
        # Stap 3: Zoeken met Numpy
        distances = np.linalg.norm(embeddings - query_vector, axis=1)
        top_indices = np.argsort(distances)[:3]
        
        return {
            "query": query,
            "top_answers": [chunks[i] for i in top_indices]
        }
    except Exception as e:
        return {"error": "Verwerkingsfout", "details": str(e), "raw_result_type": str(type(result))}
