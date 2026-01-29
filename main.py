from fastapi import FastAPI
import pickle
import numpy as np
import requests

app = FastAPI()

# Milieudata laden
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    chunks = data['chunks']
    embeddings = np.array(data['embeddings']).astype('float32')

API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

@app.get("/")
def home():
    return {"status": "Online - Safe Mode Active"}

@app.get("/ask")
def ask_law(query: str):
    response = requests.post(API_URL, json={"inputs": query})
    result = response.json()
    
    # VEILIGHEIDSFILTER: Check of resultaat een foutmelding is van Hugging Face
    if isinstance(result, dict) and "error" in result:
        return {
            "query": query, 
            "top_answers": ["De AI-motor wordt momenteel geladen bij Hugging Face. Probeer het over 20 seconden nogmaals."],
            "debug_info": result
        }
    
    # Als alles goed is, verwerk de vector
    query_vector = np.array(result).astype('float32')
    
    # Zoeken met Numpy
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    top_indices = np.argsort(distances)[:3]
    
    answers = [chunks[i] for i in top_indices]
    return {"query": query, "top_answers": answers}
