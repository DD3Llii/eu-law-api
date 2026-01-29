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

# DE NIEUWE ROUTER URL
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

@app.get("/")
def home():
    return {"status": "Milieuwet API Online - Router Mode"}

@app.get("/ask")
def ask_law(query: str):
    # Roep de AI-motor aan
    response = requests.post(API_URL, json={"inputs": query})
    result = response.json()
    
    # Check of Hugging Face een fout geeft of nog aan het laden is
    if isinstance(result, dict) and "error" in result:
        return {"query": query, "top_answers": ["AI start nog op. Probeer over 10 seconden opnieuw."], "debug": result}
    
    # Verwerk de resultaten als alles goed gaat
    query_vector = np.array(result).astype('float32')
    distances = np.linalg.norm(embeddings - query_vector, axis=1)
    top_indices = np.argsort(distances)[:3]
    
    answers = [chunks[i] for i in top_indices]
    return {"query": query, "top_answers": answers}
