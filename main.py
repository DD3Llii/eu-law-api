import os, pickle, requests, numpy as np
from fastapi import FastAPI

app = FastAPI()

# 1. We stappen over naar een puur 'Embedding' model (geen similarity gedoe)
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# 2. Laden data
with open('data.pkl', 'rb') as f:
    db = pickle.load(f)
    chunks, embeddings = db['chunks'], np.array(db['embeddings']).astype('float32')

@app.get("/")
def home(): return {"status": "Online"}

@app.get("/ask")
def ask(query: str):
    # De 'wait_for_model' is cruciaal om de timeout te voorkomen
    r = requests.post(API_URL, headers=headers, json={"inputs": query, "options": {"wait_for_model": True}})
    
    # Als Hugging Face getallen terugstuurt, rekenen we direct uit
    if r.status_code == 200:
        vector = np.array(r.json()).astype('float32')
        if vector.ndim > 1: vector = vector[0]
        
        distances = np.linalg.norm(embeddings - vector, axis=1)
        indices = np.argsort(distances)[:3]
        return {"query": query, "results": [chunks[i] for i in indices]}
    
    return {"status": "error", "hf_response": r.json()}
