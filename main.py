import os, pickle, zipfile, numpy as np
from fastapi import FastAPI, HTTPException
from google import genai

app = FastAPI(title="Multi-Law Legal AI - Compressed")

# Config
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

def load_full_database():
    all_chunks = []
    all_embeddings = []
    
    # We laden beide zip-bestanden in
    for filename in ['data1.zip', 'data2.zip']:
        if os.path.exists(filename):
            with zipfile.ZipFile(filename, 'r') as z:
                with z.open(filename.replace('.zip', '.pkl')) as f:
                    data = pickle.load(f)
                    all_chunks.extend(data['chunks'])
                    all_embeddings.append(data['embeddings'])
    
    if not all_chunks:
        return [], None
        
    return all_chunks, np.vstack(all_embeddings).astype('float32')

chunks, embeddings = load_full_database()

@app.get("/")
def home():
    return {"status": "Online", "database_size": len(chunks)}

@app.get("/ask")
def ask(query: str):
    if not query or not chunks:
        raise HTTPException(status_code=400, detail="Systeem niet gereed of geen vraag.")

    try:
        res = client.models.embed_content(
            model="text-embedding-004",
            contents=query,
            config={'task_type': 'RETRIEVAL_QUERY'}
        )
        query_vector = np.array(res.embeddings[0].values).astype('float32')
        
        # Berekening
        distances = np.linalg.norm(embeddings - query_vector, axis=1)
        top_indices = np.argsort(distances)[:5]
        
        return {"query": query, "results": [chunks[i] for i in indices]}
    except Exception as e:
        return {"error": str(e)}
