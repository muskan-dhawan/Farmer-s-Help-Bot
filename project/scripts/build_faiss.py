import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Path adjustment
csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kcc_dataset.csv")

if not os.path.exists(csv_path):
    print(f"❌ Dataset not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

texts = []
meta = []

for _, row in df.iterrows():
    text = f"Q: {row['QueryText']} A: {row['KccAns']}"
    texts.append(text)
    meta.append({
        "state": str(row.get("StateName", "")),
        "crop": str(row.get("Crop", ""))
    })

print(f"Encoding {len(texts)} texts...")
embeddings = model.encode(texts, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

faiss.write_index(index, "data/faiss/faiss_index.bin")
np.save("data/faiss/texts.npy", texts)
np.save("data/faiss/meta.npy", meta)

print("✅ FAISS index built successfully (data/faiss/faiss_index.bin, data/faiss/texts.npy, data/faiss/meta.npy)")
