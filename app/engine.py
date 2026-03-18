import requests
import json
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import os

class EndeeClient:
    def __init__(self, base_url="http://endee-oss:8080"):
        self.base_url = base_url

    def create_index(self, name, dimension=512):
        url = f"{self.base_url}/api/v1/index/create"
        payload = {
            "name": name,
            "dimension": dimension,
            "metric": "cosine",
            "hnsw_config": {
                "m": 16,
                "ef_construction": 200
            }
        }
        response = requests.post(url, json=payload)
        return response.json()

    def insert_vector(self, index_name, vec_id, vector, metadata=None):
        url = f"{self.base_url}/api/v1/index/{index_name}/insert"
        payload = {
            "id": vec_id,
            "vector": vector.tolist() if isinstance(vector, np.ndarray) else vector,
            "payload": metadata or {}
        }
        response = requests.post(url, json=payload)
        return response.json()

    def search(self, index_name, vector, top_k=5):
        url = f"{self.base_url}/api/v1/index/{index_name}/search"
        payload = {
            "vector": vector.tolist() if isinstance(vector, np.ndarray) else vector,
            "top_k": top_k,
            "ef": 50
        }
        response = requests.post(url, json=payload)
        return response.json()

class VisionEngine:
    def __init__(self, model_name='clip-ViT-B-32'):
        self.model = SentenceTransformer(model_name)
    
    def get_image_embedding(self, image_path):
        img = Image.open(image_path)
        return self.model.encode(img)
    
    def get_text_embedding(self, text):
        return self.model.encode(text)

ENDEE_INDEX = "vision_index"

def initialize_system():
    client = EndeeClient()
    # Try to create index, might already exist
    try:
        client.create_index(ENDEE_INDEX)
    except:
        pass
    return client, VisionEngine()
