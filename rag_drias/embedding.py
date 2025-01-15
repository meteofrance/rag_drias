from typing import List

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from rag_drias.settings import PATH_MODELS

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Embedding(Embeddings):
    """Wrapper class for embedding models
    that can't be instantiated from HuggingFaceEmbeddings.
    Can be easily used with Chroma database.
    """

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(device)

    def encode(self, text):
        text = text.replace("\n", " ")
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        avg_pooling = torch.mean(last_hidden_states, dim=1)
        result = avg_pooling.cpu().numpy()[0].tolist()
        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        docs = [self.encode(text) for text in tqdm(texts, desc="Embedding")]
        return docs

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.encode(text)


def get_embedding(model_name: str = "sentence-camembert-large") -> Embedding:
    print("Loading Camembert...")
    try:
        return Embedding(PATH_MODELS / model_name)
    except OSError:
        return Embedding(model_name)


def test_embedding(embedding: Embedding):
    sentence1 = "This is a cat."
    sentence2 = "This is a dog."
    sentence3 = "I like train."
    embedding1 = embedding.embed_query(sentence1)
    embedding2 = embedding.embed_query(sentence2)
    embedding3 = embedding.embed_query(sentence3)
    print(sentence1, embedding1[:10])
    print(sentence2, embedding2[:10])
    print(sentence3, embedding3[:10])
    print("dot product 1 x 2 : ", np.dot(embedding1, embedding2))
    print("dot product 1 x 3 : ", np.dot(embedding1, embedding3))
