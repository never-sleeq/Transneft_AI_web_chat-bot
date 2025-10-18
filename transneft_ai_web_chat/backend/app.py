from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import sys

sys.path.append(os.path.dirname(__file__))

from QAv3 import TransneftRetriever, Saiga2Reader

app = FastAPI(title="Transneft AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8005",
        "http://127.0.0.1:8005"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/api/qa")
async def options_qa():
    return {"message": "OK"}

print("🔄 Загружаем модели...")

retriever = TransneftRetriever(
    triplets_path="triplets.json",
    index_path="index.faiss",
    contexts_path="chunks.json",
    model_path="./bge-m3"
)

reader = Saiga2Reader(model_path="./model-q4_K.gguf")

print("✅ Модели готовы!")

class QARequest(BaseModel):
    question: str
    top_k: int = 3

class QAResponse(BaseModel):
    answer: str

@app.post("/api/qa", response_model=QAResponse)
async def qa(request: QARequest):
    try:
        results = retriever.retrieve(request.question, top_k=request.top_k)
        contexts = [r["context"] for r in results]
        answer = reader.generate_answer(request.question, contexts)
        return QAResponse(answer=answer)
    except Exception as e:
        print(f"Ошибка в ИИ: {e}")
        raise HTTPException(status_code=500, detail="Не удалось сгенерировать ответ")