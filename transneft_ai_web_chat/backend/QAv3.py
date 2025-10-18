import json
import os
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel
from llama_cpp import Llama


class TransneftRetriever:
    def __init__(
        self,
        triplets_path: str = "triplets.json",
        index_path = "index.faiss",
        contexts_path: str = "chunks.json",
        model_path: str = "bge-m3"
    ):
        self.triplets_path = triplets_path
        self.index_path = index_path
        self.contexts_path = contexts_path
        self.model_path = model_path

        with open(triplets_path, encoding="utf-8") as f:
            triplets = json.load(f)
        self.contexts = [t["context"] for t in triplets]

        print("Загрузка модели bge-m3...")
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=False,
            device="cpu"
        )

        if os.path.exists(index_path):
            print("Загружаем существующий FAISS-индекс...")
            self.index = faiss.read_index(index_path)
        else:
            print("Создаём новый FAISS-индекс...")
            self._build_index()

    def _build_index(self):
        print("Генерация эмбеддингов...")
        embeddings = self.model.encode(
            self.contexts,
            batch_size=16,
            max_length=8192, 
            return_dense=True,
            return_sparse=False, 
            return_colbert_vecs=False
        )["dense_vecs"]

        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        index_dir = os.path.dirname(self.index_path)
        if index_dir:  
            os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.contexts_path, "w", encoding="utf-8") as f:
            json.dump(self.contexts, f, ensure_ascii=False, indent=2)
        print(f"Индекс сохранён: {self.index_path}")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        q_emb = self.model.encode(
            [query],
            batch_size=1,
            max_length=8192,
            return_dense=True
        )["dense_vecs"]
        q_emb = np.array(q_emb, dtype=np.float32)
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "context": self.contexts[idx],
                "score": float(score),
                "index": int(idx)
            })
        return results

    def evaluate_retrieval(self, top_k: int = 10) -> dict:
        """Оценка качества: MRR@10, Hit Rate@10"""
        with open(self.triplets_path, encoding="utf-8") as f:
            triplets = json.load(f)

        reciprocal_ranks = []
        hit_count = 0

        for triplet in triplets:
            question = triplet["question"]
            true_context = triplet["context"]
            results = self.retrieve(question, top_k=top_k)

            rank = None
            for i, res in enumerate(results, 1):
                if res["context"] == true_context:
                    rank = i
                    break

            if rank:
                reciprocal_ranks.append(1 / rank)
                hit_count += 1
            else:
                reciprocal_ranks.append(0)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        hit_rate = hit_count / len(triplets)

        return {
            "mrr@10": round(mrr, 4),
            "hit_rate@10": round(hit_rate, 4)
        }


class Saiga2Reader:
    def __init__(self, model_path: str = "model-q4_K.gguf"):
        print("Загрузка генеративного ридера (Saiga2-7B)...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,          
            n_threads=4,         
            n_gpu_layers=0,      
            verbose=False
        )

    def generate_answer(self, question: str, contexts: list[str], max_tokens: int = 400) -> str:

        context_text = " ".join(contexts)
        
        prompt = f"""Ты — официальный цифровой консультант ПАО «Транснефть». 
Ответь на вопрос, используя ТОЛЬКО информацию из приведённого контекста. 
Не придумывай ничего.

Контекст: {context_text}
Вопрос: {question}
Ответ:"""
        
        # Генерация
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            echo=False,
            temperature=0.1,    
            top_p=0.9
        )
        
        answer = output["choices"][0]["text"].strip()
        return answer if answer else "Информация отсутствует."


if __name__ == "__main__":
    # Инициализация
    retriever = TransneftRetriever()
    reader = Saiga2Reader()

    # Пример полного QA-цикла
    query = "В результате чего сформирован уставной капитал Транснефти?"
    print(f"\n❓ Вопрос: {query}")

    # 1. Ретривер: находим контексты
    results = retriever.retrieve(query, top_k=3)
    contexts = [r["context"] for r in results]

    # 2. Ридер: генерируем ответ
    answer = reader.generate_answer(query, contexts)
    print(f"\nОтвет: {answer}")

    # Опционально: показать использованные контексты
    print("\nИспользованные контексты:")
    for i, ctx in enumerate(contexts, 1):
        print(f"{i}. {ctx[:150]}...")