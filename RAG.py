import numpy as np
import requests
import faiss
import streamlit as st
import pickle
from typing import List, Dict, Any
import time
import json

class PersianRAGSystem:
    """سیستم RAG بهینه شده برای زبان فارسی با API Jina"""
    
    def __init__(self, jina_api_key: str):
        """مقداردهی اولیه سیستم RAG"""
        
        if not jina_api_key:
            raise ValueError("❌ API Key مورد نیاز است")
            
        self.jina_api_key = jina_api_key
        self.api_url = "https://api.jina.ai/v1/embeddings"
        
        # متغیرهای ذخیره‌سازی
        self.chunks = []
        self.embeddings = None
        self.vector_index = None
        self.chunk_metadata = []
        
        st.write("✅ سیستم RAG با API Jina مقداردهی شد")
        st.write(f"🌐 پشتیبانی چندزبانه: فارسی، انگلیسی، عربی")

    def _call_jina_api(self, texts: List[str], task: str = "retrieval.passage") -> List[List[float]]:
        """فراخوانی API Jina برای تولید embeddings"""
        
        headers = {
        'Content-Type': 'application/json',  # ✅ تغییر به single quote
        'Authorization': f'Bearer jina_f79a39580af2409c9192df3695351ff6-Up2UwFsKAGW1Az1UoKfK2-bJGzA'  # ✅ تغییر به single quote
        }
        
        embeddings = []
        batch_size = 10  # پردازش دسته‌ای برای کارایی بهتر
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            data = {
                "model": "jina-embeddings-v3",
                "task": task,
                "input": texts,
            }
            
            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                
                result = response.json()
                for item in result["data"]:
                    embeddings.append(item["embedding"])
                    
                # تاخیر کوتاه برای جلوگیری از rate limiting
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                st.write(f"❌ خطا در API call: {e}")
                raise
        
        return embeddings

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """تبدیل چانک‌ها به بردارهای embedding"""
        
        st.write(f"🔄 در حال تولید embeddings برای {len(chunks)} چانک...")
        
        # فراخوانی API Jina
        embeddings_list = self._call_jina_api(chunks, task="retrieval.passage")
        
        # تبدیل به numpy array
        embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # نرمال‌سازی
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        
        st.write(f"✅ تولید embeddings با موفقیت انجام شد")
        st.write(f"📐 ابعاد هر embedding: {embeddings.shape[1]}")
        
        return embeddings

    def build_vector_index(self, embeddings: np.ndarray):
        """ساخت ایندکس FAISS برای جستجوی سریع"""
        
        st.write("🏗️ در حال ساخت ایندکس برداری...")
        
        dimension = embeddings.shape[1]
        
        if len(embeddings) < 1000:
            self.vector_index = faiss.IndexFlatIP(dimension)
        else:
            nlist = min(100, len(embeddings) // 10)
            quantizer = faiss.IndexFlatIP(dimension)
            self.vector_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.vector_index.train(embeddings)
        
        self.vector_index.add(embeddings)
        st.write(f"✅ ایندکس برداری ساخته شد: {self.vector_index.ntotal} بردار")

    def add_documents(self, chunks: List[str]):
        """اضافه کردن اسناد به سیستم RAG"""
        
        self.chunks = chunks
        
        self.chunk_metadata = [
            {
                'chunk_id': i,
                'length': len(chunk),
                'preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
            }
            for i, chunk in enumerate(chunks)
        ]
        
        self.embeddings = self.create_embeddings(chunks)
        self.build_vector_index(self.embeddings)
        
        st.write(f"🎯 {len(chunks)} سند با موفقیت به سیستم اضافه شد")

    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """جستجوی چانک‌های مشابه برای سوال"""
        
        if self.vector_index is None:
            raise ValueError("❌ ابتدا باید اسناد را به سیستم اضافه کنید")
        
        st.write(f"🔍 جستجو برای: {query[:50]}...")
        
        # تبدیل سوال به embedding از طریق API
        query_embeddings_list = self._call_jina_api([query], task="retrieval.passage")  # ✅ تغییر task
        query_embedding = np.array(query_embeddings_list, dtype=np.float32)
        
        # نرمال‌سازی
        norm = np.linalg.norm(query_embedding)
        query_embedding = query_embedding / norm
        
        # جستجو در ایندکس
        scores, indices = self.vector_index.search(query_embedding, top_k)
        
        # آماده‌سازی نتایج
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:
                results.append({
                    'rank': i + 1,
                    'chunk_id': idx,
                    'similarity_score': float(score),
                    'text': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx]
                })
        
        st.write(f"✅ {len(results)} نتیجه مرتبط یافت شد")
        return results

    def save_system(self, filepath: str):
        """ذخیره سیستم RAG برای استفاده مجدد"""
        
        system_data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else None,
            'chunk_metadata': self.chunk_metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(system_data, f)
        
        if self.vector_index is not None:
            faiss.write_index(self.vector_index, filepath.replace('.pkl', '.faiss'))
        
        st.write(f"💾 سیستم در {filepath} ذخیره شد")

    def load_system(self, filepath: str):
        """بارگذاری سیستم RAG ذخیره شده"""
        
        with open(filepath, 'rb') as f:
            system_data = pickle.load(f)
        
        self.chunks = system_data['chunks']
        self.chunk_metadata = system_data['chunk_metadata']
        
        if system_data['embeddings']:
            self.embeddings = np.array(system_data['embeddings'], dtype=np.float32)
            
        faiss_path = filepath.replace('.pkl', '.faiss')
        try:
            self.vector_index = faiss.read_index(faiss_path)
            st.write(f"📂 سیستم از {filepath} بارگذاری شد")
        except:
            st.write("⚠️ فایل ایندکس یافت نشد، ایندکس مجدد ساخته می‌شود...")
            if self.embeddings is not None:
                self.build_vector_index(self.embeddings)