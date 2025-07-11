from groq import Groq
from typing import List, Dict, Any, Optional
import time

class PersianLLMGenerator:
    """مولد پاسخ هوشمند فارسی با استفاده از مدل qwen3-32b"""
    
    def __init__(self, groq_api_key: str):
        """مقداردهی اولیه مولد LLM"""
        
        if not groq_api_key:
            raise ValueError("❌ API Key Groq مورد نیاز است")
            
        self.groq_client = Groq(api_key=groq_api_key)
        self.model_name = "qwen/qwen3-32b"
        
        # تنظیمات پیشرفته برای بهینه‌سازی فارسی
        self.generation_config = {
            "temperature": 0.2,        # کاهش تصادفی برای دقت بیشتر
            "max_tokens": 2048,        # طول مناسب برای پاسخ‌های تفصیلی فارسی
            "top_p": 0.9,              # کنترل تنوع پاسخ
            "frequency_penalty": 0.1,   # جلوگیری از تکرار
            "presence_penalty": 0.1     # تشویق به تنوع محتوا
        }
        
        # print("✅ مولد LLM فارسی با موفقیت مقداردهی شد")
        # print(f"🤖 مدل: {self.model_name}")

    def _build_persian_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]], 
                            conversation_history: Optional[List[Dict]] = None) -> str:
        """ساخت prompt بهینه شده برای زبان فارسی"""
        
        # ساخت context از چانک‌های بازیابی شده
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            similarity_score = chunk.get('similarity_score', 0)
            text = chunk.get('text', '')
            context_parts.append(f"منبع {i} (امتیاز: {similarity_score:.3f}):\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # پایه prompt برای فارسی
        base_prompt = f"""تو یک دستیار هوشمند فارسی هستی که بر اساس اسناد ارائه شده پاسخ دقیق و جامع می‌دهی.

📋 **اسناد مرجع:**
{context}

🎯 **رهنمودهای مهم:**
1. پاسخت را فقط بر اساس اسناد ارائه شده بده
2. اگر اطلاعات کافی در اسناد نیست، صادقانه بگو
3. پاسخ را به زبان فارسی و با ساختار منطقی ارائه کن
4. از منابع موجود نقل قول مناسب ارائه ده
5. اگر سوال چندبخشی است، پاسخ را بخش‌بندی کن

❓ **سوال کاربر:** {query}

💡 **پاسخ جامع:**"""

        # اضافه کردن تاریخچه مکالمه در صورت وجود
        if conversation_history:
            history_text = "\n".join([
                f"کاربر: {msg['user']}\nدستیار: {msg['assistant']}" 
                for msg in conversation_history[-3:]  # آخرین 3 مکالمه
            ])
            base_prompt = f"""📜 **تاریخچه مکالمه اخیر:**
{history_text}

{base_prompt}"""

        return base_prompt

    def generate_response(self, query: str, retrieved_chunks: List[Dict[str, Any]], 
                         conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """تولید پاسخ هوشمند بر اساس context بازیابی شده"""
        
        # print(f"🤖 در حال تولید پاسخ برای: {query[:50]}...")
        
        # ساخت prompt
        prompt = self._build_persian_prompt(query, retrieved_chunks, conversation_history)
        
        try:
            # ارسال درخواست به Groq
            start_time = time.time()
            
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "تو یک دستیار هوشمند فارسی هستی که پاسخ‌های دقیق و مفصل ارائه می‌دهی. همیشه بر اساس منابع ارائه شده پاسخ بده و اگر اطلاعات کافی نداری، صادقانه اعلام کن."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                **self.generation_config
            )
            
            end_time = time.time()
            
            # استخراج پاسخ
            generated_answer = response.choices[0].message.content
            
            # آماده‌سازی نتیجه نهایی
            result = {
                'answer': generated_answer,
                'sources_used': len(retrieved_chunks),
                'generation_time': round(end_time - start_time, 2),
                'model_info': {
                    'model': self.model_name,
                    'tokens_used': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                },
                'retrieval_context': [
                    {
                        'source_id': i,
                        'similarity': chunk.get('similarity_score', 0),
                        'preview': chunk.get('text', '')[:200] + "..."
                    }
                    for i, chunk in enumerate(retrieved_chunks, 1)
                ]
            }
            
            # print(f"✅ پاسخ با موفقیت تولید شد")
            # print(f"⏱️ زمان تولید: {result['generation_time']} ثانیه")
            # print(f"📊 توکن‌های استفاده شده: {result['model_info']['tokens_used']['total_tokens']}")
            
            return result
            
        except Exception as e:
            print(f"❌ خطا در تولید پاسخ: {str(e)}")
            return {
                'answer': f"⚠️ متأسفانه در تولید پاسخ خطایی رخ داد: {str(e)}",
                'sources_used': 0,
                'generation_time': 0,
                'error': str(e)
            }

    def query_with_rag(self, rag_system, query: str, top_k: int = 5, 
                      conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """پردازش کامل سوال با سیستم RAG"""
        
        # print(f"🔄 پردازش سوال RAG: {query}")
        
        try:
            # مرحله 1: بازیابی اسناد مرتبط
            retrieved_chunks = rag_system.search_similar_chunks(query, top_k=top_k)
            
            if not retrieved_chunks:
                return {
                    'answer': "⚠️ متأسفانه اسناد مرتبطی برای پاسخ به سوال شما یافت نشد.",
                    'sources_used': 0,
                    'retrieval_context': []
                }
            
            # مرحله 2: تولید پاسخ
            response = self.generate_response(query, retrieved_chunks, conversation_history)
            
            return response
            
        except Exception as e:
            print(f"❌ خطا در پردازش RAG: {str(e)}")
            return {
                'answer': f"⚠️ خطا در پردازش سوال: {str(e)}",
                'error': str(e)
            }

# کلاس مدیریت مکالمه
class ConversationManager:
    """مدیریت تاریخچه مکالمات"""
    
    def __init__(self, max_history: int = 10):
        self.conversation_history = []
        self.max_history = max_history
    
    def add_exchange(self, user_query: str, assistant_response: str):
        """اضافه کردن مکالمه جدید"""
        self.conversation_history.append({
            'user': user_query,
            'assistant': assistant_response,
            'timestamp': time.time()
        })
        
        # حفظ حداکثر تعداد مکالمه
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_recent_history(self, count: int = 3) -> List[Dict]:
        """دریافت آخرین مکالمات"""
        return self.conversation_history[-count:] if self.conversation_history else []
    
    def clear_history(self):
        """پاک کردن تاریخچه"""
        self.conversation_history.clear()