import streamlit as st
from langchain_community.document_loaders import PyPDFium2Loader
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import re
from RAG import PersianRAGSystem
from LLM import PersianLLMGenerator, ConversationManager

class App:
    def __init__(self):
        self.translations = {
            'fa' : {
                'change_language': 'تغییر زبان',
                'current_language': 'زبان فعلی: فارسی',
                'app_title': 'کار با PDF',
                'placeholder_genre': 'لطفا در باره ی PDF سوالی بپرسید .',
                'select_file' : 'يک فايل PDf برگزينيد .'
            },
            'en' : {
                'change_language': 'Change Language',
                'current_language': 'Current Language: English',
                'app_title': 'work with PDF',
                'placeholder_genre': 'Please ask a question about pdf .',
                'select_file' : 'Please select a PDF file .'
            }
        }

        if 'language' not in st.session_state:
            st.session_state.language = 'fa'

        if st.button(self.translations[st.session_state.language]['change_language']):
            st.session_state.language = 'en' if st.session_state.language == 'fa' else "fa"

        self.language = st.session_state.language
        self.text_align, self.direction = self.get_text_alignment()
        self.inject_css()
        self.model = None

    def get_text_alignment(self):
            if self.language == 'fa':
                return 'right', 'rtl'
            else:
                return 'left', 'ltr'    
        
    def inject_css(self):
            st.markdown(f"""
                <style>
                        
                    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500&display=swap');
                    
                    * {{
                        font-family: 'Vazirmatn', sans-serif;
                        direction: {self.direction};
                        text-align: {self.text_align};
                    }}

                </style>
            """, unsafe_allow_html=True)  

    def minimal_persian_fix(self, text):
                # Ascending
                # تصحیح کاراکترهای اشتباه
                char_fixes = {
                    'ͷ': 'ک',  # کاراکتر اشتباه به جای ک
                    'ͺ': 'ک',  # کاراکتر اشتباه دیگر
                    'ͽ': 'گ',  # کاراکتر اشتباه به جای گ
                    'ؤ': 'و',  # و همزه‌دار عربی به فارسی
                    'ى': 'ی',  # ی عربی به فارسی
                    'ة': 'ه',  # ته مربوطه عربی به فارسی
                    'ك': 'ک',  # کاف عربی به فارسی
                    'ي': 'ی',  # یا عربی به فارسی
                }

                # جایگزینی کاراکترها
                for wrong, correct in char_fixes.items():
                    text = text.replace(wrong, correct)

                # تصحیح مشکل خاص chr(876) + فاصله
                text = text.replace(chr(876) + ' ', 'ی ')
                text = text.replace(chr(876) + chr(13), 'ی ')
                text = text.replace(chr(876), 'ی')  # برای موارد دیگر chr(876)

                return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
            try:
                # Load the PDF using PyPDFium2Loader
                loader = PyPDFium2Loader(pdf_path)
                documents = loader.load()
                text = ""

                # Iterate through all documents (each represents a page)
                for doc in documents:
                    original_content = doc.page_content
                    corrected_content = self.minimal_persian_fix(original_content)
                    text += corrected_content + "\n"  # Add newline between pages
                
                return text
            except Exception as e:
                return f"Error extracting text: {str(e)}"
        

    def advanced_persian_chunking(self, text: str) -> List[str]:
        """
        پیاده‌سازی پیشرفته تقسیم متن فارسی به چانک‌های بهینه
        """
        
        # 1. پیش‌پردازش اولیه متن
        def preprocess_text(text: str) -> str:
            # حذف فاصله‌های اضافی
            text = re.sub(r'\s+', ' ', text)
            # تنظیم فاصله‌ها قبل و بعد علائم نگارشی فارسی
            text = re.sub(r'\s*([.!?؟:؛،])\s*', r'\1 ', text)
            # حذف خطوط خالی اضافی
            text = re.sub(r'\n\s*\n', '\n\n', text)
            return text.strip()
        
        # 2. تعریف جداکننده‌های بهینه برای فارسی
        persian_separators = [
            "\n\n\n",          # پاراگراف‌های جدا
            "\n\n",            # پاراگراف‌ها
            "\n",              # خطوط جدید
            "؟",               # علامت سوال فارسی
            "!",               # علامت تعجب
            ".",               # نقطه
            "؛",               # نقطه ویرگول فارسی
            ":",               # دو نقطه
            "،",               # ویرگول فارسی
            ",",               # ویرگول انگلیسی
            " و ",             # حرف ربط فارسی
            " که ",            # کلمات ربط فارسی
            " در ",
            " از ",
            " به ",
            " با ",
            " ",               # فاصله
            ""                 # کاراکتر خالی
        ]
        
        # 3. تنظیمات بهینه برای متن فارسی
        chunk_size = 800       # اندازه مناسب برای فارسی
        chunk_overlap = 150    # overlap کافی برای حفظ context
        
        # 4. ایجاد text splitter بهینه
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=persian_separators,
            keep_separator=True,    # حفظ جداکننده‌ها
            add_start_index=True,   # اضافه کردن ایندکس شروع
        )
        
        # 5. پیش‌پردازش متن
        processed_text = preprocess_text(text)
        
        # 6. تقسیم به چانک‌ها
        chunks = text_splitter.split_text(processed_text)
        
        # 7. پس‌پردازش چانک‌ها
        def postprocess_chunk(chunk: str) -> str:
            # حذف فاصله‌های ابتدا و انتها
            chunk = chunk.strip()
            # اطمینان از وجود حداقل طول
            if len(chunk) < 50:
                return None
            # اضافه کردن نقطه در صورت نیاز
            if not chunk.endswith(('.', '!', '؟', ':', '؛')):
                chunk += '.'
            return chunk
        
        # 8. فیلتر و بهینه‌سازی چانک‌ها
        final_chunks = []
        for chunk in chunks:
            processed_chunk = postprocess_chunk(chunk)
            if processed_chunk:
                final_chunks.append(processed_chunk)
        
        # 9. اطلاعات آماری
        st.write(f"📊 آمار تقسیم‌بندی:")
        st.write(f"   • تعداد کل چانک‌ها: {len(final_chunks)}")
        st.write(f"   • متوسط طول چانک: {sum(len(chunk) for chunk in final_chunks) // len(final_chunks) if final_chunks else 0} کاراکتر")
        st.write(f"   • کوتاه‌ترین چانک: {min(len(chunk) for chunk in final_chunks) if final_chunks else 0} کاراکتر")
        st.write(f"   • بلندترین چانک: {max(len(chunk) for chunk in final_chunks) if final_chunks else 0} کاراکتر")
        
        return final_chunks

    # 10. تابع اضافی برای بررسی کیفیت چانک‌ها
    def analyze_chunks_quality(self, chunks: List[str]) -> dict:
        """تحلیل کیفیت چانک‌های تولید شده"""
        
        analysis = {
            'total_chunks': len(chunks),
            'avg_length': sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            'min_length': min(len(chunk) for chunk in chunks) if chunks else 0,
            'max_length': max(len(chunk) for chunk in chunks) if chunks else 0,
            'empty_chunks': sum(1 for chunk in chunks if len(chunk.strip()) == 0),
            'short_chunks': sum(1 for chunk in chunks if len(chunk) < 100),
            'optimal_chunks': sum(1 for chunk in chunks if 200 <= len(chunk) <= 1000),
        }
        
        return analysis
    
    def Grok_API_key(self):
        api_key = st.sidebar.text_input("Grok API Key", type="password")
        if not api_key:
            st.info("Please enter Grok API key to continue.")
            st.stop()
        return api_key

    def display_app(self):
        st.set_page_config(page_title="Persian NotebookLM 📚", page_icon= "content/PARS-LM-NOTEBOOK.png")
        st.title("Persian NotebookLM 📚")
        st.write(self.translations[self.language]['current_language'])
        st.header(self.translations[self.language]['app_title'], divider="red") 
        Grok_api = self.Grok_API_key()  # دریافت API key از کاربر
        if Grok_api:
            st.session_state["grok_api_key"] = Grok_api  # ذخیره در session state
            
        uploaded_file = st.file_uploader(f"{self.translations[self.language]['select_file']} PDF", type=["pdf"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            file_path = Path("temp.pdf")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            extracted_text = self.extract_text_from_pdf(file_path)
            
            # ✅ مرحله جدید: تقسیم متن به چانک‌ها
            progress_bar.progress(50)
            status_text.text("در حال تقسیم متن به بخش‌های کوچکتر...")

            #  # Display extracted text
            st.subheader("متن استخراج‌شده:")
            st.text_area("متن", extracted_text, height=400)

            text_chunks = self.advanced_persian_chunking(extracted_text)

            progress_bar.progress(75)
            status_text.text("در حال ایجاد بردارهای embedding...")

            rag_system = get_rag_system()
            rag_system.add_documents(text_chunks)
            
            progress_bar.progress(100)
            status_text.text("سیستم RAG آماده است!")
            
            # ✅ مرحله جدید: LLM Integration
            st.subheader("💬 پرسش و پاسخ هوشمند")
            
            # ایجاد session state برای تاریخچه
            if 'conversation_manager' not in st.session_state:
                st.session_state.conversation_manager = ConversationManager()
            
            # دریافت سوال از کاربر
            user_question = st.text_input("🤔 سوال خود را در مورد محتوای PDF بپرسید:")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("❓ پرسیدن", type="primary")
            with col2:
                clear_button = st.button("🗑️ پاک کردن تاریخچه")
            
            if clear_button:
                st.session_state.conversation_manager.clear_history()
                st.success("✅ تاریخچه مکالمات پاک شد")
            
            if ask_button and user_question:
                with st.spinner("🤖 در حال پردازش سوال..."):
                    # دریافت LLM generator
                    llm_generator = get_llm_generator()
                    
                    # دریافت تاریخچه اخیر
                    recent_history = st.session_state.conversation_manager.get_recent_history()
                    
                    # پردازش سوال با RAG
                    response = llm_generator.query_with_rag(
                        rag_system=rag_system,
                        query=user_question,
                        top_k=5,
                        conversation_history=recent_history
                    )
                    
                    # نمایش پاسخ
                    st.subheader("🎯 پاسخ:")
                    st.write(response['answer'])
                    
                    # نمایش اطلاعات اضافی
                    if 'model_info' in response:
                        with st.expander("📊 جزئیات تولید پاسخ"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("منابع استفاده شده", response['sources_used'])
                            with col2:
                                st.metric("زمان تولید", f"{response['generation_time']}s")
                            with col3:
                                st.metric("توکن‌های کل", response['model_info']['tokens_used']['total_tokens'])
                    
                    # نمایش منابع مرجع
                    if 'retrieval_context' in response and response['retrieval_context']:
                        with st.expander("📚 منابع مرجع"):
                            for source in response['retrieval_context']:
                                st.write(f"**منبع {source['source_id']}** (امتیاز: {source['similarity']:.3f})")
                                st.write(source['preview'])
                                st.divider()
                    
                    # ذخیره در تاریخچه
                    st.session_state.conversation_manager.add_exchange(
                        user_question, 
                        response['answer']
                    )
            
            # نمایش تاریخچه مکالمات
            if st.session_state.conversation_manager.conversation_history:
                with st.expander("📜 تاریخچه مکالمات"):
                    for i, exchange in enumerate(reversed(st.session_state.conversation_manager.conversation_history)):
                        st.write(f"**سوال {len(st.session_state.conversation_manager.conversation_history)-i}:** {exchange['user']}")
                        st.write(f"**پاسخ:** {exchange['assistant']}")
                        st.divider()
            
            # Clean up temporary file
            file_path.unlink()                           

@st.cache_resource
def get_rag_system():
    JINA_API_KEY = "jina_f79a39580af2409c9192df3695351ff6-Up2UwFsKAGW1Az1UoKfK2-bJGzA"  # ✅ دریافت API key
    return PersianRAGSystem(JINA_API_KEY)      # ✅ پاس دادن API key

@st.cache_resource  
def get_llm_generator():
    GROQ_API_KEY = st.session_state['grok_api_key']  # جایگزین با API key واقعی
    return PersianLLMGenerator(GROQ_API_KEY)

if __name__ == "__main__":
    app = App()
    app.display_app()