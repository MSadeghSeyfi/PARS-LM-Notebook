import streamlit as st

class App:
    def __init__(self):
        st.set_page_config(page_title="Persian NotebookLM 📚", page_icon= "content/PARS-LM-NOTEBOOK.png")
        st.title("Persian NotebookLM 📚")
        self.translations = {
            'fa' : {
                'change_language': 'تغییر زبان',
                'current_language': 'زبان فعلی: فارسی',
                'app_title': 'کار با PDF',
                'placeholder_genre': 'لطفا در باره ی PDF سوالی بپرسید .',
            },
            'en' : {
                'change_language': 'Change Language',
                'current_language': 'Current Language: English',
                'app_title': 'work with PDF',
                'placeholder_genre': 'Please ask a question about pdf .',
            }
        }

        if 'language' not in st.session_state:
            st.session_state.language = 'fa'

        self.language = st.session_state.language

        if st.button(self.translations[st.session_state.language]['change_language']):
            st.session_state.language = 'en' if st.session_state.language == 'fa' else "fa"
                
        st.write(self.translations[self.language]['current_language'])

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

    def display_app(self):
            st.header(self.translations[self.language]['app_title'] , divider="red") 


if __name__ == "__main__":
    app = App()
    app.display_app()