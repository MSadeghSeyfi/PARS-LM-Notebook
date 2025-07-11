from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import StreamlitChatMessageHistory
from langchain_cohere import ChatCohere

class PARSLMNOTEBOOK:
    def __init__(self, language='fa'):
        self.chat_history = StreamlitChatMessageHistory(key="special_app_key")
        self.language = language
        self.translations = self.get_translations()

    def get_translations(self):
            return {
                'fa': {
                    ('system_prompt', f'تو یک مدل فوق العاده قوی هستی بر اساس {self.chathistory} و با استفاده از دانش {self.context} پاسخ سوالات کاربران را به زبان فارسی میدهی !'),
                    ('placeholder', '{chat_history}'),
                    ('human', '{input}'),
                },
                'en': {
                    ('system_prompt', f'You are an incredibly powerful model that answers users’ questions in Persian based on {self.chathistory} and using the knowledge from {self.context}!'),
                    ('placeholder', '{chat_history}'),
                    ('human', '{input}'),
                }
            }    