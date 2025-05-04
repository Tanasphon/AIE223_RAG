import os
import warnings
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import requests
import json
from datetime import datetime
from typing import List, Dict, Any
import spacy
import chromadb
from chromadb.config import Settings

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

__all__ = ['RAGSystem']

class RAGSystem:
    def __init__(self):
        # === R: Retrieval Initialization ===
        # Initialize Gemini API configuration
        self.api_key = "AIzaSyAqFTZqLMHvviA5juyzUz-RsHoiZYlj-LE"
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Initialize NLP models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize ChromaDB client and collection
        self.chroma_client = chromadb.Client(Settings(persist_directory="vector_db"))
        self.chroma_collection = self.chroma_client.get_or_create_collection("cat_docs")
        
        # Initialize text data
        self.documents = []
        
        # Initialize chat history with enhanced memory
        self.chat_history = []
        self.conversation_summary = ""
        
        # Initialize query enhancement
        self.query_expansion_terms = {
            'cat': ['feline', 'kitten', 'breed', 'pet'],
            'breed': ['type', 'species', 'variety', 'kind'],
            'characteristic': ['trait', 'feature', 'quality', 'attribute']
        }
        
        # Create report directory
        self.report_dir = "report"
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
        
        # Load cat data
        self.load_cat_data()
    
    def load_cat_data(self):
        """R: Retrieval - Load cat data and store in ChromaDB"""
        try:
            with open('data\cat_data.txt', 'r', encoding='utf-8') as file:
                content = file.read()
                # Split into paragraphs
                self.documents = [p.strip() for p in content.split('\n\n') if p.strip()]
                # Create embeddings for all documents
                embeddings = self.embedding_model.encode(self.documents).tolist()
                # Delete and recreate collection for a fresh start
                self.chroma_client.delete_collection("cat_docs")
                self.chroma_collection = self.chroma_client.create_collection("cat_docs")
                self.chroma_collection.add(
                    documents=self.documents,
                    embeddings=embeddings,
                    ids=[str(i) for i in range(len(self.documents))]
                )
        except FileNotFoundError:
            print("Warning: cat_data.txt not found. Please ensure the file exists in the same directory.")
    
    def enhance_query(self, query: str) -> str:
        """A: Augmentation - Expand query using synonyms, chat history, and NLP key phrases"""
        # 1. Start with original query tokens
        tokens = query.lower().split()
        enhanced_terms = set(tokens)
        # 2. Add synonyms from expansion terms
        for token in tokens:
            if token in self.query_expansion_terms:
                enhanced_terms.update(self.query_expansion_terms[token])
        # 3. Add noun phrases from spaCy (query)
        doc = self.nlp(query)
        enhanced_terms.update([chunk.text.lower() for chunk in doc.noun_chunks])
        # 4. Add noun phrases from conversation summary
        if self.conversation_summary:
            doc_sum = self.nlp(self.conversation_summary)
            enhanced_terms.update([chunk.text.lower() for chunk in doc_sum.noun_chunks])
        # 5. Add keywords from all chat history (user+assistant)
        for msg in self.chat_history[-len(self.chat_history):]:
            doc_hist = self.nlp(msg['content'])
            enhanced_terms.update([chunk.text.lower() for chunk in doc_hist.noun_chunks])
        # 6. Return as a single string
        return ' '.join(sorted(enhanced_terms))
    
    def update_conversation_summary(self, query: str, answer: str):
        """A: Augmentation - Update conversation summary for context"""
        # Use NLP to extract key information
        doc = self.nlp(f"{query} {answer}")
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Update summary with new information
        self.conversation_summary = ' '.join(key_phrases)
        if len(self.conversation_summary) > 200:  # Limit summary length
            self.conversation_summary = self.conversation_summary[:200] + "..."
    
    def search_documents(self, query: str, k: int = 10) -> list:
        """R: Retrieval - Search documents using ChromaDB vector search"""
        enhanced_query = self.enhance_query(query)
        query_embedding = self.embedding_model.encode([enhanced_query]).tolist()
        results = self.chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=k
        )
        # Return the matched documents
        return results.get('documents', [[]])[0] if results.get('documents') else []
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """G: Generation - Generate answer with enhanced prompt engineering"""
        if not context:
            return "I don't have enough information about that specific topic."
        
        # Add to chat history
        self.add_to_history("user", query)
        
        # Update conversation summary
        self.update_conversation_summary(query, "")
        
        # Prepare enhanced prompt
        prompt = self._create_enhanced_prompt(query, context)
        
        # Make API request
        try:
            response = self._make_api_request(prompt)
            answer = self._process_api_response(response)
            
            # Update conversation summary with answer
            self.update_conversation_summary(query, answer)
            
            # Save to report
            self.save_to_report(query, context, answer)
            return answer
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.add_to_history("assistant", error_msg)
            self.save_to_report(query, context, error_msg)
            return error_msg
    
    def _create_enhanced_prompt(self, query: str, context: List[str]) -> str:
        """A/G: Augmentation/Generation - Create enhanced prompt with conversation context and context chunks"""
        all_user_questions = self.get_all_user_questions()
        full_history = self.get_full_chat_history()
        return f"""You are a cat breed expert assistant. Answer questions based on the provided context.

Below is the full chat history (user and assistant) so far:
{full_history}

All previous user questions:
{all_user_questions}

Current conversation focus:
{self.conversation_summary}

Relevant information:
{context}

Question: {query}

Guidelines:
1. Always answer as accurately and completely as possible, using only the provided context and chat history.
2. If the user asks about their previous questions, list them in order and provide clear references.
3. If the user asks about previous answers, summarize or quote them as needed.
4. If the question is unclear, politely ask for clarification or provide a general overview.
5. If you do not have enough information, state so honestly and suggest what the user could ask next.
6. Use clear, concise, and friendly language. Avoid jargon unless you explain it.
7. Always consider the full conversation context and avoid repeating information unnecessarily.
8. Engage the user by offering to answer follow-up questions or provide more details if needed.
9. If the user asks for a list, use bullet points or numbering for clarity.
10. Never make up facts. If unsure, say so and offer to help with related information.
"""
    
    def _make_api_request(self, prompt: str) -> requests.Response:
        """G: Generation - Make API request to LLM"""
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        
        return requests.post(
            f"{self.api_url}?key={self.api_key}",
            headers=headers,
            data=json.dumps(data)
        )
    
    def _process_api_response(self, response: requests.Response) -> str:
        """G: Generation - Process LLM API response"""
        if response.status_code == 200:
            result = response.json()
            answer = result['candidates'][0]['content']['parts'][0]['text']
            self.add_to_history("assistant", answer)
            return answer
        else:
            raise Exception(f"API request failed with status code {response.status_code}")
    
    def check_answer_quality(self, answer: str, query: str, context: list) -> dict:
        """A: Augmentation - Check answer quality"""
        quality = {
            "is_empty": False,
            "is_too_short": False,
            "is_off_topic": False,
            "is_uncertain": False,
            "score": 1.0  # 1.0 = ดี, 0.0 = แย่
        }
        # 1. ตรวจสอบว่าคำตอบว่างเปล่า
        if not answer or not answer.strip():
            quality["is_empty"] = True
            quality["score"] -= 0.5
        # 2. ตรวจสอบว่าคำตอบสั้นเกินไป
        if len(answer.strip().split()) < 15:
            quality["is_too_short"] = True
            quality["score"] -= 0.2
        # 3. ตรวจสอบว่าคำตอบไม่เกี่ยวข้อง (off-topic)
        topic_keywords = ["cat", "breed", "characteristic", "pet", "feline"]
        if not any(word in answer.lower() for word in topic_keywords):
            quality["is_off_topic"] = True
            quality["score"] -= 0.3
        # 4. ตรวจสอบว่าคำตอบมีความไม่มั่นใจ/ไม่รู้
        uncertain_phrases = [
            "i don't know", "not sure", "cannot answer", "ไม่มีข้อมูล", "ไม่ทราบ", "ขออภัย", "sorry"
        ]
        if any(phrase in answer.lower() for phrase in uncertain_phrases):
            quality["is_uncertain"] = True
            quality["score"] -= 0.3
        # 5. (Optional) ตรวจสอบว่าคำตอบมีข้อมูลจาก context หรือไม่
        if context and not any(ctx.lower() in answer.lower() for ctx in context):
            quality["score"] -= 0.1
        # Ensure score is between 0 and 1
        quality["score"] = max(0.0, min(1.0, quality["score"]))
        return quality
    
    def save_to_report(self, query: str, context: List[str], answer: str):
        """A: Augmentation - Save chat session to report file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # ตรวจสอบคุณภาพคำตอบ
        quality = self.check_answer_quality(answer, query, context)
        with open(os.path.join(self.report_dir, f"chat_{timestamp}.txt"), "w", encoding="utf-8") as f:
            f.write("=== Chat Session ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("=== Query ===\n")
            f.write(f"{query}\n\n")
            f.write("=== Enhanced Query ===\n")
            f.write(f"{self.enhance_query(query)}\n\n")
            f.write("=== Context Chunks (from embedding search) ===\n")
            for i, doc in enumerate(context, 1):
                f.write(f"\nContext chunk {i}:\n{doc}\n")
            f.write("\n")
            f.write("=== Answer ===\n")
            f.write(f"{answer}\n")
            f.write("\n")
            f.write("=== Answer Quality ===\n")
            for k, v in quality.items():
                f.write(f"{k}: {v}\n")
            f.write("\n" + "="*50 + "\n")
    
    def add_to_history(self, role: str, content: str):
        """A: Augmentation - Add a message to chat history"""
        self.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Keep only the last 10 messages to prevent memory issues
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
    
    def get_history_context(self) -> str:
        """A: Augmentation - Get the last 5 messages as context"""
        if not self.chat_history:
            return ""
            
        history_context = "Previous conversation:\n"
        for msg in self.chat_history[-5:]:
            history_context += f"{msg['role']}: {msg['content']}\n"
        return history_context
    
    def get_all_user_questions(self) -> str:
        """A: Augmentation - Get all user questions from chat history"""
        return "\n".join([msg["content"] for msg in self.chat_history if msg["role"] == "user"])
    
    def get_full_chat_history(self) -> str:
        """A: Augmentation - Get the full chat history"""
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.chat_history])
    
    def query(self, question: str, chat_history: list = None) -> str:
        """RAG Pipeline: Retrieval, Augmentation, and Generation"""
        # If chat_history is provided, use it to set self.chat_history
        if chat_history is not None:
            self.chat_history = chat_history.copy()
        # Search for relevant documents
        relevant_docs = self.search_documents(question)
        # Generate answer
        answer = self.generate_answer(question, relevant_docs)
        return answer