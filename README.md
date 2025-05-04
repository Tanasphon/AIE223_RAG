# AIE223_RAG

## Cat Info RAG Chatbot 🐱

A beautiful, bilingual (Thai/English) chatbot that answers all about cats! Powered by Google Gemini API, Streamlit, and advanced retrieval techniques (RAG: Retrieval-Augmented Generation).

### Features

#### 1. Retrieval (R) - Advanced Document Search
- **Vector Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for high-quality text embeddings
- **ChromaDB Integration**: Efficient vector database for storing and searching document embeddings
- **Smart Chunking**: Automatically splits documents into meaningful sections for better context retrieval
- **Cosine Similarity**: Uses cosine similarity for finding the most relevant document chunks
- **Configurable Search**: Adjustable number of context chunks (default: 10) for optimal response quality

#### 2. Augmentation (A) - Context Enhancement
- **Query Expansion**: Automatically enhances user queries with relevant terms and context
- **Chat History Integration**: Maintains conversation context for more natural interactions
- **Language Detection**: Automatically detects and handles non-English input
- **Quality Assessment**: Evaluates answer quality based on relevance and completeness
- **Report Generation**: Creates detailed reports of each interaction for analysis

#### 3. Generation (G) - Intelligent Response Creation
- **Gemini Pro Integration**: Uses Google's Gemini Pro model for high-quality responses
- **Context-Aware Prompts**: Crafts prompts that include relevant context and chat history
- **Expert Persona**: Guides the model to respond as a cat expert
- **Fallback Handling**: Gracefully handles cases where information is not available
- **Bilingual Support**: Provides responses in both English and Thai

#### 4. User Interface
- **Modern Design**: Clean, responsive Streamlit interface with custom styling
- **Real-time Chat**: Interactive chat interface with message history
- **Sidebar Navigation**: Easy access to example questions and information
- **Fixed Input Bar**: Always-visible input area for seamless interaction
- **Visual Feedback**: Loading indicators and clear message formatting

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Tanasphon/AIE223_RAG.git
cd AIE223_RAG
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your Gemini API key
- Get your API Key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## File Structure
```
AIE223_RAG/
├── app.py                      # Streamlit UI
├── rag_core.py                 # RAG implementation (vector search, LLM, etc.)
├── data/
│   └── cat_knowledge.txt      # Cat knowledge base (English)
├── requirements.txt           # Python dependencies
└── .env                       # Gemini API key configuration
```

---

## Usage
- Open the web app and start chatting about cats!
- Use the sidebar for example questions (English)
- Click to expand/collapse answers, and copy any answer easily

---

## Credits
- Developed for AIE223

## Contact
- tanasphon@bumail.net
- jeehan.sutt@bumail.net 