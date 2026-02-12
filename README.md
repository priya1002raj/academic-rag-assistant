#  Academic RAG-Based Question Answering Assistant

An AI-powered Retrieval-Augmented Generation (RAG) system that answers academic questions using custom PDF documents.  

Built using **FAISS**, **LangChain**, **HuggingFace Embeddings**, and **Groq LLM**, with a clean **Streamlit interface** for interactive querying.


##  Project Overview

This project implements a full Retrieval-Augmented Generation pipeline:

1.  Load academic PDFs  
2.  Split text into semantic chunks  
3.  Generate embeddings  
4.  Store embeddings in FAISS vector database  
5.  Retrieve relevant chunks for a user query  
6.  Generate grounded answers using Groq LLM  
7.  Display answer with source citations  

The system ensures answers are generated **only from the provided documents**, reducing hallucinations.


##  Architecture

```
User Question
      ↓
Retriever (FAISS)
      ↓
Relevant Chunks
      ↓
Groq LLM (llama-3.1-8b-instant)
      ↓
Grounded Answer + Sources
```


##  Tech Stack

- **Python**
- **LangChain**
- **FAISS (Vector Database)**
- **HuggingFace Embeddings (MiniLM)**
- **Groq LLM**
- **Streamlit**
- **Sentence Transformers (Semantic Evaluation)**


##  Project Structure

```
academic-rag-assistant/
│
├── app.py                    # Streamlit UI
├── rag_chat.py               # RAG pipeline logic
├── ingest.py                 # PDF ingestion & vector creation
├── evaluate_rag.py           # Keyword-based evaluation
├── semantic_evaluate.py      # Semantic similarity evaluation
├── check_models.py           # Lists available Groq models
├── requirements.txt
└── .gitignore
```


##  Installation & Setup

###  Clone Repository

```bash
git clone https://github.com/priya1002raj/academic-rag-assistant.git
cd academic-rag-assistant
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Add Environment Variable

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

##  Ingest PDFs

Place your PDFs inside:

```
data/pdfs/
```

Then run:

```bash
python ingest.py
```

This creates the FAISS vector database.

---

##  Run the App

```bash
streamlit run app.py
```

Ask academic questions in the browser interface.

---

##  Evaluation Results

###  Keyword-Based Accuracy
- Achieved **100% accuracy** on baseline academic evaluation set.

###  Semantic Evaluation (Cosine Similarity)
- Average Semantic Accuracy: **82.68%**
- High contextual grounding
- Minimal hallucination observed

---

##  Features

✔ Retrieval-Augmented Generation  
✔ Source citation with page numbers  
✔ Groq ultra-fast inference  
✔ Semantic evaluation pipeline  
✔ Modular and extensible architecture  
✔ Clean Streamlit interface  

---

##  Future Improvements

- Hybrid Search (BM25 + FAISS)
- Cross-Encoder Reranking
- RAGAS Evaluation Integration
- Deployment on HuggingFace Spaces
- Multi-document support
- Conversation memory

---

##  Author

**Priya Raj**  
AI/ML Enthusiast | Data & RAG Systems  

---

##  License

MIT License
