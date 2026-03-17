# 🎬 YouTube RAG — AI-Powered Video Q&A

Ask questions about any YouTube video in any language. Powered by LLaMA 3.1, Groq, FAISS, and FastAPI.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![LangChain](https://img.shields.io/badge/LangChain-Latest-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-purple)

---

## 🚀 Features

- 🎥 **Multi-video support** — Load up to 5 YouTube videos per session
- 🌍 **Multilingual** — Works with videos in any language, answers in your language
- 💬 **Conversation memory** — Remembers previous questions in the session
- 📚 **Source citations** — Shows which transcript chunks the answer came from
- 📝 **Auto summary** — Generates video summary + suggested questions on load
- 📥 **Export chat** — Download full conversation as a styled HTML file
- 🔒 **Rate limiting** — Per-IP daily limits + per-minute burst protection
- 👤 **Per-user sessions** — Each user gets isolated state via cookies

---

## 🏗️ Architecture
```
YouTube URL
    ↓
YouTube Transcript API → raw transcript (any language)
    ↓
LangChain Text Splitter → chunks (1000 tokens, 200 overlap)
    ↓
HuggingFace Embeddings (multilingual-MiniLM-L12-v2) → vectors
    ↓
FAISS Vector Store → semantic search index
    ↓
User Question → top 4 relevant chunks retrieved
    ↓
Groq LLaMA 3.1 8B + conversation history → answer
    ↓
FastAPI → JSON response → HTML/CSS/JS UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | LLaMA 3.1 8B Instant via Groq |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Orchestration** | LangChain |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML + CSS + Vanilla JS |
| **Rate Limiting** | SlowAPI |
| **Transcript** | YouTube Transcript API |

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/youtube-rag.git
cd youtube-rag
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add API keys
Create a `.env` file:
```
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_hf_token_here
```

Get free keys:
- Groq → [console.groq.com](https://console.groq.com)
- HuggingFace → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5. Run
```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📁 Project Structure
```
youtube-rag/
├── src/
│   ├── __init__.py
│   ├── rag.py          # transcript fetching, chunking, vector store
│   └── chain.py        # LLM chain, prompts, summary generation
├── templates/
│   └── index.html      # full frontend UI
├── main.py             # FastAPI app, endpoints, session management
├── requirements.txt
├── .env                # API keys (never commit this)
└── .gitignore
```

---

## 🔒 Rate Limiting

| Limit | Value |
|---|---|
| Video loads per minute (per IP) | 5 |
| Questions per minute (per IP) | 20 |
| Videos per day (per IP) | 10 |
| Questions per day (per IP) | 100 |
| Videos per session | 5 |
| Messages per session | 50 |

---

## 📄 License

MIT License — free to use and modify.