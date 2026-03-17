# 🎬 YouTube RAG — AI-Powered Video Q&A

Ask questions about any YouTube video in any language. Powered by LLaMA 3.1, Groq, FAISS, and FastAPI.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![LangChain](https://img.shields.io/badge/LangChain-Latest-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1_8B-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Features

- 🎥 **Multi-video support** — Load up to 5 YouTube videos per session and ask questions across all of them
- 🌍 **Multilingual** — Works with videos in any language (Hindi, English, Arabic, Spanish, and 50+ more), always replies in the language you ask in
- 💬 **Conversation memory** — Remembers your previous questions within the session for contextual answers
- 📚 **Source citations** — Shows the exact transcript chunks used to generate each answer
- 📝 **Auto summary** — Instantly generates a video summary and 3 suggested questions after loading
- 📥 **Export chat** — Download your full Q&A conversation as a beautifully styled HTML file (printable as PDF)
- 🔒 **Multi-layer rate limiting** — Per-IP daily limits + per-minute burst protection to prevent abuse
- 👤 **Per-user sessions** — Every user gets fully isolated state via cookies, no interference between users
- ⚠️ **Usage counters** — Real-time video and message usage displayed in the UI header
- 🛡️ **Graceful error handling** — Friendly messages for rate limits, missing transcripts, and API errors

---

## 🏗️ Architecture
```
YouTube URL
    ↓
YouTube Transcript API → raw transcript (any language)
    ↓
LangChain Text Splitter → chunks (1000 chars, 200 overlap)
    ↓
HuggingFace Embeddings (paraphrase-multilingual-MiniLM-L12-v2) → vectors
    ↓
FAISS Vector Store → semantic similarity index
    ↓
User Question → top 4 relevant chunks retrieved
    ↓
Groq LLaMA 3.1 8B Instant + chat history → answer
    ↓
FastAPI → JSON response → HTML/CSS/JS UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | LLaMA 3.1 8B Instant via Groq API |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace) |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Orchestration** | LangChain |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML + CSS + Vanilla JS (no framework) |
| **Rate Limiting** | SlowAPI (per-IP, per-minute + daily) |
| **Session Management** | Cookie-based per-user isolated sessions |
| **Transcript Fetching** | YouTube Transcript API (multilingual) |
| **Env Management** | python-dotenv |

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/youtube-rag.git
cd youtube-rag
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add API keys
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_hf_token_here
```

Get your free API keys:
- **Groq** → [console.groq.com](https://console.groq.com) (free, no credit card)
- **HuggingFace** → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free)

### 5. Run the server
```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 📁 Project Structure
```
youtube-rag/
├── src/
│   ├── __init__.py
│   ├── rag.py          # transcript fetching, chunking, FAISS vector store
│   └── chain.py        # LLM chain, prompts, summary generation
├── templates/
│   └── index.html      # full frontend UI (HTML + CSS + JS)
├── main.py             # FastAPI app, all endpoints, session management
├── requirements.txt    # Python dependencies
├── .env                # API keys (never commit this)
├── .gitignore
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the main UI |
| `POST` | `/load` | Load a YouTube video by URL |
| `POST` | `/ask` | Ask a question about loaded video(s) |
| `POST` | `/clear` | Clear current session |
| `GET` | `/status` | Get current session usage stats |

### Example usage with curl

**Load a video:**
```bash
curl -X POST http://localhost:8000/load \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this video?"}'
```

---

## 🔒 Rate Limiting & Protection

| Limit | Value | Scope |
|---|---|---|
| Video loads per minute | 5 | Per IP |
| Questions per minute | 20 | Per IP |
| Videos per day | 10 | Per IP |
| Questions per day | 100 | Per IP |
| Videos per session | 5 | Per user session |
| Messages per session | 50 | Per user session |
| Max concurrent sessions | 100 | Server-wide |

Limits reset daily at midnight. Users see friendly messages when limits are hit — no raw API errors exposed.

---

## 🌍 Supported Languages

The multilingual embedding model supports 50+ languages including English, Hindi, Arabic, Chinese, Spanish, French, German, Japanese, Korean, Portuguese, Russian, and more.

The LLaMA 3.1 8B model will always respond in the language you ask in, regardless of the transcript language.
