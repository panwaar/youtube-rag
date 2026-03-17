import os
import uuid
from collections import defaultdict
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Cookie
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.rag import build_vectorstore, add_video_to_store
from src.chain import build_chain, generate_summary_and_questions

# --- Rate limiter ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory="templates")

# --- Limits ---
MAX_VIDEOS_PER_SESSION = 5
MAX_MESSAGES_PER_SESSION = 50
MAX_VIDEOS_PER_IP_PER_DAY = 10
MAX_QUESTIONS_PER_IP_PER_DAY = 100

# --- In-memory stores ---
sessions = {}
ip_usage = defaultdict(lambda: {
    "videos": 0,
    "questions": 0,
    "date": datetime.now().date()
})

# --- Helpers ---
def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "vector_store": None,
            "retriever": None,
            "chain": None,
            "chat_history": [],
            "loaded_videos": []
        }
    return sessions[session_id]

def get_ip_usage(ip: str) -> dict:
    usage = ip_usage[ip]
    if usage["date"] != datetime.now().date():
        ip_usage[ip] = {
            "videos": 0,
            "questions": 0,
            "date": datetime.now().date()
        }
    return ip_usage[ip]

def cleanup_old_sessions():
    if len(sessions) > 100:
        oldest_keys = list(sessions.keys())[:50]
        for key in oldest_keys:
            del sessions[key]

# --- Request Models ---
class LoadRequest(BaseModel):
    youtube_url: str

class AskRequest(BaseModel):
    question: str

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def root(
    request: Request,
    session_id: str = Cookie(default=None)
):
    response = templates.TemplateResponse("index.html", {"request": request})
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=3600,
            httponly=True
        )
    return response

@app.post("/load")
@limiter.limit("5/minute")
async def load_video(
    request: Request,
    body: LoadRequest,
    session_id: str = Cookie(default=None)
):
    if not session_id:
        session_id = str(uuid.uuid4())

    client_ip = request.client.host
    ip = get_ip_usage(client_ip)

    # IP daily video limit
    if ip["videos"] >= MAX_VIDEOS_PER_IP_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit of {MAX_VIDEOS_PER_IP_PER_DAY} videos reached. Come back tomorrow."
        )

    cleanup_old_sessions()
    session = get_session(session_id)

    # Session video limit
    if len(session["loaded_videos"]) >= MAX_VIDEOS_PER_SESSION:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_VIDEOS_PER_SESSION} videos per session. Clear session to load more."
        )

    try:
        if session["vector_store"] is None:
            vector_store, retriever, transcript, video_id = build_vectorstore(body.youtube_url)
        else:
            vector_store, retriever, transcript, video_id = add_video_to_store(
                session["vector_store"], body.youtube_url
            )

        # Duplicate check
        if video_id in session["loaded_videos"]:
            return JSONResponse({
                "message": "Video already loaded ✅",
                "video_id": video_id,
                "summary": None,
                "questions": [],
                "videos_loaded": len(session["loaded_videos"]),
                "videos_remaining": MAX_VIDEOS_PER_SESSION - len(session["loaded_videos"])
            })

        # Update session
        session["vector_store"] = vector_store
        session["retriever"] = retriever
        session["chain"] = build_chain(retriever)
        session["chat_history"] = []
        session["loaded_videos"].append(video_id)

        # Increment IP counter
        ip["videos"] += 1

        meta = generate_summary_and_questions(transcript)

        response = JSONResponse({
            "message": "Video loaded successfully ",
            "video_id": video_id,
            "summary": meta["summary"],
            "questions": meta["questions"],
            "videos_loaded": len(session["loaded_videos"]),
            "videos_remaining": MAX_VIDEOS_PER_SESSION - len(session["loaded_videos"])
        })
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=3600,
            httponly=True
        )
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
@limiter.limit("20/minute")
async def ask_question(
    request: Request,
    body: AskRequest,
    session_id: str = Cookie(default=None)
):
    if not session_id or session_id not in sessions:
        raise HTTPException(
            status_code=400,
            detail="No session found. Please load a video first."
        )

    client_ip = request.client.host
    ip = get_ip_usage(client_ip)

    # IP daily question limit
    if ip["questions"] >= MAX_QUESTIONS_PER_IP_PER_DAY:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit of {MAX_QUESTIONS_PER_IP_PER_DAY} questions reached. Come back tomorrow."
        )

    session = get_session(session_id)

    if session["chain"] is None:
        raise HTTPException(
            status_code=400,
            detail="No video loaded. Please load a video first."
        )

    # Session message limit
    messages_used = len(session["chat_history"]) // 2
    if messages_used >= MAX_MESSAGES_PER_SESSION:
        raise HTTPException(
            status_code=400,
            detail=f"Reached {MAX_MESSAGES_PER_SESSION} message limit. Clear session to continue."
        )

    try:
        result = session["chain"]({
            "question": body.question,
            "chat_history": session["chat_history"]
        })

        session["chat_history"].append({"role": "user", "content": body.question})
        session["chat_history"].append({"role": "assistant", "content": result["answer"]})

        if len(session["chat_history"]) > 40:
            session["chat_history"] = session["chat_history"][-40:]

        ip["questions"] += 1
        messages_used = len(session["chat_history"]) // 2

        return {
            "question": body.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "messages_used": messages_used,
            "messages_remaining": MAX_MESSAGES_PER_SESSION - messages_used
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_session(
    request: Request,
    session_id: str = Cookie(default=None)
):
    if session_id and session_id in sessions:
        del sessions[session_id]
    response = JSONResponse({"message": "Session cleared ✅"})
    response.delete_cookie("session_id")
    return response

@app.get("/status")
async def session_status(
    request: Request,
    session_id: str = Cookie(default=None)
):
    client_ip = request.client.host
    ip = get_ip_usage(client_ip)

    if not session_id or session_id not in sessions:
        return {
            "videos_loaded": 0,
            "videos_remaining": MAX_VIDEOS_PER_SESSION,
            "messages_used": 0,
            "messages_remaining": MAX_MESSAGES_PER_SESSION,
            "ip_videos_remaining": MAX_VIDEOS_PER_IP_PER_DAY - ip["videos"],
            "ip_questions_remaining": MAX_QUESTIONS_PER_IP_PER_DAY - ip["questions"]
        }

    session = sessions[session_id]
    messages_used = len(session["chat_history"]) // 2

    return {
        "videos_loaded": len(session["loaded_videos"]),
        "videos_remaining": MAX_VIDEOS_PER_SESSION - len(session["loaded_videos"]),
        "messages_used": messages_used,
        "messages_remaining": MAX_MESSAGES_PER_SESSION - messages_used,
        "ip_videos_remaining": MAX_VIDEOS_PER_IP_PER_DAY - ip["videos"],
        "ip_questions_remaining": MAX_QUESTIONS_PER_IP_PER_DAY - ip["questions"]
    }