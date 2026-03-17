import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=os.environ["GROQ_API_KEY"]
)

qa_prompt = PromptTemplate(
    template="""You are a helpful assistant. The transcript may be in any language.
Always reply in the same language the user is asking in.
If the user asks in English, always reply in English regardless of transcript language.
Answer ONLY from the provided transcript context.
If the answer is not in the context, say "I don't know based on this video."

Previous conversation:
{chat_history}

Context from transcript:
{context}

Question: {question}

Answer:""",
    input_variables=["chat_history", "context", "question"]
)

summary_prompt = PromptTemplate(
    template="""You are a helpful assistant. Read this YouTube video transcript and provide:
1. A concise 2-3 sentence summary of the video.
2. Exactly 3 interesting questions a viewer might ask about this video.

Transcript (first 3000 chars):
{transcript}

Respond in this exact JSON format with no extra text:
{{
  "summary": "your summary here",
  "questions": ["question 1", "question 2", "question 3"]
}}""",
    input_variables=["transcript"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

def build_chain(retriever):
    def run_chain(inputs: dict):
        question = inputs["question"]
        history = inputs.get("chat_history", [])

        docs = retriever.invoke(question)
        context = format_docs(docs)
        chat_history_text = format_chat_history(history)

        final_prompt = qa_prompt.invoke({
            "chat_history": chat_history_text,
            "context": context,
            "question": question
        })

        try:
            answer = llm.invoke(final_prompt)
            answer_text = answer.content if hasattr(answer, "content") else str(answer)
        except Exception as e:
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "429" in error_str:
                if "tokens per day" in error_str:
                    answer_text = "⚠️ Daily AI limit reached. Please try again tomorrow."
                else:
                    answer_text = "⚠️ Too many requests. Please wait a moment and try again."
            else:
                raise

        sources = [
            {
                "content": doc.page_content[:300],
                "video_id": doc.metadata.get("video_id", "unknown")
            }
            for doc in docs
        ]

        return {"answer": answer_text, "sources": sources}

    return run_chain

def generate_summary_and_questions(transcript: str) -> dict:
    try:
        prompt = summary_prompt.invoke({"transcript": transcript[:3000]})
        response = llm.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {
            "summary": "Summary could not be generated.",
            "questions": [
                "What is the main topic of this video?",
                "What are the key takeaways?",
                "Can you summarize this video?"
            ]
        }