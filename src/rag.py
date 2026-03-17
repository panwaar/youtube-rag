import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def extract_video_id(youtube_input: str) -> str:
    if "v=" in youtube_input:
        return youtube_input.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in youtube_input:
        return youtube_input.split("youtu.be/")[-1].split("?")[0]
    return youtube_input.strip()

def fetch_transcript(video_id: str) -> str:
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(
                ['en', 'en-GB', 'en-US', 'hi', 'hi-IN']
            )
        except NoTranscriptFound:
            try:
                transcript = transcript_list.find_generated_transcript(
                    ['en', 'en-GB', 'en-US', 'hi', 'hi-IN']
                )
            except NoTranscriptFound:
                transcript = next(iter(transcript_list))

        fetched = transcript.fetch()
        return " ".join(chunk.text for chunk in fetched)

    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except StopIteration:
        raise ValueError("No transcripts available for this video.")
    except Exception as e:
        raise ValueError(f"Could not fetch transcript: {str(e)}")

def chunk_transcript(transcript: str, video_id: str):
    texts = splitter.split_text(transcript)
    return splitter.create_documents(
        [transcript],
        metadatas=[{"video_id": video_id}] * len(texts)
    )

def build_vectorstore(youtube_input: str):
    video_id = extract_video_id(youtube_input)
    transcript = fetch_transcript(video_id)
    chunks = chunk_transcript(transcript, video_id)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return vector_store, retriever, transcript, video_id

def add_video_to_store(vector_store, youtube_input: str):
    video_id = extract_video_id(youtube_input)
    transcript = fetch_transcript(video_id)
    chunks = chunk_transcript(transcript, video_id)
    vector_store.add_documents(chunks)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return vector_store, retriever, transcript, video_id