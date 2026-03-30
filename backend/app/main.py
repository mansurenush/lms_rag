
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
import re
from threading import Lock
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from configs.config_loader import load_config


APP_TITLE = "Local Assist Chat API"
API_PREFIX = "/api/v1"

_CFG = {}
try:
    _CFG = load_config()
except Exception:
    
    _CFG = {}


def _cfg_get(path: str, default: str) -> str:
    cur: Any = _CFG
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return str(cur) if cur is not None else default


def _cfg_get_int(path: str, default: int) -> int:
    cur: Any = _CFG
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    try:
        return int(cur)
    except Exception:
        return default



CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", _cfg_get("paths.chroma_persist_dir", "data/chroma_moodle_docs_501"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", _cfg_get("paths.chroma_collection", "moodle_docs_501"))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", _cfg_get("embeddings.model_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", str(_cfg_get_int("retrieval.top_k", 5))))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", str(_cfg_get_int("chat.max_history_messages", 10))))
RETRIEVAL_HISTORY_USER_TURNS = int(
    os.getenv("RETRIEVAL_HISTORY_USER_TURNS", str(_cfg_get_int("chat.retrieval_history_user_turns", 2)))
)


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", _cfg_get("llm.ollama_base_url", "http://127.0.0.1:11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", _cfg_get("llm.ollama_model", "qwen3:1.7b"))
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", str(_cfg_get_int("llm.timeout_seconds", 120))))
LLM_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", str(_cfg_get_int("llm.num_gpu", 0))))
LLM_RANDOM_SEED = int(os.getenv("OLLAMA_RANDOM_SEED", str(_cfg_get_int("llm.random_seed", 42))))

SYSTEM_PROMPT_TEMPLATE = os.getenv(
    "SYSTEM_PROMPT_TEMPLATE",
    _cfg_get(
        "llm.system_prompt_template",
        "You are a Moodle support assistant.\nUse ONLY the provided CONTEXT as factual ground.\nIf context is insufficient, explicitly say it is insufficient and suggest what to search next in Moodle docs.\nAlways answer in {target_language}.\nCONTEXT:\n{context}",
    ),
)


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: str


class ChatMessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class SourceItem(BaseModel):
    source_url: str | None = None
    title: str | None = None
    score: float | None = None


class ChatMessageResponse(BaseModel):
    message_id: str
    role: str = "assistant"
    content: str
    sources: list[SourceItem]


@dataclass
class SessionState:
    created_at: str
    messages: list[dict[str, str]] = field(default_factory=list)


class ChatService:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = Lock()
        self._vectorstore = None

    def create_session(self) -> CreateSessionResponse:
        sid = str(uuid4())
        created_at = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            self._sessions[sid] = SessionState(created_at=created_at, messages=[])
        return CreateSessionResponse(session_id=sid, created_at=created_at)

    def _get_session(self, session_id: str) -> SessionState:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail={"error": "session_not_found"})
        return session

    def _get_vectorstore(self):
        if self._vectorstore is not None:
            return self._vectorstore

        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self._vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        return self._vectorstore

    @staticmethod
    def _build_context(hits: list[tuple[Any, float]]) -> tuple[str, list[SourceItem]]:
        blocks: list[str] = []
        sources: list[SourceItem] = []
        for idx, (doc, score) in enumerate(hits, start=1):
            source_url = doc.metadata.get("source_url")
            title = doc.metadata.get("title")
            text = doc.page_content.strip()
            blocks.append(f"[{idx}] title={title}\nsource={source_url}\n{text}")
            sources.append(SourceItem(source_url=source_url, title=title, score=float(score)))
        return "\n\n---\n\n".join(blocks), sources

    def _retrieve(self, user_query: str) -> tuple[str, list[SourceItem]]:
        try:
            vs = self._get_vectorstore()
            hits = vs.similarity_search_with_score(user_query, k=RAG_TOP_K)
        except Exception as e:  
            raise HTTPException(
                status_code=503,
                detail={"error": "vector_store_unavailable", "details": str(e)},
            ) from e
        return self._build_context(hits)

    def _build_retrieval_query(self, session: SessionState, current_user_text: str) -> str:
        """
        History-aware retrieval query:
        combine latest N user turns with the current question,
        so follow-up questions keep topic continuity.
        """
        if RETRIEVAL_HISTORY_USER_TURNS <= 0:
            return current_user_text

        prev_user_turns = [m["content"] for m in session.messages if m.get("role") == "user"]
        if prev_user_turns:
            prev_user_turns = prev_user_turns[-RETRIEVAL_HISTORY_USER_TURNS :]
            return "\n".join(prev_user_turns + [current_user_text])
        return current_user_text

    @staticmethod
    def _detect_language(user_text: str) -> str:
        
        return "Russian" if re.search(r"[А-Яа-яЁё]", user_text) else "English"

    @staticmethod
    def _system_prompt(context: str, target_language: str) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(context=context, target_language=target_language)

    @staticmethod
    def _trim_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if len(messages) <= MAX_HISTORY_MESSAGES:
            return messages
        return messages[-MAX_HISTORY_MESSAGES:]

    def _call_ollama(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": messages,
            "options": {"num_gpu": LLM_NUM_GPU, "seed": LLM_RANDOM_SEED},
        }
        try:
            with httpx.Client(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
                resp = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:  
            raise HTTPException(
                status_code=503,
                detail={"error": "llm_unavailable", "details": str(e)},
            ) from e

        content = ((data.get("message") or {}).get("content") or "").strip()
        if not content:
            raise HTTPException(
                status_code=502,
                detail={"error": "llm_empty_response", "details": "No content in Ollama response"},
            )
        return content

    def send_message(self, session_id: str, content: str) -> ChatMessageResponse:
        session = self._get_session(session_id)
        retrieval_query = self._build_retrieval_query(session, content)
        context, sources = self._retrieve(retrieval_query)
        target_language = self._detect_language(content)

        history = self._trim_history(session.messages)
        llm_messages = [{"role": "system", "content": self._system_prompt(context, target_language)}] + history + [
            {"role": "user", "content": content}
        ]
        assistant_content = self._call_ollama(llm_messages)

        message_id = str(uuid4())
        with self._lock:
            session.messages.append({"role": "user", "content": content})
            session.messages.append({"role": "assistant", "content": assistant_content})
            session.messages = self._trim_history(session.messages)

        return ChatMessageResponse(
            message_id=message_id,
            role="assistant",
            content=assistant_content,
            sources=sources,
        )


app = FastAPI(title=APP_TITLE)
chat_service = ChatService()


@app.get(f"{API_PREFIX}/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(f"{API_PREFIX}/chat/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    return chat_service.create_session()


@app.post(f"{API_PREFIX}/chat/sessions/{{session_id}}/messages", response_model=ChatMessageResponse)
def send_message(session_id: str, payload: ChatMessageRequest) -> ChatMessageResponse:
    return chat_service.send_message(session_id=session_id, content=payload.content)

