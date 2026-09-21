"""HTTP API: python api.py запускает сервер на порту 8100."""

from dataclasses import asdict
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from agent.types import AgentError, AgentResult, LLMError

load_dotenv(Path(__file__).resolve().parent / ".env")
app = FastAPI(title="AI-секретарь госоргана")


class Appeal(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Текст обращения не может быть пустым")
        return value


def get_runner() -> Callable[[str], AgentResult]:
    from agent.pipeline import run
    return run


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL", "")}


@app.post("/appeal", response_model=AgentResult)
def appeal(body: Appeal, runner: Callable = Depends(get_runner)):
    try:
        return asdict(runner(body.text))
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (AgentError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100)
