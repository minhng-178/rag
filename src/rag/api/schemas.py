"""Pydantic request/response models for the REST API."""

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    page: int


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source] = []


class IngestResponse(BaseModel):
    message: str
    loaded_files: list[str] = []
