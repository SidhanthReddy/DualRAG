from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime
import uuid

from state_rag_enums import ArtifactType, ArtifactSource


class Artifact(BaseModel):
    # Identity
    artifact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ArtifactType
    name: str
    file_path: str

    # Payload
    content: str
    language: str  # tsx, ts, js, css, json

    # State
    version: int = 1
    is_active: bool = True
    source: ArtifactSource

    # Semantics
    dependencies: List[str] = []  # artifact_ids

    # Metadata
    framework: Optional[str] = "react"
    styling: Optional[str] = "tailwind"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # ------------------
    # Validators
    # ------------------

    @validator("content")
    def content_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Artifact content cannot be empty")
        return v

    @validator("file_path")
    def file_path_must_look_valid(cls, v):
        if "/" not in v and "\\" not in v:
            raise ValueError("file_path must include directory structure")
        return v

    @validator("language")
    def language_must_be_known(cls, v):
        allowed = {"tsx", "ts", "js", "css", "json"}
        if v not in allowed:
            raise ValueError(f"Unsupported language: {v}")
        return v
