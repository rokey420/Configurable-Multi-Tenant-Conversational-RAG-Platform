from typing import List, Optional
from pydantic import BaseModel, Field

class Citation(BaseModel):
    source: str = Field(min_length=1)
    page: Optional[int] = Field(default=None, ge=0)

class LLMResponse(BaseModel):
    answer: str = Field(min_length=1)
    citations: List[Citation] = Field(default_factory=list)