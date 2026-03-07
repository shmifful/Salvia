from pydantic import BaseModel

class TextRank(BaseModel):
    text: str
    n: int = 2

class JustText(BaseModel):
    text: str