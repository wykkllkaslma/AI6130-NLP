from fastapi import FastAPI
from pydantic import BaseModel
from scripts.answer_module import answer

app = FastAPI()

class Query(BaseModel):
    q: str

@app.post("/chat")
def chat(q: Query):
    ans, refs = answer(q.q)
    return {"answer": ans, "references": refs}