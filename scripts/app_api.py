from fastapi import FastAPI
from pydantic import BaseModel
from scripts.answer_module import answer

# Create FastAPI application instance
app = FastAPI()

# Define request body model
class Query(BaseModel):
    # Query string field
    q: str

# Define chat endpoint route
@app.post("/chat")
def chat(q: Query):
    """
    Endpoint to handle chat requests
    
    Args:
        q (Query): Request body object containing query string
    
    Returns:
        dict: Dictionary containing answer and reference information
    """
    # Call answer function to process query and get response
    ans, refs = answer(q.q)
    # Return JSON response
    return {"answer": ans, "references": refs}