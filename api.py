from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from generate import generate_text, stoi
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="ShakespeareGPT API")

# Allow our portfolio website to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://cj-thomas.com",
        "https://www.cj-thomas.com",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"]
)


# Define what the website sends to our API
class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=256)
    temperature: float = Field(default=0.8, gt=0, le=2.0)
    new_chars: int = Field(default=250, ge=1, le=500)


# Check that the API is running
@app.get("/")
def home():
    return {"message": "ShakespeareGPT API is running!"}


# Generate dialogue using our trained Transformer
@app.post("/generate")
def generate(request: GenerateRequest):

    # Check for characters the model doesn't recognize
    unsupported = set(request.prompt) - set(stoi)

    if unsupported:
        raise HTTPException(
            status_code=400,
            detail="Prompt contains unsupported characters."
        )

    # Generate text using our Transformer
    result = generate_text(
        prompt=request.prompt,
        temperature=request.temperature,
        new_chars=request.new_chars
    )

    return {
        "generated_text": request.prompt + result
    }