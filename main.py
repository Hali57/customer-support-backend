from fastapi import FastAPI, HTTPException, Request
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://customer-support-bot-frontend.vercel.app"],  # Adjust if needed
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow these methods
    allow_headers=["Content-Type", "Authorization"],  # Allow common headers
)

# Load model and tokenizer
model_name = "HaliG/customer-support-chatbot"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Move model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.get("/")
def home():
    return {"message": "Customer Support Chatbot API is running!"}

@app.options("/predict/")
async def preflight():
    """Handles the OPTIONS request sent by browsers before making a POST request."""
    return {}

@app.post("/predict/")
async def predict(request: Request):
    try:
        data = await request.json()
        if "query" not in data:
            raise HTTPException(status_code=400, detail="Missing 'query' field in request body")

        input_text = data["query"]
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=100)

        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return {"response": response_text}

    except Exception as e:
        return {"error": str(e)}
