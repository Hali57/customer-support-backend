from fastapi import FastAPI
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from transformers import AutoModel, AutoTokenizer



# Initialize FastAPI app
app = FastAPI()

# # List of origins that are allowed to make requests
# origins = [
#     "http://localhost:5173",
#     "http://127.0.0.1:5173",  # Added this line
# ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load model and tokenizer
model_name = "HaliG/customer-support-chatbot"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.get("/")
def home():
    return {"message": "Customer Support Chatbot API is running!"}

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    print("Received Data:", data)  # ✅ Debugging line

    if "query" not in data:
        return {"error": "Invalid JSON format, 'query' key missing"}, 400

    input_text = data["query"]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=100)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"response": response}

# Run the server with: uvicorn app:app --reload
