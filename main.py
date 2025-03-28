from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 Load the model and tokenizer during app startup
logging.info("Loading the model and tokenizer...")  # Log for tracking model loading
model_name = "HaliG/customer-support-chatbot"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to("cpu")
logging.info("Model and tokenizer loaded successfully!")  # Confirm successful load

@app.get("/")
def home():
    return {"message": "Customer Support Chatbot API is running on Railway!"}

@app.post("/predict/")
async def predict(request: Request):
    logging.info(f"Received request: {request.method}")
    
    if request.method != "POST":
        return {"error": "Only POST requests are allowed."}

    try:
        data = await request.json()
        input_text = data.get("query", "")
        logging.info(f"Input Text: {input_text}")

        # Prepare input for the model
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cpu")

        # 🔥 Generate response from the model
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=100)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        logging.info(f"Generated Response: {response}")
        return {"response": response}
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {"error": "An error occurred during prediction."}
