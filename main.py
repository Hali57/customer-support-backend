from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "HaliG/customer-support-chatbot"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Quantization for faster inference
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
model.to("cpu")

logger.info("Model loaded and quantized successfully.")

@app.get("/")
def home():
    return {"message": "Customer Support Chatbot API is running on Railway!"}

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    input_text = data.get("query", "")
    logger.info(f"Received request: POST\nInput Text: {input_text}")

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    try:
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=50)  # Reduce max_length to speed up
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"Generated Response: {response}")
        return {"response": response}
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return {"error": "Error processing the request. Please try again later."}
