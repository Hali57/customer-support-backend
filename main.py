from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

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
model = T5ForConditionalGeneration.from_pretrained(model_name).to("cpu")

@app.get("/")
def home():
    return {"message": "Customer Support Chatbot API is running on Railway!"}

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    input_text = data["query"]
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=100)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {"response": response}
