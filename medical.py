import os
import threading
import torch
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel

load_dotenv("my_api.py")
hf_token = os.getenv("HF_TOKEN")
login(hf_token, add_to_git_credential=False)

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# VRAM cleanup
torch.cuda.empty_cache()

st.set_page_config(page_title="Mistral Chatbot", layout="centered")
st.title("💬 Mistral Chatbot (LoRA, 4-bit)")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)

# 4-bit Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load base model in 4-bit
@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Load your LoRA adapter (e.g., medical tuned)
    model = PeftModel.from_pretrained(base_model, r"/mnt/c/Users/thehr/Downloads/mistral-lora-finetuned")
    model.eval()
    return model

model = load_model()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
if prompt := st.chat_input("Say something..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Tokenize and truncate for long context
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    st.chat_message("assistant").markdown(decoded)
    st.session_state.messages.append({"role": "assistant", "content": decoded})

    # Clean up VRAM
    del inputs, output
    torch.cuda.empty_cache()
