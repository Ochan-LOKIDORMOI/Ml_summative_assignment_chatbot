import gradio as gr
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

# Load saved model and tokenizer (Ensure the path is correct)
model_path = "chatbot_model"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def clean_text(text):
    """Function to clean the input text."""
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
    text = text.strip().lower()  # Strip and convert to lower case
    return text

def chatbot(query):
    """Function to generate the chatbot response."""
    query = clean_text(query)
    input_ids = tokenizer(query, return_tensors="pt", max_length=250, truncation=True)

    # inputs are on the correct device CPU
    inputs = {key: value.to(device) for key, value in input_ids.items()}

    # Generate the model's response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=250,
        num_beams=5,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="text",
    title="HealthCare Customer Support Chatbot",
    description="Ask me anything related to health care!"
)

# Launch the interface with a public link
iface.launch(share=True)