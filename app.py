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
    if not query.strip():
        return "Please enter a query or select a question."
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

# Predefined questions
questions = [
    "What are the symptoms of COVID-19?",
    "How can I boost my immune system?",
    "What should I do if I have a fever?",
    "Can you explain the side effects of vaccines?",
    "How do I manage stress and anxiety?",
    "What are the benefits of regular exercise?",
    "Which foods help improve digestion?"
]

# Create Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# **HealthCare Customer Support Chatbot**")
    gr.Markdown("Enter your query or select a question from the dropdown menu to get health-related information!")
    
    text_input = gr.Textbox(lines=2, placeholder="Enter your query here...", label="Your Question")
    dropdown_input = gr.Dropdown(choices=questions, label="Or select a question", interactive=True)
    submit_button = gr.Button("Submit")
    output_text = gr.Textbox(label="Chatbot Response", interactive=False)
    
    def update_textbox(selected_question):
        return selected_question
    
    dropdown_input.change(update_textbox, inputs=dropdown_input, outputs=text_input)
    submit_button.click(chatbot, inputs=text_input, outputs=output_text)

# Launch the interface with a public link
iface.launch(share=True)
