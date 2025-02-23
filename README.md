# SUMMATIVE ASSIGNMENT - BUILD DOMAIN-SPECIFIC A CHAT-BOT

## **Testing the Conversation**
![Image](https://github.com/user-attachments/assets/b0d61eb6-1cc7-47cc-a790-607f8b169092)

## **Chatbot Purpose and Domain Alignment**
This chatbot is a domain-specific conversational agent trained on **3,000** queries related to the healthcare and finance sectors.

It aims to provide automated responses to common user inquiries, improving accessibility and efficiency in these industries.

The dataset is structured into four key columns:
- **Query:** User input representing real-world questions.
- **Response:** Predefined chatbot answers.
- **Intent:** Categorization of query purpose.
- **Domain:** Context classification (e.g., healthcare, finance).


## **Repository Structure**

The repository is organized as follows:

```
Ml_summative_assignment_chatbot/
│── chatbot_model/        # Pre-trained or fine-tuned chatbot model files
│── data/                 # Dataset used for training the chatbot
│── notebook/             # Jupyter notebooks for experimentation and analysis
│── README.md             # Project documentation
│── app.py                # Main application script to run the chatbot
```

# **Preprocessing Steps**
The dataset undergoes rigorous preprocessing to ensure data quality and model efficiency:

## **1. Data Cleaning**
- Removal of unwanted characters (e.g., line breaks, HTML/XML tags).
- Conversion to lowercase for normalization.
- Removal of extra spaces to standardize input.

## **2. Handling Missing Values**
- The dataset was checked for missing values, ensuring completeness

## **3. Data Splitting**
- The dataset was split into 80% training and 20% validation sets

## **Tokenization and Normalization**
The chatbot leverages T5 tokenizer for processing textual data:

- Uses WordPiece tokenization to efficiently handle subwords.
- Applies padding and truncation for uniform sequence lengths.
- Converts text into numerical representations for model training.


