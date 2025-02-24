# SUMMATIVE ASSIGNMENT - BUILD DOMAIN-SPECIFIC A CHAT-BOT

## **Testing the Conversation**
![Image](https://github.com/user-attachments/assets/b0d61eb6-1cc7-47cc-a790-607f8b169092)

## **Chatbot Purpose and Domain Alignment**
This chatbot is a domain-specific conversational agent trained on **3,000** queries related to the `healthcare` and `finance` sectors.

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

# **Hyperparameter Tuning and Performance Optimization**
The chatbot model was fine-tuned using T5-small with a thorough exploration of hyperparameters to achieve optimal performance.

Multiple training configurations were tested to improve validation metrics `(BLEU, ROUGE)` beyond baseline performance.

## **The following hyperparameters were tuned:**

| Hyperparameter  | Baseline Value | Optimized Value | Impact |
|----------------|---------------|----------------|--------|
| Learning Rate  | 5e-5          | **3e-5**       | Improved stability, reduced overfitting |
| Batch Size     | 16            | **8**          | Lower memory usage, better generalization |
| Warmup Steps   | 0             | **500**        | Prevents sudden loss spikes |
| Weight Decay   | 0.01          | **0.001**      | Regularization improved generalization |
| Epochs        | 3             | **6**          | Improved performance without overfitting |

---

## **Performance Improvements Over Baseline**

The following metrics were used to compare model performance before and after hyperparameter tuning:

| Metric         | Baseline | Optimized Model | Improvement |
|---------------|---------|----------------|-------------|
| BLEU Score    | 0.82    | **0.91**       | +10.9%      |
| ROUGE-1 F1    | 0.87    | **0.95**       | +9.2%       |
| ROUGE-2 F1    | 0.85    | **0.95**       | +11.8%      |
| ROUGE-L F1    | 0.86    | **0.96**       | +11.6%      |

# **Usage Instructions**

## **1. Clone the Repository**
  `git clone https://github.com/Ochan-LOKIDORMOI/Ml_summative_assignment_chatbot.git`
  
  `cd Ml_summative_assignment_chatbot`

## **2. Install Dependencies**
  `pip install -r requirements.txt`

## **Run the Chatbot Application**
  `python app.py`

## **4. Interact with the Chatbot**

Once the application is running on gradio, enter queries in the terminal or web interface:

- **ealthcare Example:** "What are the side effects of the COVID-19 vaccine?"

- **Finance Example:** "How do I check my account balance?"

Author:

**Ochan LOKIDORMOI**

