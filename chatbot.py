import random
import json
import nltk
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load intents (You can expand this list)
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "Good morning", "Good evening"],
            "responses": ["Hello!", "Hi there!", "Hey! How can I help you today?"]
        },
        {
            "tag": "ai",
            "patterns": [
                "What is AI?",
                "Tell me about Artificial Intelligence",
                "What can AI do?",
                "How does AI work?",
                "Examples of AI?"
            ],
            "responses": [
                "AI stands for Artificial Intelligence – it's the simulation of human intelligence by machines.",
                "AI enables computers to learn and make decisions like humans.",
                "AI is used in chatbots, recommendation systems, self-driving cars, and more."
            ]
        },
        {
            "tag": "python",
            "patterns": [
                "What is Python?",
                "Is Python easy to learn?",
                "Tell me about Python programming",
                "What can I build with Python?",
                "Why is Python so popular?"
            ],
            "responses": [
                "Python is a popular programming language known for its simplicity and versatility.",
                "Python is great for web development, data science, AI, automation, and more.",
                "Yes! Python is beginner-friendly and widely used across industries."
            ]
        },
        {
            "tag": "webdev",
            "patterns": [
                "What is a web developer?",
                "How to become a web developer?",
                "Skills needed for web development?",
                "Frontend vs backend?",
                "What does a web developer do?"
            ],
            "responses": [
                "A web developer builds and maintains websites and web apps.",
                "They work with technologies like HTML, CSS, JavaScript, and frameworks.",
                "Web developers can specialize in frontend, backend, or full-stack development."
            ]
        },
        {
            "tag": "vscode",
            "patterns": [
                "What is VS Code?",
                "How to use Visual Studio Code?",
                "Best extensions for VS Code?",
                "Why use VS Code?",
                "VS Code vs other editors?"
            ],
            "responses": [
                "VS Code is a lightweight and powerful code editor from Microsoft.",
                "It supports many languages and has great extensions for Python, web development, and more.",
                "VS Code is highly customizable, fast, and beginner-friendly."
            ]
        },
        {
            "tag": "html",
            "patterns": [
                "What is HTML?",
                "Why is HTML important?",
                "Explain HTML structure",
                "HTML in web development?",
                "What does HTML stand for?"
            ],
            "responses": [
                "HTML stands for HyperText Markup Language.",
                "It's used to structure the content on web pages.",
                "Every website you visit is built using HTML."
            ]
        },
        {
            "tag": "css",
            "patterns": [
                "What is CSS?",
                "Why use CSS?",
                "How does CSS work?",
                "CSS in frontend development?",
                "What does CSS stand for?"
            ],
            "responses": [
                "CSS stands for Cascading Style Sheets.",
                "It is used to style and layout web pages.",
                "CSS controls colors, fonts, spacing, and design."
            ]
        },
        {
            "tag": "javascript",
            "patterns": [
                "What is JavaScript?",
                "Why is JavaScript important?",
                "Is JavaScript frontend or backend?",
                "What can I build with JavaScript?",
                "Difference between JavaScript and Java?"
            ],
            "responses": [
                "JavaScript is a programming language used for web development.",
                "It makes web pages interactive and dynamic.",
                "JavaScript can be used on both frontend and backend."
            ]
        },
        {
            "tag": "codtech",
            "patterns": [
                "What is CodTech IT Solutions?",
                "Tell me about CodTech company",
                "What services does CodTech provide?",
                "Is CodTech a good company?",
                "Details of CodTech IT Solutions"
            ],
            "responses": [
                "CodTech IT Solutions is a technology company specializing in software development and IT services.",
                "They offer services like web development, mobile apps, cloud solutions, and digital transformation.",
                "CodTech is known for delivering high-quality tech solutions for modern businesses."
            ]
        },
        {
            "tag": "thanks",
            "patterns": ["Thanks", "Thank you", "Thanks a lot", "That's helpful", "Thank u"],
            "responses": ["You're welcome!", "Glad to help!", "Happy to assist!"]
        }
    ]
}

# Preprocess and build dataset
corpus = []
tags = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(pattern)
        tags.append(intent["tag"])

# Create and train model pipeline
model = make_pipeline(TfidfVectorizer(), LogisticRegression())
model.fit(corpus, tags)

# Response generator
def get_response(user_input):
    tag = model.predict([user_input])[0]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# Chat interface
def chat():
    print("🤖 ChatBot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            print("🤖 ChatBot: Goodbye!")
            break
        response = get_response(user_input)
        print(f"🤖 ChatBot: {response}")

# Run chatbot
if __name__ == "__main__":
    chat()
