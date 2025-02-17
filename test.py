from transformers import pipeline
import json
import os
import spacy
import torch
import tensorflow as tf
import tensorflow_hub as hub
from textblob import TextBlob
from torch import nn
import cv2
import pytesseract
from PIL import Image

def load_tf_sentiment_model():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


class PostQualityModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=256):
        super(PostQualityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def load_model():
    """Loads the text generation model."""
    return pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

def analyze_sentiment(text, tf_model):
    """Uses TensorFlow's Universal Sentence Encoder for sentiment analysis."""
    embeddings = tf_model([text])
    sentiment_score = float(tf.reduce_mean(embeddings))
    return sentiment_score

def extract_text_from_image(image_path):
    """Extracts text from an image using Tesseract OCR."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

def validate_certificate(image_path, valid_certificates):
    """Validates if a certificate is recognized based on extracted text."""
    extracted_text = extract_text_from_image(image_path)
    for cert in valid_certificates:
        if cert.lower() in extracted_text.lower():
            return f"Valid certificate detected: {cert}"
    return "Certificate not recognized. Ensure clarity and proper alignment."


def analyze_profile_picture(image_path):
    """Analyzes a profile picture to check for face presence and clarity."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return "No face detected. Use a clear professional photo."
    return "Profile picture looks good! Ensure proper lighting and clarity."

def main():
    """Main function to handle user interaction."""
    print("LinkedIn Personal Assistant GPT")
    model = load_model()
    tf_model = load_tf_sentiment_model()
    pt_model = PostQualityModel()
    valid_certificates = ["Google Data Analytics", "AWS Certified Solutions Architect", "Microsoft Azure Fundamentals"]


while True:
    print(
        "\nOptions: 1. Generate Post  2. Optimize Profile  3. Validate Certificate  4. Analyze Profile Picture  5. Exit")
    choice = input("Choose an option: ")

    if choice == "1":
        prompt = input("Enter your post idea: ")
        post, sentiment, _ = generate_post(prompt, model, tf_model, pt_model)
        print("\nGenerated Post:")
        print(post)
        print(f"\nSentiment Score: {sentiment:.2f} (Higher is better)")
    elif choice == "2":
        summary = input("Enter your profile summary: ")
        print("\nProfile Optimization Tips:")
        print(optimize_profile(summary))
    elif choice == "3":
        image_path = input("Enter the path to your certificate image: ")
        print(validate_certificate(image_path, valid_certificates))
    elif choice == "4":
        image_path = input("Enter the path to your profile picture: ")
        print(analyze_profile_picture(image_path))
    elif choice == "5":
        break
    else:
        print("Invalid choice. Please select again.")s