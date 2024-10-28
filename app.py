import streamlit as st
import pandas as pd
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer, GPT2Model
import google.generativeai as genai

# Set up Google API Key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gen-lang-client-0711936357-ec9fd589702f.json"
api_key = "AIzaSyCbRx_gwIgU_XGBTI1IQyV3X6-F-QyWId8"  # Replace with your actual API key
genai.configure(api_key=api_key)

# Streamlit UI styling and layout
st.markdown("""
    <style>
    .stApp {background-color: #f9f9f9;}
    .main-title {color: #333; font-size: 30px; text-align: center;}
    .section-header {font-size: 22px; margin-top: 15px; color: #333;}
    .result-box {background-color: #eef2f7; padding: 15px; border-radius: 10px; margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Culture Fit Assessment Tool</h1>', unsafe_allow_html=True)

# Input fields for user data
st.markdown('<h2 class="section-header">Enter Job Description and Resume</h2>', unsafe_allow_html=True)
job_description = st.text_area("Job Description", height=200)
resume_text = st.text_area("Resume", height=200)

st.markdown('<h2 class="section-header">Behavioral Question Responses</h2>', unsafe_allow_html=True)
behavioral_questions = [
    "Describe a time you worked in a team.",
    "How do you handle conflict?",
    "What motivates you to work hard?",
    "Describe a challenging project you worked on.",
    "How do you prioritize tasks when under pressure?"
]
behavioral_answers = [st.text_input(question) for question in behavioral_questions]

# Load Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to calculate culture fit using BERT model
def calculate_culture_fit_score(job_desc, resume_text, behavioral_answers):
    job_embedding = embedding_model.encode([job_desc])
    resume_embedding = embedding_model.encode([resume_text])
    behavioral_embeddings = [embedding_model.encode(answer) for answer in behavioral_answers]

    # Calculate similarity scores
    similarity_job_resume = cosine_similarity(job_embedding, resume_embedding)[0][0]
    similarity_behavioral = np.mean([cosine_similarity(job_embedding, [be]).flatten()[0] for be in behavioral_embeddings])

    culture_fit_score = (0.7 * similarity_job_resume) + (0.3 * similarity_behavioral)
    reasoning = f"Resume aligns with job description by {similarity_job_resume:.2f}. Behavioral responses add fit with score {similarity_behavioral:.2f}."
    return culture_fit_score, reasoning

# Configure Gemini model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

# Gemini model prompt generator
def generate_gemini_assessment(job_description, resume_text, behavioral_answers):
    prompt = f"""
    Job Description: {job_description}

    Resume: {resume_text}

    Behavioral Questions and Answers:
    1. {behavioral_answers[0]}
    2. {behavioral_answers[1]}
    3. {behavioral_answers[2]}
    4. {behavioral_answers[3]}
    5. {behavioral_answers[4]}

    Please assess the culture fit based on alignment with company culture pillars and provide a score and reasoning.
    """
    try:
        # Call to generate content directly with all necessary configuration included
        response = genai.generate_content(
            model="gemini-1.5-flash",
            prompt=prompt,
            **generation_config
        )
        return response.text if response else "No response generated."
    except Exception as e:
        return f"Error during API call: {e}"

# Run assessments based on user input
if st.button("Analyze Culture Fit"):
    if job_description and resume_text and all(behavioral_answers):
        # Calculate culture fit using BERT model
        score, reasoning = calculate_culture_fit_score(job_description, resume_text, behavioral_answers)
        st.subheader("BERT Model Culture Fit Assessment")
        st.write(f"Culture Fit Score: {round(score * 100, 2)}")
        st.write(f"Reasoning: {reasoning}")

        # Calculate culture fit using Gemini model
        gemini_response = generate_gemini_assessment(job_description, resume_text, behavioral_answers)
        st.subheader("Gemini Model Culture Fit Assessment")
        st.write(gemini_response)
    else:
        st.error("Please fill in all fields to proceed.")
