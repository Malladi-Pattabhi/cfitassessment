import streamlit as st
import pandas as pd
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

# Load GPT-2 model and tokenizer for local generative model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# GPT-2 model prompt generator
def generate_gpt2_assessment(job_description, resume_text, behavioral_answers):
    # Formulate the prompt
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
    # Encode prompt and generate response
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        inputs,
        max_length=500,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )

    # Decode and return the generated text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

# Run assessments based on user input
if st.button("Analyze Culture Fit"):
    if job_description and resume_text and all(behavioral_answers):
        # Calculate culture fit using BERT model
        score, reasoning = calculate_culture_fit_score(job_description, resume_text, behavioral_answers)
        st.subheader("BERT Model Culture Fit Assessment")
        st.write(f"Culture Fit Score: {round(score * 100, 2)}")
        st.write(f"Reasoning: {reasoning}")

        # Calculate culture fit using GPT-2 model
        gpt2_response = generate_gpt2_assessment(job_description, resume_text, behavioral_answers)
        st.subheader("GPT-2 Model Culture Fit Assessment")
        st.write(gpt2_response)
    else:
        st.error("Please fill in all fields to proceed.")
