import gradio as gr
import numpy as np
import pandas as pd
import pickle
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

model = pickle.load(open('salary_predictor.sav', 'rb'))
data = pd.read_csv('final salary.csv')
job_counts = data['Job Title'].value_counts()
job_titles = data['Job Title'].unique().tolist()

def predict_salary(age, gender, education, experience, job_title):
    try:
        gender_map = {'Male': 1, 'Female': 0}
        education_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
        gender_encoded = gender_map.get(gender, 0)
        education_encoded = education_map.get(education, 0)
        job_freq = job_counts.get(job_title, 0)
        input_features = [[
            float(age),
            float(gender_encoded),
            float(education_encoded),
            float(experience),
            float(job_freq)
        ]]
        prediction = model.predict(input_features)[0]
        return f"${prediction:,.2f} per annum"
    except Exception as e:
        return f"Error: {str(e)}"

client = OpenAI()

def suggest_jobs_stream(age, gender, education, experience, job_title):
    try:
        prompt = (
            f"Respond in clear, short bullet points with emojis. Avoid markdown like **bold**. Stream output like a brochure."
            f"Suggest 5 relevant job openings in India or America for a {age}-year-old {gender} "
            f"with {education} and {experience} years of experience as a {job_title}. Also make sure to give the salary per annum for the job\n\n"
            f"Then, also suggest certifications or technical skills to acquire for better opportunities.\n\n"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You're a career advisor who gives warm, clear, and practical suggestions. No markdown formatting. You also welcome the user by saying \" Welcome to Career Suggestions Project by Harissh Krishna , here are some job openigns that i curated based on your career details  \""},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )


        full_response = ""
        for chunk in response:
         
            if chunk.choices[0].delta.content:
               
                full_response += chunk.choices[0].delta.content
                
                yield full_response

    except Exception as e:
        yield f"Error fetching job suggestions: {str(e)}"


with gr.Blocks(title="Career Compensation & Opportunity Estimator") as app:
    gr.Markdown("##  Career Compensation & Opportunity Estimator")
    gr.Markdown("Enter your profile details below to estimate your salary and get personalized job and skill suggestions powered by Open AI GPT-4o-mini.")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", value=25)
            gender = gr.Radio(choices=["Male", "Female"], label="Gender")
            education = gr.Dropdown(["Bachelor's", "Master's", "PhD"], label="Education")

        with gr.Column():
            experience = gr.Number(label="Years of Experience", value=2)
            job_title = gr.Dropdown(choices=job_titles, label="Job Title")

    with gr.Row():
        predict_button = gr.Button("Predict Salary")
        salary_output = gr.Textbox(label="Predicted Salary")

    with gr.Row():
        suggest_button = gr.Button("Suggest Jobs and Skills")
        job_output = gr.Textbox(label="Job Suggestions", lines=20, interactive=False, show_copy_button=True)

    predict_button.click(predict_salary, inputs=[age, gender, education, experience, job_title], outputs=salary_output)
    suggest_button.click(fn=suggest_jobs_stream, inputs=[age, gender, education, experience, job_title], outputs=job_output)

if __name__ == "__main__":
    app.launch()
