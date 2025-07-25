import gradio as gr
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

loaded_model = pickle.load(open('salary_predictor.sav', 'rb'))
original_dataset = pd.read_csv('final salary.csv')
job_counts = original_dataset['Job Title'].value_counts()
unique_job_titles = original_dataset['Job Title'].unique().tolist()

def predict_salary(age, gender, education, experience, job_title):
    gender_le = LabelEncoder()
    gender_le.fit(['Male', 'Female'])
    gender_encoded = gender_le.transform([gender])[0]
    edu_le = LabelEncoder()
    edu_le.fit(["Bachelor's", "Master's", "PhD"])
    edu_encoded = edu_le.transform([education])[0]
    job_freq = job_counts.get(job_title, 0)
    X_input = [[float(age), float(gender_encoded), float(edu_encoded), float(experience), float(job_freq)]]
    predicted_salary = loaded_model.predict(X_input)[0]
    return f"Predicted salary per annum: ${predicted_salary:,.2f}"


with gr.Blocks() as demo:
    gr.Markdown("# Salary Predictor")
    age = gr.Number(label="Age")
    gender = gr.Radio(["Male", "Female"], label="Gender")
    education = gr.Dropdown(["Bachelor's", "Master's", "PhD"], label="Education Level")
    experience = gr.Number(label="Years of Experience")
    job = gr.Dropdown(unique_job_titles, label="Job Title")
    output = gr.Textbox(label="Predicted Salary")
    predict_button = gr.Button("Predict Salary")
    predict_button.click(
        predict_salary,
        inputs=[age, gender, education, experience, job],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True)
