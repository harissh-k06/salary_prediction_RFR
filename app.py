import gradio as gr
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your trained model
loaded_model = pickle.load(open('salary_predictor.sav', 'rb'))

# Load the original dataset to get job title frequencies
original_dataset = pd.read_csv('final salary.csv')
job_counts = original_dataset['Job Title'].value_counts()

def predict_salary(age, gender, education, experience, job_title):
    """
    Function to predict salary based on input parameters
    """
    try:
        # Encode gender
        gender_le = LabelEncoder()
        gender_le.fit(['Male', 'Female'])
        gender_encoded = gender_le.transform([gender])[0]
        
        # Encode education
        edu_le = LabelEncoder()
        edu_le.fit(["Bachelor's", "Master's", "PhD"])
        education_encoded = edu_le.transform([education])[0]
        
        # Get job title frequency
        job_freq = job_counts.get(job_title, 0)
        
        # Prepare input data
        X_input = [[
            float(age),
            float(gender_encoded),
            float(education_encoded),
            float(experience),
            float(job_freq)
        ]]
        
        # Make prediction
        predicted_salary = loaded_model.predict(X_input)[0]
        
        return f"₹{predicted_salary:,.2f}"
        
    except Exception as e:
        return f"Error: {str(e)}"

# Get unique job titles from the dataset for dropdown
unique_job_titles = original_dataset['Job Title'].unique().tolist()

# Create Gradio interface
with gr.Blocks(title="Salary Predictor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💰 Salary Prediction Tool")
    gr.Markdown("Enter your details below to get a predicted salary estimate!")
    
    with gr.Row():
        with gr.Column():
            age_input = gr.Number(
                label="Age",
                minimum=18,
                maximum=100,
                value=25,
                info="Enter your age (18-100)"
            )
            
            gender_input = gr.Radio(
                choices=["Male", "Female"],
                label="Gender",
                value="Male"
            )
            
            education_input = gr.Dropdown(
                choices=["Bachelor's", "Master's", "PhD"],
                label="Education Level",
                value="Bachelor's"
            )
            
        with gr.Column():
            experience_input = gr.Number(
                label="Years of Experience",
                minimum=0,
                maximum=50,
                value=2,
                info="Enter years of work experience"
            )
            
            job_title_input = gr.Dropdown(
                choices=unique_job_titles,
                label="Job Title",
                value=unique_job_titles[0] if unique_job_titles else "Data Analyst",
                info="Select your job position"
            )
    
    with gr.Row():
        predict_btn = gr.Button("🔮 Predict Salary", variant="primary", size="lg")
    
    with gr.Row():
        output = gr.Textbox(
            label="Predicted Salary",
            placeholder="Click 'Predict Salary' to see the result",
            lines=2
        )
    
    # Connect the button to the prediction function
    predict_btn.click(
        fn=predict_salary,
        inputs=[age_input, gender_input, education_input, experience_input, job_title_input],
        outputs=[output]
    )
    
    # Add some example inputs
    gr.Examples(
        examples=[
            [30, "Female", "Master's", 4, "Data Analyst"],
            [35, "Male", "Bachelor's", 8, "Software Engineer"],
            [28, "Female", "PhD", 3, "Research Scientist"],
        ],
        inputs=[age_input, gender_input, education_input, experience_input, job_title_input],
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        share=True,  # Creates a public link you can share
        debug=True   # Shows errors in the interface
    )
