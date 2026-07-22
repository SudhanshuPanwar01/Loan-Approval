# app.py

import os
import joblib
import gradio as gr


# ==========================================================
# Load the trained model
# ==========================================================
try:
    deployed_rf = joblib.load("loan_prediction_model.pkl")
except Exception as e:
    print(f"Warning: Model not found or error loading. {e}")
    deployed_rf = None



# ==========================================================
# Prediction Function
# ==========================================================
def predict_loan_status(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
):

    # Input validation
    values = [
        no_of_dependents,
        education,
        self_employed,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]


    # Empty field check
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please fill all input fields."


    # Type conversion
    try:
        no_of_dependents = int(no_of_dependents)
        education = int(education)
        self_employed = int(self_employed)

        income_annum = float(income_annum)
        loan_amount = float(loan_amount)

        loan_term = int(loan_term)
        cibil_score = int(cibil_score)

        residential_assets_value = float(residential_assets_value)
        commercial_assets_value = float(commercial_assets_value)
        luxury_assets_value = float(luxury_assets_value)
        bank_asset_value = float(bank_asset_value)

    except (ValueError, TypeError):
        return "❌ Please enter valid numeric values."


    # Negative value validation
    numeric_values = [
        no_of_dependents,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value
    ]

    if any(v < 0 for v in numeric_values):
        return "❌ Negative values are not allowed."


    # CIBIL validation
    if not 300 <= cibil_score <= 900:
        return "❌ CIBIL score must be between 300 and 900."


    # Dependents validation
    if no_of_dependents > 20:
        return "❌ Number of dependents cannot exceed 20."


    # Model availability check
    if deployed_rf is None:
        return "❌ Model loading failed. Check your .pkl file."


    # Model prediction
    try:

        input_data = [[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]]


        prediction = deployed_rf.predict(input_data)

        confidence = None

        if hasattr(deployed_rf, "predict_proba"):
            probability = deployed_rf.predict_proba(input_data)
            confidence = max(probability[0]) * 100


        if prediction[0] == 1:

            result = """
🟢 LOAN APPROVED

The applicant meets the required criteria.
"""

        else:

            result = """
🔴 LOAN REJECTED

The applicant does not meet the approval criteria.
"""


        if confidence:
            result += f"\n\n📊 Confidence Score: {confidence:.2f}%"


        return result


    except Exception as e:
        return f"❌ Prediction failed.\n\nError: {str(e)}"
# ==========================================================
# Application Description
# ==========================================================

DESCRIPTION = """
# 🏦 AI Loan Approval Prediction System

Welcome to the AI-powered Loan Approval Prediction System.

This application uses a trained **Random Forest Machine Learning Model**
to analyze applicant details and predict whether a loan application is:

🟢 Approved

or

🔴 Rejected


### Features:

✔ Personal Information Analysis  
✔ Financial Profile Evaluation  
✔ Asset Value Assessment  
✔ CIBIL Score Checking  
✔ Machine Learning Based Prediction  


Enter the applicant details and click **Submit** to generate the prediction.
"""


# ==========================================================
# Developer Information
# ==========================================================

developer_info = """

## 👨‍💻 About Developer

**Created by:** Sudhanshu Panwar


🔗 **GitHub:**  
https://github.com/SudhanshuPanwar01


📷 **Instagram:**  
https://www.instagram.com/_rockstar._.__


---

## 🛠️ Technology Stack


🐍 Python

🤖 Scikit-Learn

🌲 Random Forest Classifier

📊 Pandas

🎨 Gradio

📦 Joblib


---

⭐ AI Loan Approval Prediction System
"""


# ==========================================================
# Custom CSS Styling
# ==========================================================

css = """

body {

    background:
    linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a,
        #0891b2
    );

}


.gradio-container {

    max-width:1200px !important;

    margin:auto;

    padding:25px;

    border-radius:25px;

    background:
    rgba(255,255,255,0.08);

    backdrop-filter:
    blur(15px);

    border:
    1px solid rgba(255,255,255,0.2);

    box-shadow:
    0px 20px 50px rgba(0,0,0,0.35);

}



h1 {

    text-align:center;

    color:#38bdf8 !important;

    font-size:42px !important;

    font-weight:800 !important;

}



h2,h3,h4 {

    color:white !important;

}



p {

    color:#e2e8f0 !important;

    font-size:16px;

}



label {

    color:white !important;

    font-weight:bold !important;

}



input,
textarea,
select {

    border-radius:12px !important;

}



button {

    background:

    linear-gradient(
        90deg,
        #2563eb,
        #06b6d4
    ) !important;


    color:white !important;

    border:none !important;

    border-radius:15px !important;

    font-size:18px !important;

    font-weight:bold !important;

    transition:0.3s ease;

}

button:hover {
    transform:
    translateY(-3px);
    box-shadow:
    0px 10px 25px rgba(6,182,212,0.5);
}

textarea {
    font-size:17px !important;
    background:white !important;
}

footer {
    display:none !important;
}

"""


# ==========================================================
# Gradio Theme
# ==========================================================

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate"

)
# ==========================================================
# Gradio Interface
# ==========================================================

interface = gr.Interface(
    fn=predict_loan_status,
    inputs=[

        gr.Number(
            label="👥 Number of Dependents"
        ),
        gr.Dropdown(
            choices=[
                ("Graduate", 1),
                ("Not Graduate", 0)
            ],
            label="🎓 Education Status"
        ),
        gr.Dropdown(
            choices=[
                ("Yes", 1),
                ("No", 0)
            ],
            label="💼 Self Employed?"
        ),
        gr.Number(
            label="💰 Annual Income"
        ),
        gr.Number(
            label="🏦 Loan Amount Requested"
        ),
        gr.Number(
            label="📅 Loan Term"
        ),
        gr.Number(
            label="📊 CIBIL Score (300-900)"
        ),
        gr.Number(
            label="🏠 Residential Assets Value"
        ),
        gr.Number(
            label="🏢 Commercial Assets Value"
        ),
        gr.Number(
            label="💎 Luxury Assets Value"
        ),
        gr.Number(
            label="🏛️ Bank Asset Value"
        )

    ],



    outputs=gr.Textbox(
        label="📋 Loan Assessment Result",
        lines=8

    ),

    title="🏦 AI Loan Approval Prediction System",
    description=DESCRIPTION,
    article=developer_info,
    theme=theme,
    css=css

)
# ==========================================================
# Launch Application
# ==========================================================

if __name__ == "__main__":

    interface.launch(
        server_name="0.0.0.0"
        server_port=int(
            os.environ.get(
                "PORT",
                7860
            )
        )

    )
