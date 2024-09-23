import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

def get_clean_data():
    data_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed_lung_cancer.csv')
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found at {data_file}")
    data = pd.read_csv(data_file)
    if 'Unnamed: 0' in data.columns:
        data = data.drop('Unnamed: 0', axis=1)
    return data

def add_sidebar():
    st.sidebar.markdown("""
    <style>
    .sidebar-button {
        background-color: red;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.sidebar.button("About", key="about", help="Learn more", type="primary", use_container_width=True):
        st.sidebar.info("This is a Lung Cancer Risk Predictor app that uses Machine Learning to assess the risk of lung cancer based on various factors.")

    if st.sidebar.button("How to Use", key="how_to_use", help="Learn how", type="primary", use_container_width=True):
        st.sidebar.markdown("""
        ### How to Use
        
        This tool uses a Logistic Regression model trained on a dataset of lung cancer patients. Here's how it works:

        1. Input patient data using the sliders and input fields in the sidebar.
        2. The app scales the input data using StandardScaler.
        3. The scaled data is then fed into the trained model.
        4. The model predicts the probability of lung cancer.
        5. Results are displayed as a radar chart and probability scores.

        **Scientific Details:**
        - Model: Logistic Regression
        - Scaling: StandardScaler
        - Features: Age, Gender, Smoking habits, and various symptoms
        - Accuracy: 96.7%

        **Machine Learning Process:**
        1. Data Preparation: We used a dataset of lung cancer patients, removing any unnamed columns.
        2. Feature Selection: We selected relevant features including age, gender, smoking habits, and various symptoms.
        3. Data Splitting: The dataset was split into training (80%) and testing (20%) sets.
        4. Feature Scaling: We used StandardScaler to normalize the feature values.
        5. Model Training: A Logistic Regression model was trained on the scaled training data.
        6. Model Evaluation: The model's performance was evaluated using accuracy score and classification report on the test set.
        7. Model Persistence: The trained model, scaler, and feature names were saved using pickle for use in this application.

        Remember, this tool is for educational purposes and should not replace professional medical advice.
        """)

    if st.sidebar.button("Contact", key="Connect?", help="Contact information", type="primary", use_container_width=True):
        st.sidebar.info("For any queries or feedback, please contact: ggengineerco@gmail.com.")
        st.sidebar.markdown("<span style='color: red;'>From Engineer</span>", unsafe_allow_html=True)

    st.sidebar.header("Lung Cancer Risk Factors")
    
    data = get_clean_data()
    if data is None:
        st.stop()
    
    input_dict = {}

    # Add Gender selection
    input_dict["GENDER"] = st.sidebar.radio(
        "Gender",
        options=[1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female",
        help="Select the patient's gender"
    )

    # Add Age input
    input_dict["AGE"] = st.sidebar.number_input(
        "Age",
        min_value=int(data["AGE"].min()),
        max_value=int(data["AGE"].max()),
        value=int(data["AGE"].mean()),
        help="Enter the patient's age"
    )
    
    slider_labels = [
        ("Smoking", "SMOKING", "Indicate the patient's smoking habits"),
        ("Yellow Fingers", "YELLOW_FINGERS", "Presence of yellow fingers, often associated with smoking"),
        ("Anxiety", "ANXIETY", "Level of anxiety experienced by the patient"),
        ("Peer Pressure", "PEER_PRESSURE", "Influence of peer pressure on the patient"),
        ("Chronic Disease", "CHRONIC_DISEASE", "Presence of any chronic diseases"),
        ("Fatigue", "FATIGUE ", "Level of fatigue experienced by the patient"),
        ("Allergy", "ALLERGY ", "Presence of allergies"),
        ("Wheezing", "WHEEZING", "Frequency of wheezing"),
        ("Alcohol Consuming", "ALCOHOL_CONSUMPTION", "Level of alcohol consumption"),
        ("Coughing", "COUGHING", "Frequency of coughing"),
        ("Shortness of Breath", "SHORTNESS_OF_BREATH", "Frequency of shortness of breath"),
        ("Swallowing Difficulty", "SWALLOWING_DIFFICULTY", "Difficulty in swallowing"),
        ("Chest Pain", "CHEST_PAIN", "Frequency of chest pain"),
    ]

    for label, key, help_text in slider_labels:
        if key not in data.columns:
            st.warning(f"Column '{key}' not found in the dataset. Skipping this feature.")
            continue
        
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
            help=help_text
        )
    
    return input_dict

def get_radar_chart(input_data):
    categories = list(input_data.keys())
    values = list(input_data.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Patient Data'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 2]
            )),
        showlegend=False
    )
    
    return fig

def add_predictions(input_data):
    model_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'lung_cancer_model.pkl')
    scaler_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl')
    feature_names_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'feature_names.pkl')

    if not os.path.exists(model_file) or not os.path.exists(scaler_file) or not os.path.exists(feature_names_file):
        st.error("Model, scaler, or feature names file not found. Please check if the files exist.")
        return

    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)

    with open(feature_names_file, 'rb') as f:
        feature_names = pickle.load(f)
  
    features = pd.DataFrame([input_data])
    features = features.reindex(columns=feature_names, fill_value=0)
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
  
    st.subheader("Lung Cancer Risk Prediction")
    st.markdown("The prediction is:")
  
    if prediction[0] == 0:
        st.markdown("<span class='diagnosis benign'>This person does not have lung cancer</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='diagnosis malicious'>This person has lung cancer</span>", unsafe_allow_html=True)
    
    proba = model.predict_proba(features_scaled)[0]
    st.markdown(f"Probability of not having lung cancer: **{proba[0]:.2f}**")
    st.markdown(f"Probability of having lung cancer: <span style='color: red;'>**{proba[1]:.2f}**</span>", unsafe_allow_html=True)
  
    st.markdown("This app can assist medical professionals in assessing lung cancer risk, but should not be used as a substitute for a professional diagnosis.")

    return prediction, proba

def visualize_data(data):
    st.subheader("Data Visualization")

    # Distribution of features
    st.write("Distribution of Features")
    for column in data.columns:
        if column != 'GENDER' and column != 'LUNG_CANCER':
            fig = px.pie(data, names=column, title=f"Distribution of {column}")
            st.plotly_chart(fig)
            st.write(f"Let's look at the {column} pie chart. Each slice represents a different value for {column}. "
                     f"The size of each slice shows how common that value is. Bigger slices mean more people have that value, "
                     f"while smaller slices are less common. It's like dividing a pizza where each topping represents a different value, "
                     f"and the amount of each topping shows how often it occurs.")

    # Correlation representation
    st.write("Feature Relationships")
    corr = data.corr().abs()
    corr_sum = corr.sum().sort_values(ascending=False)
    top_corr = corr_sum.head(5)
    fig = px.pie(values=top_corr.values, names=top_corr.index, title='Top 5 Most Related Features')
    st.plotly_chart(fig)
    st.write("This pie chart shows which health factors are most connected to others. The bigger the slice, "
             "the more that factor tends to change along with other factors. It's like seeing which ingredients "
             "in a recipe tend to be used together. This helps doctors understand which health signs often come as a package deal.")

    # Feature importance plot
    st.write("Feature Importance")
    model_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'lung_cancer_model.pkl')
    feature_names_file = os.path.join(os.path.dirname(__file__), '..', 'model', 'feature_names.pkl')
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(feature_names_file, 'rb') as f:
        feature_names = pickle.load(f)
    
    importance = pd.DataFrame({'feature': feature_names, 'importance': abs(model.coef_[0])})
    importance = importance.sort_values('importance', ascending=False)
    fig = px.pie(importance, values='importance', names='feature', title='Feature Importance')
    st.plotly_chart(fig)
    st.write("Think of this pie chart as a 'health clue pie'. Each slice represents a different health factor. "
             "The bigger the slice, the more important that factor is in understanding lung cancer risk. "
             "It's like a detective's notebook, where the size of each clue shows how crucial it is to solving the case. "
             "This helps doctors focus on the most telling signs when assessing someone's health.")

def collect_historical_data():
    st.subheader("Historical Data Collection")

    # User input fields
    medical_history = st.text_area("Medical History", placeholder="Enter patient's medical history...")
    drug_usage = st.text_area("Drug Usage", placeholder="Enter patient's drug usage history...")
    family_history = st.text_area("Family History", placeholder="Enter patient's family medical history...")
    occupational_exposure = st.text_area("Occupational Exposure", placeholder="Enter any occupational exposure to harmful substances...")
    lifestyle_factors = st.text_area("Lifestyle Factors", placeholder="Enter relevant lifestyle factors (e.g., diet, exercise)...")
    previous_treatments = st.text_area("Previous Treatments", placeholder="Enter any previous cancer treatments...")
    environmental_factors = st.text_area("Environmental Factors", placeholder="Enter relevant environmental factors...")

    # File upload
    uploaded_files = st.file_uploader("Upload relevant documents (PDF, DOCX, Images)", accept_multiple_files=True, type=['pdf', 'docx', 'png', 'jpg', 'jpeg'])

    if st.button("Generate Action Steps"):
        # Process the inputs and generate action steps
        action_steps = generate_action_steps(medical_history, drug_usage, family_history, occupational_exposure, lifestyle_factors, previous_treatments, environmental_factors, uploaded_files)
        
        st.subheader("Recommended Action Steps")
        for step in action_steps:
            st.write(f"- {step}")

def generate_action_steps(medical_history, drug_usage, family_history, occupational_exposure, lifestyle_factors, previous_treatments, environmental_factors, uploaded_files):
    # This function would contain logic to generate action steps based on the inputs
    # For now, we'll return some placeholder steps
    steps = [
        "Schedule a comprehensive medical examination",
        "Conduct specific tests based on the patient's medical history",
        "Review and adjust current medications if necessary",
        "Recommend lifestyle changes based on identified risk factors",
        "Suggest genetic counseling if family history indicates high risk",
        "Provide resources for quitting smoking or reducing alcohol consumption if applicable",
        "Schedule follow-up appointments to monitor progress"
    ]
    return steps

def generate_pdf_report(input_data, prediction, proba):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Lung Cancer Risk Prediction Report")

    # Patient Data
    c.setFont("Helvetica", 12)
    y = height - 80
    for key, value in input_data.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20

    # Prediction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y - 20, "Prediction:")
    c.setFont("Helvetica", 12)
    prediction_text = "Does not have lung cancer" if prediction[0] == 0 else "Has lung cancer"
    c.drawString(50, y - 40, prediction_text)

    # Probabilities
    c.drawString(50, y - 60, f"Probability of not having lung cancer: {proba[0]:.2f}")
    c.drawString(50, y - 80, f"Probability of having lung cancer: {proba[1]:.2f}")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

def main():
    st.set_page_config(
        page_title="Lung Cancer Risk Predictor",
        page_icon=":lungs:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    css_file = os.path.join(os.path.dirname(__file__), '..', 'assets', 'style.css')
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    else:
        st.warning("Style file not found. The app will continue without custom styling.")
    
    input_data = add_sidebar()
    
    with st.container():
        st.markdown("<h1 style='color: red;'>Lung Cancer Risk Predictor</h1>", unsafe_allow_html=True)
        st.write("This app predicts the risk of lung cancer based on various factors. Please input the patient's data using the sidebar for an accurate prediction.")
        st.write("The prediction is based on machine learning algorithms trained on historical data.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Data Visualization", "Diagnose Yourself", "Additional Information"])
    
    with tab1:
        col1, col2 = st.columns([4,1])
        
        with col1:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart)
        with col2:
            prediction, proba = add_predictions(input_data)

        if st.button("Generate PDF Report"):
            pdf = generate_pdf_report(input_data, prediction, proba)
            st.download_button(
                label="Download PDF Report",
                data=pdf,
                file_name="lung_cancer_risk_report.pdf",
                mime="application/pdf"
            )

    with tab2:
        data = get_clean_data()
        visualize_data(data)

    with tab3:
        st.subheader("Diagnose Yourself")
        
        # Additional user inputs
        st.write("Please provide the following information:")
        smoking_status = st.selectbox("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"])
        if smoking_status != "Non-smoker":
            smoking_time = st.slider("How long have you been smoking (in years)?", 0, 50, 0)
        
        chemical_exposure = st.selectbox("Exposure to Chemicals", ["None", "Occasional", "Frequent"])
        if chemical_exposure != "None":
            chemical_exposure_time = st.slider("How long have you been exposed to chemicals (in years)?", 0, 50, 0)
        
        family_history = st.selectbox("Family History of Lung Cancer", ["No", "Yes"])
        if family_history == "Yes":
            family_history_time = st.slider("How long ago was the family member diagnosed (in years)?", 0, 50, 0)
        
        shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Occasional", "Frequent"])
        if shortness_of_breath != "No":
            shortness_of_breath_time = st.slider("How long have you experienced shortness of breath (in years)?", 0, 50, 0)
        
        chest_pain = st.selectbox("Chest Pain", ["No", "Occasional", "Frequent"])
        if chest_pain != "No":
            chest_pain_time = st.slider("How long have you experienced chest pain (in years)?", 0, 50, 0)
        
        # Update input_data with new inputs
        input_data.update({
            "SMOKING": 0 if smoking_status == "Non-smoker" else 1 if smoking_status == "Former smoker" else 2,
            "CHEMICAL_EXPOSURE": 0 if chemical_exposure == "None" else 1 if chemical_exposure == "Occasional" else 2,
            "FAMILY_HISTORY": 1 if family_history == "Yes" else 0,
            "SHORTNESS_OF_BREATH": 0 if shortness_of_breath == "No" else 1 if shortness_of_breath == "Occasional" else 2,
            "CHEST_PAIN": 0 if chest_pain == "No" else 1 if chest_pain == "Occasional" else 2
        })
        
        if st.button("Submit Data for Analysis"):
            # Load the trained model and scaler
            model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'lung_cancer_model.pkl')
            scaler_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scaler.pkl')
            feature_names_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'feature_names.pkl')
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
            
            # Prepare user input data
            user_input = np.array([input_data[feature] for feature in feature_names]).reshape(1, -1)
            scaled_input = scaler.transform(user_input)
            
            # Make prediction
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]
            
            st.write("## Analysis Results")
            st.write(f"Lung Cancer Risk: {'High' if prediction == 1 else 'Low'}")
            st.write(f"Probability: {probability:.2f}")
            
            st.write("## Recommended Steps")
            if prediction == 1:
                st.write("Based on the analysis, you have a high risk of lung cancer. You are advised to:")
                st.write("1. Consult with a healthcare professional immediately for a thorough evaluation.")
                st.write("2. Schedule a low-dose CT scan for early detection.")
                st.write("3. If you smoke, consider joining a smoking cessation program.")
                st.write("4. Improve your diet and exercise routine to boost overall health.")
                st.write("5. Minimize exposure to chemicals and other environmental risk factors.")
            else:
                st.write("Based on the analysis, you have a low risk of lung cancer. Here are some recommended steps to maintain your health:")
                st.write("1. Continue regular check-ups with your healthcare provider.")
                st.write("2. Maintain a healthy lifestyle with a balanced diet and regular exercise.")
                st.write("3. Avoid exposure to known lung cancer risk factors like smoking and air pollution.")
                st.write("4. Stay vigilant about any changes in your respiratory health.")
            
            st.write("## Factors Influencing the Analysis")
            feature_importance = model.coef_[0]
            sorted_idx = np.argsort(feature_importance)
            top_features = [feature_names[i] for i in sorted_idx[-5:]]
            st.write("Given below are the factors that have the most influence on your health:")
            for feature in top_features:
                st.write(f"- {feature}: {input_data[feature]}")
            
            st.write("## In-depth Analysis of your health")
            st.write("Smoking Status:", smoking_status)
            if smoking_status == "Current smoker":
                st.write("Smoking is a major risk factor for lung cancer. Consider joining a smoking cessation program.")
            elif smoking_status == "Former smoker":
                st.write("While quitting smoking reduces risk, former smokers should remain vigilant about their lung health.")
            
            st.write("Chemical Exposure:", chemical_exposure)
            if chemical_exposure != "None":
                st.write("Exposure to certain chemicals can increase lung cancer risk. Consider ways to minimize exposure.")
            
            st.write("Family History:", family_history)
            if family_history == "Yes":
                st.write("A family history of lung cancer may indicate a genetic predisposition. Regular screenings are crucial.")
            
            st.write("Shortness of Breath:", shortness_of_breath)
            st.write("Chest Pain:", chest_pain)
            if shortness_of_breath != "No" or chest_pain != "No":
                st.write("These symptoms could be indicative of various lung conditions. A medical evaluation is recommended.")
            
            st.write("Remember, this analysis is based on a simplified model and should not replace professional medical advice. Always consult with a healthcare provider for accurate diagnosis and treatment.")

    with tab4:
        st.subheader("General Information about Lung Cancer")
        st.write("""
        Lung cancer is a type of cancer that begins in the lungs. It is one of the most common cancers worldwide.

        **Risk Factors:**
        - Smoking
        - Exposure to secondhand smoke
        - Exposure to radon gas
        - Family history
        - Exposure to asbestos and other carcinogens

        **Prevention Methods:**
        - Don't smoke
        - Avoid secondhand smoke
        - Test your home for radon
        - Avoid carcinogens at work
        - Eat a healthy diet
        - Exercise regularly

        For more information, please visit:
        - [American Cancer Society](https://www.cancer.org/cancer/lung-cancer.html)
        - [National Cancer Institute](https://www.cancer.gov/types/lung)
        - [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/cancer)
        """)

if __name__ == '__main__':
    main()