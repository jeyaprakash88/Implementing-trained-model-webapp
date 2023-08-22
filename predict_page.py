import streamlit as st
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.exceptions import NotFittedError
import numpy as np
from explore_page import get_data_from_excel


# Function to load the model, label encoder, and preprocessor
def load_model():
    """
    Load the saved steps of a machine learning model from a pickled file.

    This function reads the contents of 'saved_steps.pkl', a pickled file, and returns
    the loaded data. The pickled file should contain all the necessary information
    related to the pre-trained model and its associated preprocessing steps.

    Returns:
        data (any): The loaded data from the pickled file. This could include the
        pre-trained model, feature transformations, label encodings, and any other
        necessary steps used during training.

    Raises:
        FileNotFoundError: If the 'saved_steps.pkl' file does not exist in the current
        working directory or the specified path.
        PickleError: If there is an issue while unpickling the file, such as data
        corruption or an unsupported pickle format.

    Example:
        # Assuming 'saved_steps.pkl' contains the saved model and preprocessing steps
        loaded_data = load_model()
        model = loaded_data['model']
        preprocessor = loaded_data['preprocessor']

        # Use the loaded model and preprocessor for predictions
        X_new = preprocessor.transform(new_data)
        predictions = model.predict(X_new)
    """
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor_model = data["model"]
le_preprocessor = data["preprocessor"]
le_encoder = data["label_encoder"]

# Function to preprocess the user input and make predictions


def show_predict_page():
    """
    Show the user interface for predicting repair demand based on user inputs.

    This function displays a Streamlit web page where the user can input various
    parameters related to the property and repair details. The user can select
    options from dropdown menus and input numeric values using number inputs.
    After clicking the "Predict Repair" button, the function will preprocess
    the user input, make predictions using the pre-trained machine learning model,
    and display the predicted repair demand category.

    Returns:
        None

    Example:
        # Assuming the show_predict_page() function is defined and imported.
        show_predict_page()
    """
    st.title("Predicting repair demand")    
    
    # Select box for sortrd-cde-1
    sortrd_cde_1 = st.selectbox("Sort Code", ['0A', '0B', '0C', '0D', '0E', '0F', '0G', '0L', '0M', '0O', '0P', '0R', '0S', '0T', '0W', '0X', '0Z', 'A', 'AS', 'B', 'BC', 'BR', 'C', 'CC', 'CO', 'D', 'DH', 'DM', 'DR', 'DT', 'E', 'E1', 'EH', 'EI', 'EW', 'EX', 'F', 'FI', 'G', 'GE', 'GF', 'GH', 'GS', 'H', 'HP', 'IN', 'LS', 'M', 'MA', 'MI', 'NS', 'OT', 'P', 'PC', 'PD', 'PE', 'PL', 'PO', 'PT', 'PW', 'R', 'S', 'SC', 'SH', 'SP', 'TI', 'TV', 'VC', 'W', 'WD', 'WT'])

    jobsts_cde = st.selectbox("Job Status Code", ["70", "90", "20", "69", "60", "40", "62", "30", "10", "27", "6B", "6G", "2G", "26", "35"])

    # Select box for locality name
    locality_name = st.selectbox("Locality Name", ['Gloucester', 'Gloucestershire', 'Stroud', 'Tewkesbury'])

    town_mapping = st.selectbox("Town", ["Gloucester", "Stonehouse", "Ashchurch", "Hucclecote", "Rodborough", "Cheltenham", "Thrupp"])

    # Select box for building type
    building_type = st.selectbox("Building Type", ['Block', 'Commercial', 'Flat', 'Garage', 'House', 'Public Bldg', 'Street', 'Virtual'])

    # Select box for pty_classification_subtype (tenu_cde)
    tenu_cde = st.selectbox("Tenure Code", ['AF', 'AN', 'AS', 'CM', 'CT', 'ES', 'ET', 'GT', 'LH', 'LI', 'MP', 'NS', 'OTHER', 'PF', 'PG', 'PP', 'PS', 'S', 'SO', 'ST', 'TM', 'VV'])

    # Select box for tenu_cde
    rntpaymd_cde = st.selectbox("Rent repayment Code", ['APA', 'BANK', 'CASH', 'DD15', 'DD4W', 'DDF', 'DDG', 'DDM1', 'DDM2', 'DDM3', 'DDM4', 'DDW', 'DDWK', 'FHB', 'MGCH', 'MONC', 'OTHER', 'PART', 'RCP1', 'RCP4', 'TPD'])

    # Select box for rntpaymd_cde
    ttncytyp_cde = st.selectbox("Tenancy Type Code", ['OTHER', 'TC', 'TF', 'VF', 'VL'])

    # Select box for repair type
    pty_classification_subtype = st.selectbox("Repair Type", ['CANCELLED', 'Inspection', 'Other', 'Repair'])

    # Select box for right-to-repair
    right_to_repair = st.slider("Right to Repair", 0, 1, 0)

    # Text boxes for integer inputs
    pr_seq_no = st.number_input("Property Sequence Number", value=0, step=1)
    void_num = st.number_input("Void Number", value=0, step=1)
    str_cde = st.number_input("Street Code", value=0, step=1)
    age = st.number_input("Property age", value=0, step=1)
    ownership_percent = st.number_input("Ownership Percentage", value=0, step=1)
    no_of_bedroom = st.number_input("Number of Bedrooms", value=0, step=1)
    days_to_complete = st.number_input("Days to Complete", value=0, step=1)

    # Convert the input values to integers
    pr_seq_no = int(pr_seq_no)
    void_num = int(void_num)
    str_cde = int(str_cde)
    age = int(age)
    ownership_percent = int(ownership_percent)
    no_of_bedroom = int(no_of_bedroom)
    days_to_complete = int(days_to_complete)

    # Create the DataFrame from the user inputs
    data = pd.DataFrame({
        'pr-seq-no': [pr_seq_no],
        'void-num': [void_num],
        'str-cde': [str_cde],
        'Age': [age],
        'ownership-%': [ownership_percent],
        'No_of_bedroom': [no_of_bedroom],
        'Days_to_complete': [days_to_complete],
        'sortrd-cde-1': [sortrd_cde_1],
        'jobsts-cde': [jobsts_cde],
        'loc-nam-2': [locality_name],
        'Building_type': [building_type],
        'tenu_cde': [tenu_cde],
        'rntpaymd_cde': [rntpaymd_cde],
        'ttncytyp_cde': [ttncytyp_cde],
        'pty_classification_subtype': [pty_classification_subtype],
        'town': [town_mapping]
    })
    # Show the "Predict" button

    ok = st.button("Predict Repair")
    if ok:
        X_preprocessed = le_preprocessor.transform(data)

        prediction = regressor_model.predict(X_preprocessed)
        prediction_label = le_encoder.inverse_transform(prediction)[0]
        st.header('Prediction Result')
        st.write(f'The predicted price category is : **{prediction_label}**')

    # Create a ticket form
    st.header('Create a Support Ticket')
    name = st.text_input("Customer Name")
    email = st.text_input("Email")
    issue_type = st.selectbox("Issue Type", ["Void Works", "Emergency Repair", "Other", "Routine Repair",
    "Inspection", "CANCELLED", "Planned Work", "Cyclical Works"])
    description = st.text_area("Issue Description")
    submit = st.button("Submit Ticket")

    if submit:
        # Store the ticket information or take further actions
        st.write("Ticket submitted successfully!")
        # ---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
# Run the web app
if __name__ == "__main__":
    show_predict_page()
