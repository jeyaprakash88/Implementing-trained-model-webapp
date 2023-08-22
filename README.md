
# App1: Property Repair Prediction Web App

## Overview

This web application is designed to predict the price category for repairs based on various property attributes. Additionally, it provides an interface for users to submit support tickets.

The application is built using Flask and Streamlit and relies on a trained XGBoost model for predictions.

## Dependencies

- Flask
- Streamlit
- Pandas
- XGBoost
- scikit-learn
- matplotlib
- seaborn
- joblib
- pickle
- mpld3

## Setup

1. Clone the repository:
```
git clone <repository_url>
```

2. Navigate to the project directory:
```
cd <project_directory>
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```
(Note: Ensure you have a `requirements.txt` file that lists all the dependencies.)

4. Run the Streamlit app:
```
streamlit run app1.py
```

## Usage

### Predictions

Users can input various attributes related to a property, such as:
- Age
- Number of bedrooms
- Locality name
- Building type
... and many more.

Based on the input, the system predicts the price category for repairs.

### Support Tickets

Users can also submit support tickets by providing their name, email, issue type, and a description of the issue.

## Notes

Ensure you have the `saved_steps.pkl` file in the project directory, which contains the trained model and preprocessing steps.

## Contributions

Feel free to fork the repository, create a feature branch, and open a Pull Request for any enhancements or fixes.

## License

Please add a suitable license based on your preferences or requirements.
