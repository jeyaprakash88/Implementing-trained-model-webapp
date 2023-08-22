from flask import Flask, render_template, request
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import mpld3
import pickle
from predict_page import show_predict_page
from explore_page import explore_page  # Import the explore_page function

import streamlit as st

page = st.sidebar.selectbox("Explore or predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    explore_page()  # Call the explore_page function from explore_page.py
