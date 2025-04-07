import streamlit as st
import pandas as pd
import joblib

# Hardcoded file paths
MODEL_PATH = "../../models/trained/random_forest_sm3.pkl"  # Model Path
DATA_PATH = "../../data/processed/X_test.joblib"  # Data path

# Hardcoded columns to display
columns_to_display = ['arrival_month', 'total_of_special_requests', 'los', 'lead_time']  # column names

# Streamlit app
st.title("Hotel Reservation Cancellation Predictor")

# Sidebar: Threshold selection
threshold = st.sidebar.slider(
    "Select Cancellation Threshold:",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
    )

# Month mapping: Number to Name
month_mapping = {
    0: "All Months", 1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# Sidebar: arrival_month selection using month names
selected_month_name = st.sidebar.selectbox(
    "Select Arrival Month:",
    list(month_mapping.values())  # Dropdown shows month names
)
# Get the corresponding month number
selected_month = {v: k for k, v in month_mapping.items()}[selected_month_name]

# Sidebar: arrival_dow selection (dynamic filter)
# arrival_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']  # Replace with relevant values from your dataset
# selected_day = st.sidebar.selectbox("Select Arrival Day of Week (arrival_dow):", arrival_days)

try:
    # Load the model
    model = joblib.load(MODEL_PATH)

    # Load the dataset
    X_test = joblib.load(DATA_PATH)
   
    # Filter dataset based on selection
    if selected_month == 0:  # "All Months" selected
        filtered_data = X_test  # Use the entire dataset
    else:
        filtered_data = X_test[X_test['arrival_month'] == selected_month]  # Filter for specific month
    
    # Select and display the hardcoded columns
    display_data = filtered_data[columns_to_display]

    # Predict cancellation probabilities
    raw_predictions = model.predict_proba(filtered_data)[:, 1]  # predicts probabilities
    predictions = (raw_predictions >= threshold).astype(int)

    # Add cancellation risk (%) and predictions columns to the table
    display_data["Cancellation Risk (%)"] = (raw_predictions * 100).round(2)  # Convert to percentages and round
    display_data["Prediction"] = predictions

    # Count total predictions equal to 1
    total_predictions = predictions.sum()

    # Display total count at the top
    st.header(f"Total Cancellations Predicted for {selected_month_name}: {total_predictions}")

    # Limit to 500 rows
    display_data = display_data.head(500)

    # Show table
    st.subheader(f"Predicted Results for {selected_month_name} (Threshold: {threshold}, First 500 Rows)")
    st.table(display_data)
except FileNotFoundError:
    st.error("Please ensure the model and dataset files are located at the specified paths.")
except KeyError:
    st.error(f"One or more of the hardcoded columns {columns_to_display} are missing from the dataset.")
