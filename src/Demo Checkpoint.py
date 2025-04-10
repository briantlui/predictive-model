
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(layout="wide")

# Hardcoded file paths
MODEL_PATH = "../../models/trained/random_forest_sm3.pkl"  # Model Path
DATA_PATH = "../../data/processed/X_test.joblib"  # Data path

# Hardcoded columns to display
columns_to_display = ['adr', 'los', 'lead_time', 'arrival_month','hotel_City Hotel', 'total_of_special_requests']  # column names

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

# Sidebar: Checkboxes for displaying hotel columns
st.sidebar.subheader("Select Hotel Type to Display:")
show_city_hotel = st.sidebar.checkbox("City Hotel", value=True)  # Default to showing City Hotel
show_resort_hotel = st.sidebar.checkbox("Resort Hotel", value=True)  # Default to showing Resort Hotel

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
    
    # Ensure the necessary columns exist
    if "hotel_City Hotel" not in filtered_data.columns or "hotel_Resort Hotel" not in filtered_data.columns:
        st.error("Missing required columns in the dataset: 'hotel_City Hotel' or 'hotel_Resort Hotel'")
    else:
        # Filter dataset based on hotel selection
        if show_city_hotel and not show_resort_hotel:
            filtered_data = filtered_data[filtered_data["hotel_City Hotel"] == 1]  # Only City Hotel rows
        elif show_resort_hotel and not show_city_hotel:
            filtered_data = filtered_data[filtered_data["hotel_City Hotel"] == 0]  # Only Resort Hotel rows
        elif show_city_hotel and show_resort_hotel:
            pass  # No filtering, display all rows

    # Select and display the hardcoded columns
    display_data = filtered_data[columns_to_display]

    # Predict cancellation probabilities
    raw_predictions = model.predict_proba(filtered_data)[:, 1]  # predicts probabilities
    predictions = (raw_predictions >= threshold).astype(int)

    # Add cancellation risk (%) and predictions columns to the table
    display_data["Cancellation Probability (%)"] = (raw_predictions * 100).round(0).astype(int)  # Convert to percentages, round to whole numbers, and format as integers
    display_data["adr"] = display_data["adr"].round(0).astype(int)  # Round 'adr' to nearest whole number and format as integers
    
    # Convert the predictions array to a Pandas Series before replacing values
    display_data["Predicted Cancel"] = ["Yes" if pred == 1 else "No" for pred in predictions]  # Replace 1/0 with Yes/No


    # Count total predictions equal to 1
    total_predictions = predictions.sum()

    # Display total count at the top
    st.header(f"Total Cancellations Predicted: {total_predictions}  \n Threshold Selected: {threshold} ")

    # Rename columns for cleaner display
    display_data = display_data.rename(columns={
    'arrival_month': 'Arrival Month',
    'hotel_City Hotel': 'City Hotel',
    'los': 'Stay Nights',
    'adr': 'Avg Daily Rate',
    'lead_time': 'Lead Time',
    'Cancellation Risk (%)': 'Cancellation Probability (%)',
    'Predicted Cancellation': 'Predicted Cancellation',
    'total_of_special_requests': 'Total Special Requests'
})
    # Reorder columns
    desired_order = ['Predicted Cancel', 'Cancellation Probability (%)'] + [col for col in display_data.columns if col not in ['Predicted Cancel', 'Cancellation Probability (%)']]
    display_data = display_data[desired_order]
    
    # Limit to 100 rows
    display_data = display_data.head(100)

    # Show table
    st.subheader(f"Reservations for {selected_month_name} ")
    st.dataframe(display_data, width=1000)
except FileNotFoundError:
    st.error("Please ensure the model and dataset files are located at the specified paths.")
except KeyError:
    st.error(f"One or more of the hardcoded columns {columns_to_display} are missing from the dataset.")




# import streamlit as st
# import pandas as pd
# import joblib

# # Hardcoded file paths
# MODEL_PATH = "../../models/trained/random_forest_sm3.pkl"  # Model Path
# DATA_PATH = "../../data/processed/X_test.joblib"  # Data path

# # Hardcoded columns to display
# columns_to_display = ['arrival_month','hotel_City Hotel', 'adr', 'los', 'lead_time']  # column names

# # Streamlit app
# st.title("Hotel Reservation Cancellation Predictor")

# # Sidebar: Threshold selection
# threshold = st.sidebar.slider(
#     "Select Cancellation Threshold:",
#     min_value=0.0,
#     max_value=1.0,
#     value=0.5,
#     step=0.05
#     )

# # Month mapping: Number to Name
# month_mapping = {
#     0: "All Months", 1: "January", 2: "February", 3: "March", 4: "April",
#     5: "May", 6: "June", 7: "July", 8: "August",
#     9: "September", 10: "October", 11: "November", 12: "December"
# }

# # Sidebar: arrival_month selection using month names
# selected_month_name = st.sidebar.selectbox(
#     "Select Arrival Month:",
#     list(month_mapping.values())  # Dropdown shows month names
# )

# # Get the corresponding month number
# selected_month = {v: k for k, v in month_mapping.items()}[selected_month_name]

# try:
#     # Load the model
#     model = joblib.load(MODEL_PATH)

#     # Load the dataset
#     X_test = joblib.load(DATA_PATH)
   
#     # Filter dataset based on selection
#     if selected_month == 0:  # "All Months" selected
#         filtered_data = X_test  # Use the entire dataset
#     else:
#         filtered_data = X_test[X_test['arrival_month'] == selected_month]  # Filter for specific month
    
#     # Select and display the hardcoded columns
#     display_data = filtered_data[columns_to_display]

#     # Predict cancellation probabilities
#     raw_predictions = model.predict_proba(filtered_data)[:, 1]  # predicts probabilities
#     predictions = (raw_predictions >= threshold).astype(int)

#     # Add cancellation risk (%) and predictions columns to the table
#     display_data["Cancellation Risk (%)"] = (raw_predictions * 100).round(2)  # Convert to percentages and round
#     display_data["Prediction"] = predictions

#     # Count total predictions equal to 1
#     total_predictions = predictions.sum()

#     # Display total count at the top
#     st.header(f"Total Cancellations Predicted for {selected_month_name}: {total_predictions}")

#     # Rename columns for cleaner display
#     display_data = display_data.rename(columns={
#     'arrival_month': 'Arrival Month',
#     'hotel_City Hotel': 'City Hotel',
#     'los': 'Length of Stay',
#     'adr': 'Average Daily Rate',
#     'lead_time': 'Lead Time',
#     'Cancellation Risk (%)': 'Cancellation Risk (%)',
#     'Predicted Cancellation': 'Predicted Cancellation'
# })

#     # Limit to 500 rows
#     display_data = display_data.head(500)

#     # Show table
#     st.subheader(f"Predicted Results for {selected_month_name} (Threshold: {threshold}, First 500 Rows)")
#     st.table(display_data)
# except FileNotFoundError:
#     st.error("Please ensure the model and dataset files are located at the specified paths.")
# except KeyError:
#     st.error(f"One or more of the hardcoded columns {columns_to_display} are missing from the dataset.")