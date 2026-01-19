
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# Set page config
st.set_page_config(page_title="Flight Fare Predictor", page_icon="✈️")

st.title("✈️ Airline Price Prediction")
st.markdown("---")

# Data Generation Function (Cached)
@st.cache_data
def load_data(n=2000):
    airlines = ['IndiGo', 'Air India', 'SpiceJet', 'Vistara', 'AirAsia']
    sources = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata']
    destinations = ['Cochin', 'Hyderabad', 'Chennai', 'Delhi']
    
    data = {
        'Airline': np.random.choice(airlines, n),
        'Source': np.random.choice(sources, n),
        'Destination': np.random.choice(destinations, n),
        'Duration_Mins': np.random.randint(60, 480, n),
        'Stops': np.random.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1]),
        'Days_Before_Flight': np.random.randint(1, 60, n)
    }
    
    df = pd.DataFrame(data)
    df['Price'] = (df['Duration_Mins'] * 10) + (df['Stops'] * 1500) + \
                  (10000 / (df['Days_Before_Flight'] + 1)) + np.random.randint(2000, 5000, n)
    return df

df = load_data()

# Model Training
@st.cache_resource
def train_model(data):
    X = pd.get_dummies(data.drop('Price', axis=1))
    y = data['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, model_columns = train_model(df)

# Sidebar UI
st.sidebar.header("Plan Your Trip")
airline = st.sidebar.selectbox("Select Airline", df['Airline'].unique())
source = st.sidebar.selectbox("Source City", df['Source'].unique())
dest = st.sidebar.selectbox("Destination City", df['Destination'].unique())
stops = st.sidebar.slider("Number of Stops", 0, 2, 0)
duration = st.sidebar.slider("Duration (Mins)", 60, 600, 120)
days_ahead = st.sidebar.slider("Days Before Flight", 1, 60, 7)

# Visualizations
st.subheader("Price Trends & Insights")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x='Price', color='Airline', title='Price Distribution by Airline')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter(df, x='Days_Before_Flight', y='Price', color='Airline', opacity=0.5, title='Booking Window vs Price')
    st.plotly_chart(fig2, use_container_width=True)

# Prediction Logic
if st.button("Calculate Estimated Fare"):
    input_data = pd.DataFrame({
        'Airline': [airline],
        'Source': [source],
        'Destination': [dest],
        'Duration_Mins': [duration],
        'Stops': [stops],
        'Days_Before_Flight': [days_ahead]
    })
    
    # One-hot encoding for input
    input_encoded = pd.get_dummies(input_data)
    
    # Align with model columns
    final_input = pd.DataFrame(columns=model_columns).fillna(0)
    for col in input_encoded.columns:
        if col in final_input.columns:
            final_input.loc[0, col] = input_encoded.loc[0, col]
    
    # Non-encoded features
    final_input['Duration_Mins'] = duration
    final_input['Stops'] = stops
    final_input['Days_Before_Flight'] = days_ahead
    
    # Ensure all columns are present and in correct order
    final_input = final_input.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(final_input)[0]
    
    st.markdown("---")
    st.metric(label="Estimated Fare", value=f"INR {prediction:,.2f}")
    st.info("Pricing relies on historical trends and booking timing.")
