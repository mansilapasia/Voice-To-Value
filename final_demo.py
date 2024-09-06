import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import altair as alt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import mysql.connector
from streamlit_navigation_bar import st_navbar
from streamlit_option_menu import option_menu
from db_utils import load_db_config,connect_to_database,fetch_data_from_table,close_connection
from home import home
from analytics import analytics
from contact import contact

# Load the database configuration
db_config = load_db_config()

# Connect to the database
conn = connect_to_database(db_config)

# Fetch data from a table
table_name = 'patient_sentiment'
df = fetch_data_from_table(conn, table_name)

st.set_page_config(
    page_title="Voice-To-Value",
    page_icon="logo.png",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@voicetovalue.com',
        'Report a bug': 'mailto:support@voicetovalue.com',
        'About': "Patient Sentiment Feedback"
    }
)


pages = ["Home",  "Analytics", "Contact"]
styles = {
    "nav": {
        "background-color": "#048ABF",  # Changed to #2B89A5 #0E869E #048ABF #078C7F
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "#ffffff",
    },
}


page = st_navbar(pages, styles=styles)
if page == "Home":
    home()
elif page == "Analytics":
    analytics()
elif page == "Contact":
    contact()

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color:  #FFFFFF;  
    }
</style>
""", unsafe_allow_html=True)


kpi_html = """
<style>
    .kpi-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 9px;
        margin-top: -50px; /* Adjusted to provide space below the title */
    }
    .kpi-card {
        background-color: #e0e0e0;
        padding: 7px;
        border-radius: 3px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: bold;
        color: #333;
    }
    .kpi-value {
        font-size: 15px;
        font-weight: bold;
        color: #0E869E;
    }
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        margin-top: -90px; /* Adjust this value to move the title up or down */
        margin-bottom: -90px; /* Adjust this value to provide space below the title */
        text-align: center;
    }
</style>
"""

# Add the title to the sidebar with custom CSS
st.sidebar.markdown(f"""
    <div class="sidebar-title">Staff Overview</div>
""", unsafe_allow_html=True)

# Display the KPI cards in the sidebar
st.sidebar.markdown(kpi_html + """
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-title">NHS hospitals</div>
        <div class="kpi-value">20</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Total staff</div>
        <div class="kpi-value">1765</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Doctor</div>
        <div class="kpi-value">200</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Nurse</div>
        <div class="kpi-value">300</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Paramedic</div>
        <div class="kpi-value">56</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Admin Staff</div>
        <div class="kpi-value">45</div>
    </div>
</div>
""", unsafe_allow_html=True)

 # Function to calculate overall sentiment score
def calculate_overall_sentiment_score(data):
    overall_sentiment_scores = data['Predicted_Sentiment'].value_counts(normalize=True)
    positive_score = overall_sentiment_scores.get('positive', 0)  # Adjusted for your sentiment labels
    return positive_score

# Function to calculate average feedback length
def calculate_average_feedback_length(df):
    df['Feedback'] = df['Feedback'].fillna('')  # Replace NaN with empty string or handle differently
    df['Length'] = df['Feedback'].apply(len)
    average_length = df['Length'].mean()
    return average_length

# Calculate average feedback length
average_length = calculate_average_feedback_length(df)

#function to calculate criticality mode
def calculate_criticality_mode(df):
    mode_value = df['Criticality'].mode().iloc[0]
    return mode_value

# Function to create a donut chart for sentiment score
def create_donut_chart_score(label, value, color, title):
    fig = go.Figure(go.Pie(
        labels=[label, ''],
        values=[value, 1 - value],
        hole=0.7,
        marker_colors=[color, 'lightgrey'],
        textinfo='none'
    ))
    
    fig.update_layout(
        title=dict(text=title, font_size=13),  # Title of the chart
        title_x=0.5,  # Center the title horizontally
        title_xanchor='center',  # Align title to the center horizontally
        annotations=[dict(text=f'{value:.2f}', x=0.5, y=0.5, font_size=20, showarrow=False)],  # Value inside the chart
        showlegend=False,
        margin=dict(t=30, b=10, l=10, r=10),  # Adjust margins for better spacing
        width=150,  # Adjust width as needed
        height=150,  # Adjust height as needed
    )
    
    return fig

# Function to create a donut chart for number of feedbacks
def create_donut_chart_number(label, value, total, color, title):
    fig = go.Figure(go.Pie(
        labels=[label, 'Total'],
        values=[value, total - value],
        hole=0.7,
        marker_colors=[color, 'lightgrey'],
        textinfo='none'
    ))
    
    fig.update_layout(
        title=dict(text=title, font_size=13),  # Title of the chart
        title_x=0.5,  # Center the title horizontally
        title_xanchor='center',  # Align title to the center horizontally
        annotations=[dict(text=f'{value}', x=0.5, y=0.5, font_size=20, showarrow=False)],  # Value inside the chart
        showlegend=False,
        margin=dict(t=30, b=10, l=10, r=10),  # Adjust margins for better spacing
        width=150,  # Adjust width as needed
        height=150,  # Adjust height as needed
    )
    
    return fig

# Function to create a donut chart for average feedback length
def create_average_feedback_length_donut_chart(length, title):
    fig = go.Figure(go.Pie(
        labels=['Average Feedback Length', ''],
        values=[length, 100 - length],
        hole=0.7,
        marker_colors=['#048ABF', 'lightgrey'],
        textinfo='none'
    ))
    
    fig.update_layout(
        title=dict(text=title, font_size=13),  # Title of the chart
        title_x=0.5,  # Center the title horizontally
        title_xanchor='center',  # Align title to the center horizontally
        annotations=[dict(text=f'{length:.2f}', x=0.5, y=0.5, font_size=20, showarrow=False)],  # Length inside the chart
        showlegend=False,
        margin=dict(t=30, b=10, l=10, r=10),  # Adjust margins for better spacing
        width=150,  # Adjust width as needed
        height=150,  # Adjust height as needed
    )
    
    return fig

#function to cretae donut chart for criticality mode
def create_donut_chart_criticality_mode(mode_value, title):
    fig = go.Figure(go.Pie(
        labels=[f'Criticality Mode: {mode_value}', ''],
        values=[1, 0],  # Use 1 and 0 to represent 100% for the single category
        hole=0.7,
        marker_colors=['#048ABF', 'lightgrey'],
        textinfo='none'
    ))
    
    fig.update_layout(
        title=dict(text=title, font_size=13),  # Title of the chart
        title_x=0.5,  # Center the title horizontally
        title_xanchor='center',  # Align title to the center horizontally
        annotations=[dict(text=f'{mode_value}', x=0.5, y=0.5, font_size=20, showarrow=False)],  # Mode value inside the chart
        showlegend=False,
        margin=dict(t=30, b=10, l=10, r=10),  # Adjust margins for better spacing
        width=150,  # Adjust width as needed
        height=150,  # Adjust height as needed
    )
    
    return fig

# Display the donut charts in the sidebar
overall_sentiment_score = calculate_overall_sentiment_score(df.copy())
total_feedbacks = df.shape[0]
average_length = calculate_average_feedback_length(df.copy())
criticality_mode = calculate_criticality_mode(df.copy())


# Sidebar title
st.sidebar.markdown("<h2 style='text-align: center; font-size: 24px; margin-top: 0;'>Metrics</h2>", unsafe_allow_html=True)

# Display the donut charts in the sidebar in a grid
with st.sidebar:
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_donut_chart_score("Overall Sentiment Score", overall_sentiment_score, '#048ABF', "Overall Sentiment Score"), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_donut_chart_number("Feedbacks", total_feedbacks, total_feedbacks, '#048ABF', "Number of Feedbacks"), use_container_width=True)
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.plotly_chart(create_average_feedback_length_donut_chart(average_length, "Average Feedback Length"), use_container_width=True)
    
    with col4:
        st.plotly_chart(create_donut_chart_criticality_mode(criticality_mode, "Criticality Mode"), use_container_width=True)

