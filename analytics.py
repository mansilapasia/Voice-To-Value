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

# Load the database configuration
db_config = load_db_config()

# Connect to the database
conn = connect_to_database(db_config)

# Fetch data from a table
table_name = 'patient_sentiment'
df = fetch_data_from_table(conn, table_name)

sentiment_colors = ['#0897B4', '#4C4A59', '#F2CDAC'] # Positive, Neutral, Negative

# Function to clean text
@st.cache_data 
def clean_text(text):
    punctuation_pattern = r"[^A-Za-z0-9]"  # Pattern to remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(punctuation_pattern, ' ', text)  # Replace non-alphanumeric characters with a space
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Function to load custom stopwords from a file
def load_custom_stopwords(file_path):
    with open(file_path, 'r') as file:
        custom_stopwords = set(file.read().splitlines())
    return custom_stopwords

# Function to generate and plot word cloud for a given sentiment
@st.cache_data
def plot_wordcloud_score(sentiment, df, custom_stopwords_file='custom_stopword.txt'):
    # Load custom stopwords
    custom_stopword = load_custom_stopwords(custom_stopwords_file)
    
    # Filter the DataFrame by sentiment
    sentiment_df = df[df['Predicted_Sentiment'] == sentiment]
    
    # Join all the feedback for the given sentiment
    text = ' '.join(sentiment_df['Feedback'].astype(str).apply(clean_text))
    
    # Remove NLTK stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stopword)
    filtered_text = ' '.join(word for word in text.split() if word not in stop_words)
    
    # Generate the word cloud
    wc = WordCloud(max_words=200,  # Limit to 20 words
                   min_font_size=10,
                   height=800,
                   width=1600,
                   background_color="white",
                   colormap='viridis').generate(filtered_text)
    
    # Plot the word cloud
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    #ax.set_title(f'{sentiment.capitalize()} Feedback', fontsize=20)
    st.pyplot(fig)

#word cloud for a given label sentiment
@st.cache_data
def plot_wordcloud(label, sentiment, df, custom_stopwords_file='custom_stopword.txt'):
    # Load custom stopwords
    custom_stopword = load_custom_stopwords(custom_stopwords_file)
    
    # Filter the DataFrame by label and sentiment
    filtered_df = df[(df['Label'] == label) & (df['Predicted_Sentiment'] == sentiment)]
    
    # Join all the feedback for the given label and sentiment
    text = ' '.join(filtered_df['Feedback'].astype(str).apply(clean_text))
    
    # Remove NLTK stopwords and custom stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stopword)
    filtered_text = ' '.join(word for word in text.split() if word not in stop_words)

    if len(filtered_text) > 0:
        # Generate the word cloud
        wc = WordCloud(max_words=250,
                       min_font_size=10,
                       height=400,
                       width=800,
                       background_color="white",
                       colormap='viridis').generate(filtered_text)
        
        # Plot the word cloud using Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{label.capitalize()} - {sentiment.capitalize()} Feedback', fontsize=16)
        st.pyplot(fig)
    else:
        st.warning("No valid words found to generate a word cloud.")
  
#overall word cloud 
@st.cache_data 
def plot_wordcloud_overall(df,custom_stopwords_file='custom_stopword.txt'):
    custom_stopword = load_custom_stopwords(custom_stopwords_file)
    # Join all the feedback for the given sentiment
    text = ' '.join(df['Feedback'].astype(str).apply(clean_text))
    
    # Remove NLTK stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stopword)
    filtered_text = ' '.join(word for word in text.split() if word not in stop_words)
    
    # Generate the word cloud
    wc = WordCloud(max_words=200,  # Limit to 200 words
                   min_font_size=10,
                   height=800,
                   width=1600,
                   background_color="white",
                   colormap='viridis').generate(filtered_text)
    
    # Plot the word cloud
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

#function to add info tooltip
def add_info_icon_tooltip(text):
    """
    Displays an information icon with tooltip on hover.

    Parameters:
    - text (str): The text to display in the tooltip.
    """
    info_icon = f"""
    <style>
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 250px;
            background-color: #f9f9f9;
            color: #000;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            top: 125%; /* Position the tooltip below the icon */
            left: 20%;
            transform: translateX(-50%);
            box-shadow: 0px 0px 10px 0px rgba(0,0,0,0.2);
            opacity: 0; /* Initially hidden */
            transition: opacity 0.3s; /* Fade-in transition */
        }}
        .tooltip .tooltiptext::after {{
            content: "";
            position: absolute;
            top: -10px; /* Adjust the arrow position */
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: transparent transparent #f9f9f9 transparent;
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1; /* Show tooltip on hover */
        }}
    </style>
    <div style="text-align: right;">
    <div class="tooltip">
        <i class="fas fa-info-circle"></i>
        <span class="tooltiptext">{text}</span>
    </div>
    </div>
    """
    st.markdown(info_icon, unsafe_allow_html=True)    

#function to calculate overall sentiment score
def calculate_sentiment_scores(df):
    total_feedback = len(df)
    overall_positive_score = (df['Predicted_Sentiment'] == 'positive').sum() / total_feedback * 100
    overall_negative_score = (df['Predicted_Sentiment'] == 'negative').sum() / total_feedback * 100
    overall_neutral_score = (df['Predicted_Sentiment'] == 'neutral').sum() / total_feedback * 100
    
    return overall_positive_score, overall_negative_score, overall_neutral_score

#function to create donut chart for overall sentiment score
def create_donut_chart(positive_score, negative_score, neutral_score):
    # Prepare data for donut chart
    labels = ['Positive', 'Negative', 'Neutral']
    values = [positive_score, negative_score, neutral_score]
    colors = ['#3CB371', '#E34234', '#FFD700']
    
    # Format values with percentage symbol
    formatted_values = [f"{round(val, 2)}%" for val in values]
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=.5, 
        marker=dict(colors=colors),
        hoverinfo='label+percent',
        textinfo='text+percent',
        #text=formatted_values
    )])
    
    fig.update_layout(
        annotations=[dict(text='Sentiment', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

color_map = {'positive': '#3CB371', 'negative': '#E34234', 'neutral': '#FFD700'}

def analytics():

    #Use Streamlit columns to place image and title/description beside each other
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("logo.png", width=200, use_column_width=True)

    with col2:
        st.markdown(
            """
            <div style="margin-left: 20px;">
                <h1 style="margin: 0;">Welcome to your Dashboard!</h1>
            </div>
            """, unsafe_allow_html=True
        )
        add_info_icon_tooltip(
    "<strong>How Data Visualizations Help You Make Decisions</strong>" +
    "<ul>" +
    "<li>Quickly understand complex healthcare metrics.</li>" +
    "<li>Spot trends over time to anticipate needs.</li>" +
    "<li>Identify areas for improvement.</li>" +
    "<li>Communicate effectively with your team.</li>" +
    "<li>Make data-driven decisions to enhance care and operations.</li>" +
    "</ul>" 
    ); 

    # Load Font Awesome
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
        """, unsafe_allow_html=True)

    # Tabs
    selected = option_menu(
        menu_title=None,
        options=["Overall Analysis", "Positive Feedback", "Negative Feedback", "Topics"],
        icons=["bar-chart", "hand-thumbs-up", "hand-thumbs-down", "palette"],  
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#ffffff"},
            "icon": {"font-size": "18px", "color": "black"},  # Smaller icon size and black color
            "nav-link": {
                "font-size": "14px",  # Smaller font size for options
                "text-align": "center",
                "margin": "0px",
                "flex-grow": "1",  # Stretch the tabs to fill the container
                "padding": "10px",
                "border-radius": "0px",  #  tabs have no rounded corners
                "border": "1px solid transparent",  # border style is consistent
                "color": "black",  # Font color remains black
                "background-color": "white"  # Tab background color
            },
            "nav-link-selected": {
                "font-size": "14px",  # Same font size for selected option
                "background-color": "#048ABF",  # Highlight color for selected tab
                "border": "1px solid #ccc",  # Border for selected option to avoid enlargement
                "color": "black"  # Font color remains black for selected tab
            }
        }
    )
    if selected=="Overall Analysis":
        
        st.subheader("Wordcloud")
        add_info_icon_tooltip(
        "<strong>WordCloud</strong>" +
        "<p>This is a word cloud! It visually shows the most frequent words in the feedback collected.</p>" +
        "<p>Bigger words appear more often in the feedback.</p>" +
        "<p>Use this to see which words stand out and get a quick sense of the main topics.</p>")
        plot_wordcloud_overall(df)

        # Define content for Decision-Making section
        insight_content = """
        <p style='text-align: left; font-size: 17px;'>
        The word cloud represents the overall feedback collected. It highlights that staff members are doing many things well, particularly in terms of communication, support, and providing a caring environment. The word cloud could be used to inform decisions around staff training, resource allocation, and process improvement. For example, if 'wait' appears frequently, it might be worth exploring ways to reduce wait times.
        </p>
        """
    
        # Display expanders
        with st.expander("Decision Insight", expanded=False):
            st.markdown(insight_content, unsafe_allow_html=True)

        #overall sentiment score
        st.subheader("Overall Sentiment Score")
        add_info_icon_tooltip(
        "<strong>Overall Summary</strong>"+
        "<p>This Donut Chart provides an overview of the overall sentiment from feedback received:</p>"+
        "<ul><li><strong>Positive:</strong> Percentage of feedback categorized as positive.</li><li><strong>Negative:</strong> Percentage of feedback categorized as negative.</li><li><strong>Neutral:</strong> Percentage of feedback categorized as neutral.</li></ul>"+
        "<p>The chart helps in understanding the distribution of sentiments among respondents.</p>")

        # Calculate sentiment scores
        positive_score, negative_score, neutral_score = calculate_sentiment_scores(df)
        # Create and display the donut chart
        fig = create_donut_chart(positive_score, negative_score, neutral_score)
        st.plotly_chart(fig, use_container_width=True)
        
        #distribution of sentiment by Criticality
        st.subheader('Distribution of Sentiment by Criticality')
        
        # Create the Plotly figure
        fig = px.histogram(
            df, 
            x="Criticality", 
            color="Sentiment", 
            barmode="group",
            category_orders={"Sentiment": ["negative", "neutral", "positive"]},
            labels={"Criticality": "Level of Criticality"},
            color_discrete_map=color_map  # Use the color map
        )

        # Customize the layout
        fig.update_layout(
            xaxis_title="Level of Criticality",
            yaxis_title="Count",
            legend_title="Sentiment",
            xaxis=dict(showgrid=False),  # Remove x-axis grid lines
            yaxis=dict(showgrid=False)  
        )

        # Display the Plotly figure in Streamlit
        st.plotly_chart(fig)
        
        insights_content = """
        The majority of feedback is positive, concentrated around a criticality level of 3. 
        Negative feedback is minimal, primarily at lower criticality levels. 
        Neutral feedback peaks at level 0, suggesting a need to understand and address this segment.
        """

        # Display the insights in an expander format
        with st.expander("Insights from the bar chart", expanded=False):
            st.markdown(insights_content)

        # Sentiment Distribution by Labels
        st.subheader("Sentiment Distribution by Topics")

        insights_content = """
        Staff feedback is overwhelmingly positive. 
        Significant negative feedback exists for environment/facilities and for Transition/co-ordination, requiring urgent attention. 
        Communication is mostly positive, but negative sentiment warrants investigation.
        """
            
        def create_sentiment_distribution_by_label_chart(df, color_map):
            sentiment_counts = (
                df.groupby(["Label", "Predicted_Sentiment"])
                .size()
                .to_frame(name="Count")
                .reset_index()
        )
            sentiment_distribution = sentiment_counts.pivot_table(
            index="Label", columns="Predicted_Sentiment", values="Count", aggfunc="sum"
        ).fillna(0)

            # Unstack data for bar chart
            sentiment_distribution = sentiment_distribution.loc[sentiment_distribution.sum(axis=1).sort_values(ascending=False).index]
            sentiment_distribution.fillna(0, inplace=True)  # Replace missing values with 0

            # Create plotly figure for stacked bar chart
            fig = go.Figure()

            # Loop through sorted categories and corresponding colors
            for sentiment, color in color_map.items():
                # Clean sentiment label (optional, if hidden characters are a concern)
                cleaned_sentiment = sentiment.strip()
                fig.add_trace(go.Bar(x=sentiment_distribution.index, y=sentiment_distribution[sentiment], name=cleaned_sentiment, marker_color=color))

            fig.update_layout(
                barmode="stack", 
                xaxis_title="Topic", 
                yaxis_title="Count",
                legend_title="Sentiment"
            )
            return fig

        # Display the stacked bar plot
        st.plotly_chart(create_sentiment_distribution_by_label_chart(df, color_map), use_container_width=True)
        with st.expander("Insights from the Stacked Bar Chart", expanded=False):
            st.markdown(insights_content)

        with st.expander("Additional Visualization by using Treemap", expanded=False):
            add_info_icon_tooltip("<strong>Overall Summary</strong>"+
                    "<ul>A treemap uses nested rectangles to depict hierarchical data; size and color show quantitative values.</ul>"+
                    "<ul> Patient feedback topics like staff, access, and communication are shown.</ul>"+
                    "<ul> Larger rectangles mean more data; colors distinguish categories. </ul>"+
                    "<ul>Treemaps aid understanding of data patterns and enable comparisons, useful for analyzing complex datasets and guiding decision-making.</ul>")
        
            # Calculate sentiment counts by label
            sentiment_counts = df.groupby(['Label', 'Predicted_Sentiment']).size().reset_index(name='Count')
            color_palette = px.colors.qualitative.G10
            # Create a treemap
            fig_treemap = px.treemap(sentiment_counts, 
                                    path=['Label', 'Predicted_Sentiment'], 
                                    values='Count',
                                    color='Label',
                                    color_discrete_sequence=color_palette
                                    )
            
            # Update layout
            fig_treemap.update_layout(
                template='plotly_white'
            )
            # Display the treemap in Streamlit
            st.plotly_chart(fig_treemap)

        # dataframe for output
        st.subheader("Feedback and Predicted Sentiment")
        st.dataframe(df[['Feedback', 'Predicted_Sentiment']], use_container_width=True)
    elif selected=="Positive Feedback":
        st.subheader("Positive Sentiment Score")
        positive_df = df[df['Predicted_Sentiment'] == 'positive']
        positive_score = (df['Predicted_Sentiment'] == 'positive').mean()
        positive_score, negative_score, neutral_score = calculate_sentiment_scores(df)

        # Data for the donut chart
        labels = ['Positive Score', 'Rest of Sentiments']
        values = [positive_score, 100 - positive_score]

        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=.5, 
            #marker=dict(colors=colors),
            marker_colors=['#3CB371', '#E5E4E2'],
            hoverinfo='label+percent',
            textinfo='text+percent',
            #text=formatted_values
        )])
        
        fig.update_layout(
            annotations=[dict(text='Sentiment', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig)

        #plot positive word cloud
        st.subheader("Positive Word Cloud")
        plot_wordcloud_score('positive', df)
        positive_insight="""
        In the positive word cloud, keywords like 'staff,' 'care,' 'caring,' 'nurse,' and 'professional' shine brightly, reflecting patients' appreciation for compassionate care and the exceptional support provided by the nursing staff.
        """
        with st.expander("Insights", expanded=False):
            st.markdown(positive_insight) 

        def get_top_positive_feedback(df):
            # Filter by positive sentiment
            positive_df = df[df['Predicted_Sentiment'] == 'positive']
            
            # Sort by Length (descending), then by Critical (descending)
            sorted_df = positive_df.sort_values(by=['Length', 'Criticality'], ascending=[False, False])
            
            # Select top 5 entries
            top_positive_feedback = sorted_df.head(5)
            
            return top_positive_feedback[['Feedback', 'Length', 'Criticality']]
        
        # Call function to get top positive feedback
        top_positive = get_top_positive_feedback(df)

        # Display the result in a table
        st.subheader("Top 5 Positive Feedback with Highest Criticality Score")
        st.dataframe(top_positive)

        # Additional insights text
        positive_insight="""
        - Thorough assessments and individualized care plans are highly valued by users and should be emphasized in service delivery.
        - Recognize and celebrate individual staff members for their positive contributions to the user experience.
        - Even with limited service engagement, a positive overall approach and interactions leave a lasting impression on users.
        - Consider adding an "Excellent" option to feedback surveys to better capture the full range of positive user sentiment.
        - By focusing on these key areas, we can further enhance user satisfaction and optimize our services.
        """
        with st.expander("Insights", expanded=False):
            st.markdown(positive_insight) 

        st.subheader("Positive Feedback and Predicted Sentiment")
        st.dataframe(positive_df[['Feedback', 'Predicted_Sentiment']], use_container_width=True)
    elif selected=="Negative Feedback":
        st.subheader("Negative Sentiment Score")
        negative_df = df[df['Predicted_Sentiment'] == 'negative']
        negative_score = (df['Predicted_Sentiment'] == 'negative').mean()
        positive_score, negative_score, neutral_score = calculate_sentiment_scores(df)
        # Data for the donut chart
        labels = ['Negative Score', 'Rest of Sentiments']
        values = [negative_score,  100-negative_score]

        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=.5, 
            #marker=dict(colors=colors),
            marker_colors=['#E34234', '#E5E4E2'],
            hoverinfo='label+percent',
            textinfo='text+percent',
            #text=formatted_values
        )])
        
        fig.update_layout(
            annotations=[dict(text='Sentiment', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig)

        #plot negative word cloud
        st.subheader("Negative Word Cloud")
        plot_wordcloud_score('negative', df)
        negative_insight="""
        In the negative word cloud, keywords such as 'time,' 'food,' 'staff,' 'service,' and 'appointment' stand out prominently, highlighting patients' frustrations with scheduling issues, meal quality, interactions with staff, overall service, and appointment management.
        """
        with st.expander("Insights", expanded=False):
            st.markdown(negative_insight) 

        def get_top_negative_feedback(df):
            # Filter by positive sentiment
            positive_df = df[df['Predicted_Sentiment'] == 'negative']
            
            # Sort by Length (descending), then by Critical (descending)
            sorted_df = positive_df.sort_values(by=['Length', 'Criticality'], ascending=[False, False])
            
            # Select top 5 entries
            top_negative_feedback = sorted_df.head(5)
            
            return top_negative_feedback[['Feedback', 'Length', 'Criticality']]

        # Call function to get top positive feedback
        top_positive = get_top_negative_feedback(df)

        # Display the result in a table
        st.subheader("Top 5 Negative Feedback with Highest Criticality Score")
        st.dataframe(top_positive)
        with st.expander("Key Insights", expanded=False):
            st.markdown("""
            - Urgent issues with premature birth care and staff behavior: The most critical feedback highlights negative experiences related to premature birth care and a doctor's perceived lack of empathy. Immediate attention is needed to address these serious concerns and prevent further negative impact on patients and families.
            - Website usability and therapist punctuality need improvement: Feedback on website inefficiency and therapist tardiness, while less critical, still indicate significant areas for improvement to enhance the overall patient experience.
            - Consider additional support for caregivers: A caregiver's struggle highlights the need to potentially offer resources and support for those caring for patients with dementia.
            """)
        
        # Recommendations expander
        with st.expander("Recommendations", expanded=False):
            st.markdown("""
            - Thoroughly investigate and address the negative premature birth experience: Conduct a comprehensive review of the specific case mentioned and implement necessary changes to protocols, training, or resources to ensure proper care for premature infants and their families.
            - Address concerns regarding doctor's behavior: Investigate the feedback and consider additional training or intervention to ensure all doctors demonstrate empathy and respect towards patients.
            - Implement measures to improve therapist punctuality: Address the issue of therapist tardiness through stricter scheduling, reminders, or performance management.
            - Explore support services for caregivers: Offer resources, counseling, or support groups for caregivers of patients with dementia to help them cope with the challenges they face.
            - Actively solicit feedback and address concerns promptly: Encourage open communication with patients and families, address negative feedback constructively, and continuously improve services based on their input.
            """)

        st.subheader("Negative Feedback and Predicted Sentiment")
        st.dataframe(negative_df[['Feedback', 'Predicted_Sentiment']], use_container_width=True)
    elif selected=="Topics":
        st.markdown("### Dive Deeper into Patient Feedback:")
        st.markdown("""
        - Understand patient experiences with topics idenitifed such as access, care received, communication,etc.
        - Assess satisfaction with provision of services, appointment scheduling, wait times, and overall clinic processes.
        - Evaluate feedback on facility cleanliness, staff courtesy, and the overall environment.
        """)

        tabs = st.radio("Select what you want to analyse", ['Topic Analysis', 'Subcategory of Topic Analysis'])

        if tabs == 'Topic Analysis':
            labels = df['Label'].unique()
            selected_label = st.selectbox("Choose a Topic", labels)

            if selected_label:
                # Filter data for selected label
                filtered_df = df[df['Label'] == selected_label]

                # Calculate sentiment counts
                sentiment_counts = filtered_df['Predicted_Sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']

                # Create a bar chart for sentiment distribution
                fig_sentiment = px.bar(sentiment_counts, x='Sentiment', y='Count', 
                                    color='Sentiment', 
                                    title=f'Sentiment Distribution for Topic: {selected_label}',
                                    labels={'Sentiment': 'Sentiment Category', 'Count': 'Count'},
                                    color_discrete_map=color_map)

                # Display the bar chart in Streamlit
                st.plotly_chart(fig_sentiment)

                sentiment = st.selectbox('Select Sentiment', ['positive', 'negative'])
        
                # Display the word cloud based on user selection
                plot_wordcloud(selected_label, sentiment, df)

                st.subheader(f"Feedback for Topic: {selected_label}")
                st.dataframe(filtered_df[['Feedback', 'Label', 'Predicted_Sentiment']], use_container_width=True)

        elif tabs == 'Subcategory of Topic Analysis':
            labels = df['Label'].unique()
            selected_label = st.selectbox("Choose a Topic", labels)

            if selected_label:
                filtered_df = df[df['Label'] == selected_label]

                subcategories = filtered_df['Subcategory'].unique()
                selected_subcategory = st.selectbox("Choose a subcategory", subcategories)

                if selected_subcategory:
                    subcategory_df = filtered_df[filtered_df['Subcategory'] == selected_subcategory]

                    subcategory_sentiment_counts = subcategory_df['Predicted_Sentiment'].value_counts().reset_index()
                    subcategory_sentiment_counts.columns = ['Sentiment', 'Count']  # Rename columns for Plotly Express
                    fig_subcategory = px.bar(subcategory_sentiment_counts, 
                                            x='Sentiment', y='Count', 
                                            labels={'Sentiment': 'Sentiment', 'Count': 'Count'},
                                            color='Sentiment',
                                            title=f'Sentiment Distribution for Subcategory: {selected_subcategory}',
                                            color_discrete_map=color_map)
                    st.plotly_chart(fig_subcategory)

                
                    st.subheader(f"Feedback for Subcategory: {selected_subcategory}")
                    st.dataframe(subcategory_df[['Feedback', 'Label', 'Subcategory', 'Predicted_Sentiment']], use_container_width=True)

