# Voice-To-Value
"Voice to Value" is an innovative sentiment analysis tool designed to address the needs of small and medium healthcare facilities by providing real-time insights into patient feedback. Leveraging advanced machine learning and natural language processing (NLP) techniques, the tool extracts and analyzes unstructured patient reviews from online sources. 
It classifies sentiments as positive, negative, or neutral and performs topic analysis to identify feedback categories such as staff, communication, facilities, treatment quality, etc. This detailed categorization helps healthcare providers pinpoint specific areas of concern and address them effectively. 
The insights are presented via a user-friendly dashboard built with Streamlit, enabling healthcare providers to proactively enhance patient care. 
The robust ETL (Extract, Transform, Load) workflow ensures clean and relevant data through pre-processing techniques like tokenization and lemmatization, all implemented within a Python framework.

# Analytical Techniques
The analytical techniques identified to achieve the goal uses NLP and machine learning models. The service uses sentiment analysis techniques to interpret patient sentiments by using a technological stack of Python and NLP to investigate how expectations from the healthcare industry, and other factors may affect patient satisfaction. Sentiment analysis, commonly known as opinion mining, is the process of obtaining, identifying, extracting, and analyzing data to determine if it is neutral, positive, or negative using artificial intelligence and NLP. Text analytics is the process of examining through unstructured text, identifying pertinent information, and converting it into useful business knowledge and sentiment analysis determines if a statement is positive, negative, or neutral. 

Based on the dataset available, the analytical components to predict patient satisfaction involve four major steps. 
The first step in the process includes data collection and preprocessing that involves tokenization, removing stop words and special characters and text lowercasing followed by the best step of feature extraction of turning unprocessed text into numerical representations for machine learning models. Term Frequency-Inverse Document Frequency (TF-IDF), sentiment lexicons, N-grams, Bag-of-Words, are some of the common techniques which extract semantic and contextual information from text. 
Next, the available datasets allow for both topic classification which is a supervised machine learning approach that has predefined topics category. 
The fourth step involves the classifiers for sentiment analysis. Upon performing traditional machine learning models that are Naïve Bayes, Logistic regression, random forest, and support vector machine classifier (SVC) which are commonly used classifiers that give efficient results, the service leverages SVC machine learning model which had the best accuracy of 81%.
Figure 1
![image](https://github.com/user-attachments/assets/452b0695-2fdc-4aea-ba56-8f6f9dab4bf0)
Figure 1 compares the ML model with evaluation metrics such as recall, precision, F1-Score and accuracy to help decide the model evaluation and validation. The generation of confusion matrix in figure, further provides a better understanding of the working of models.

Figure 2
![image](https://github.com/user-attachments/assets/e0a3138e-73d8-4f29-b94f-fdfbf5865d1b)
Figure 2 displays the Confusion Matrix metrics of the considered ML Models. The model learns about the patterns and association between sentiment labels and features during training, to make predictions on the new text. 

The predicted sentiment results of the model is represented in clearly understandable visualizations that provide the customer i.e. doctors, hospital administrators, etc. insight about the feedback given by patients and the measures that can be undertaken to improve the experience along with helping with resource allocation, providing equitable healthcare delivery for diverse patient populations and most importantly assisting private healthcare organizations in complying to accreditation of standards and regulatory requirements, enhancing compliance overall and enhancing reputation.

# Dataset
The National Health Service (NHS) Choice’s dataset, which is accessible on data.gov.uk and GitHub, exhibits a remarkable quality as it presents an extensive compilation of patient feedback and ratings gathered through the NHS friends and family test. Because it provides understanding on a variety of healthcare service-related topics such as the labels, categories, advice given by organization, criticality, timeframe, etc. the dataset is extensive and useful for forecasting patient satisfaction.

# ETL Workflow
![image](https://github.com/user-attachments/assets/99635b6e-8eb7-4b02-8471-125c5d6fc6e4)
The ETL (Extract, Transform, Load) workflow for patient sentiment analysis of NHS patient feedback data, leveraging resources for scalability, security, and collaboration is displayed in figure above.

● Quality Assurance: Rigorous cleaning, validation, and de-duplication processes are implemented to maintain data integrity.

●Workflow Design: Our ETL process is streamlined for efficiency and accuracy, minimizing the need for manual intervention. 

● Scalability: The ETL pipeline is designed to handle larger datasets and frequent uploads as needed.

● Data Security: Robust protocols are in place throughout the ETL pipeline to protect patient confidentiality. We adhere strictly to healthcare data regulations to safeguard sensitive information. 

● Monitoring & Verification: Automated checks are performed after each ETL cycle to ensure completeness and accuracy of the data. 

The workflow starts with the CSV Dataset being loaded into the MySQL database followed by using Python scripts for EDA, feature engineering and ML models to predict sentiment from the patient feedback data then the predictions generated by the ML Models are stored back to the database. The final step involves the predicted sentiment dataset connection to the visualization dashboard created using Streamlit.

Text preprocessing steps
1. Lowercasing: Converts all characters to lowercase to ensure uniformity. 
2. Replacing Contractions: Replaces the contraction "n't" with "not" to standardize the text. 
3. Removing URLs: Removes URLs to eliminate irrelevant information. 
4. Removing User Mentions: Removes mentions (e.g., @username) to focus on the actual content. 
5. Removing HTML Entities: Removes HTML entities (e.g., &). 
6. Removing Punctuation: Replaces punctuation with spaces to clean the text. 
7. Removing Extra White Spaces: Replaces multiple spaces with a single space for proper formatting. 
8. Removing Non-alphabetic Characters: Removes any non-alphabetic characters to focus on words. 
9. Tokenization: Splits the text into individual words (tokens). 
10. Lemmatization and Stop Words Removal: Lemmatizes tokens to their base form and removes common stop words. After preprocessing the patient feedback data, the next step in sentiment analysis using Machine learning models for predicting sentiments. Predicted Sentiment is later visualized using Streamlit.

# PESTAL Analysis
PESTEL is a strategic business analysis tool used to identify factors in the Political, Economic, Social, Technological, Environmental, and Legal (PESTEL) landscape that can impact an organization.The PESTEL analysis for the Go-To-Market risks associated with the Voice to Value organization is analyzed below.
![image](https://github.com/user-attachments/assets/0835da50-e8c9-43bb-b861-0cb2dba6877d)

# Business Model Canvas
![image](https://github.com/user-attachments/assets/8c5784be-48b1-477c-a2f1-41a918999e27)

# Snippets of the Final Dashboard
Landing Page
<img width="959" alt="Landing_page_1" src="https://github.com/user-attachments/assets/c272c982-d716-45bf-991d-7fcdb95149a7">
<img width="958" alt="Landing_Page_2" src="https://github.com/user-attachments/assets/fb992cc1-cc50-40cf-87e1-28766130c0d0">

Overall Analytics Tab
<img width="959" alt="Overall_Analytics_1" src="https://github.com/user-attachments/assets/4ab4333f-a033-4253-8e41-ea5f2a1c6d49">
<img width="960" alt="Overall_Analytics_2" src="https://github.com/user-attachments/assets/8bf87d12-d3cb-4e45-bfac-09e66748b989">
<img width="960" alt="Overall_Analytics_Wordcloud_2" src="https://github.com/user-attachments/assets/3854c41f-c514-4588-95aa-2a3ea7d1a876">
<img width="960" alt="Overall_Analytics_Criticality_chart" src="https://github.com/user-attachments/assets/e9fd05cb-94bf-4470-a6a8-8cb0b382332d">
<img width="959" alt="Overall_Analytics_distrbution_topic_bar" src="https://github.com/user-attachments/assets/ac7ddbb6-da6e-4a12-a1e9-efc98850566b">
<img width="960" alt="Overall_Analytics_Sentiment_chart" src="https://github.com/user-attachments/assets/c8e3b6ef-6c5e-4af2-9b2f-51e68e8761a6">
<img width="960" alt="Overall_Analytics_Treemap_1" src="https://github.com/user-attachments/assets/14ad0526-f0de-40bd-8dd4-0fbd70ce0378">
<img width="960" alt="Overall_Analytics_Treemap_2" src="https://github.com/user-attachments/assets/c95b4f91-f269-4321-ae58-386bd0283ab3">

Positive Feedback Tab
<img width="959" alt="Positive_feeback_count" src="https://github.com/user-attachments/assets/41ba8276-0eba-4bca-afa0-007137b1cc41">

Negative Feedback Tab
<img width="959" alt="negative_feedback_count" src="https://github.com/user-attachments/assets/ae9d76b9-ed7c-48d4-81de-2f549ddcf507">

Topic ans Subcategory Tab
<img width="959" alt="topic_access_bar" src="https://github.com/user-attachments/assets/f35401ed-0b0a-442a-bc31-7c249d4ab890">
<img width="959" alt="subcategory_access" src="https://github.com/user-attachments/assets/b37a74ce-d425-48b1-8be5-d9e012ec1b66">

Contact Tab
<img width="958" alt="contact" src="https://github.com/user-attachments/assets/285ee731-8605-4195-976c-ffd7bf9b6a77">










