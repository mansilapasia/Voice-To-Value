from db_utils import load_db_config, connect_to_database, fetch_data_from_table, close_connection, update_table_with_predictions
from preprocessing import text_preprocessing
from model_utils import train_models, evaluate_models, save_model_and_vectorizer, load_model_and_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def main():
    # Load the database configuration
    db_config = load_db_config()

    # Connect to the database
    conn = connect_to_database(db_config)

    # Fetch data from a table
    table_name = 'patient_sentiment'
    df = fetch_data_from_table(conn, table_name)

    # Apply preprocessing to the dataset
    df['Feedback'] = df['Feedback'].apply(text_preprocessing)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df['Feedback'], df['Sentiment'], test_size=0.2)

    # Convert each list of tokens into a single string
    X_train = [' '.join(doc) for doc in X_train]
    X_test = [' '.join(doc) for doc in X_test]

    # Initialize and fit TfidfVectorizer
    tfidf = TfidfVectorizer()
    train_tfidf_matrix = tfidf.fit_transform(X_train)
    test_tfidf_matrix = tfidf.transform(X_test)

    # Train models
    models, model_names = train_models(train_tfidf_matrix, y_train)

    # Evaluate models
    results = evaluate_models(models, model_names, test_tfidf_matrix, y_test)
    print(results)

    # Save the best model and vectorizer
    best_model = models[model_names.index('SVC')]  # Assuming SVC is the best
    save_model_and_vectorizer(best_model, tfidf)

    # Load the model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # Process feedbacks for prediction
    processed_feedbacks = [' '.join(tokens) if isinstance(tokens, list) else tokens for tokens in df['Feedback']]
    df['Processed Feedback'] = processed_feedbacks

    # Predict sentiment
    df['Predicted Sentiment'] = model.predict(vectorizer.transform(df['Processed Feedback']))

    # Calculate overall scores
    overall_positive_score = (df['Predicted Sentiment'] == 'positive').mean()
    overall_negative_score = (df['Predicted Sentiment'] == 'negative').mean()
    overall_neutral_score = (df['Predicted Sentiment'] == 'neutral').mean()

    print("Overall Positive Score:", overall_positive_score)
    print("Overall Negative Score:", overall_negative_score)
    print("Overall Neutral Score:", overall_neutral_score)

    # Create a DataFrame with Feedback and Predicted Sentiment
    # result_df = pd.DataFrame({
    #     'Feedback': df['Processed Feedback'],
    #     'Predicted Sentiment': df['Predicted Sentiment']
    # })

     # Display the head and tail of the DataFrame
    print("Head of DataFrame:")
    print(df[['Sentiment', 'Predicted Sentiment']].head().to_string(index=False))
    print("\nTail of DataFrame with new columns:")
    print(df[['Sentiment', 'Predicted Sentiment']].tail().to_string(index=False))

    # Update the database table with the predicted sentiments
    #update_table_with_predictions(conn, df, table_name)

    # Close the database connection
    close_connection(conn)

if __name__ == '__main__':
    main()