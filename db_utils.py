import mysql.connector
import pandas as pd
import configparser
import csv

def load_db_config(filename='../config.ini'):
    """Load database configuration from an INI file."""
    config = configparser.ConfigParser()
    config.read(filename)
    return config['mysql']

def connect_to_database(config):
    """Establish a connection to the MySQL database using the provided configuration."""
    conn = mysql.connector.connect(**config)
    if conn.is_connected():
        print("Connection established successfully!")
    return conn

def fetch_data_from_table(conn, table_name):
    """Fetch all data from a specified table and return as a pandas DataFrame."""
    query = f"SELECT * FROM {table_name};"
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    cursor.close()
    return df

def close_connection(conn):
    """Close the database connection."""
    if conn.is_connected():
        conn.close()
        print("Connection closed successfully!")

def import_csv_to_table(conn, csv_file, table_name):
    """Import data from a CSV file into a specified MySQL table."""
    cursor = conn.cursor()
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            cursor.execute(
                f"INSERT INTO {table_name} (Subcategory, Feedback, Criticality, Length, Sentiment, Label) VALUES (%s, %s, %s, %s, %s, %s)",
                row
            )
    conn.commit()
    cursor.close()
    print("Data imported successfully!")

def fetch_sentiment_by_label_and_subcategory(conn, table_name, label, subcategory):
    """Fetch data based on label and subcategory and return as a pandas DataFrame."""
    query = f"""
    SELECT feedback, sentiment, label, subcategory 
    FROM {table_name} 
    WHERE label = %s AND subcategory = %s;
    """
    cursor = conn.cursor()
    cursor.execute(query, (label, subcategory))
    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    cursor.close()
    return df

def update_table_with_predictions(conn, df, table_name):
    """Update the table with predicted sentiments."""
    cursor = conn.cursor()
    update_query = f"UPDATE {table_name} SET PredictedSentiment = %s WHERE Feedback = %s"

    for i, row in df.iterrows():
        cursor.execute(update_query, (row['Predicted Sentiment'], ' '.join(row['Feedback'])))

    conn.commit()
    cursor.close()
    print("Table updated with predicted sentiments successfully!")