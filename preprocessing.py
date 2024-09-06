import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def text_preprocessing(text):
    """Clean and preprocess text data."""
    url_pattern = r"(http://[^ ]*|https://[^ ]*|www\.[^ ]*)"
    user_pattern = r"@[^\s]+"
    entity_pattern = r"&[^ ]+;"
    neg_contraction = r"n't\b"
    non_alpha = r"[^a-z]"
    punctuation_pattern = r"[^A-Za-z0-9]"
    whitespace_pattern = r"[ +]"

    cleaned_text = text.lower()
    cleaned_text = re.sub(neg_contraction, " not", cleaned_text)
    cleaned_text = re.sub(url_pattern, " ", cleaned_text)
    cleaned_text = re.sub(user_pattern, " ", cleaned_text)
    cleaned_text = re.sub(entity_pattern, " ", cleaned_text)
    cleaned_text = re.sub(punctuation_pattern, " ", cleaned_text)
    cleaned_text = re.sub(whitespace_pattern, " ", cleaned_text)
    cleaned_text = re.sub(non_alpha, " ", cleaned_text)

    tokens = word_tokenize(cleaned_text)
    lemmatized_tokens = [lemmatizer.lemmatize(word, 'v') for word in tokens if word not in stop_words]

    return lemmatized_tokens