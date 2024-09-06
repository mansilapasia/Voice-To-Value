import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def train_models(X_train, y_train):
    """Train multiple models and return them along with their names."""
    models = [
        BernoulliNB(),
        RandomForestClassifier(),
        SVC(),
        LogisticRegression(max_iter=1000)
    ]
    
    trained_models = []
    for model in models:
        model.fit(X_train, y_train)
        trained_models.append(model)
    
    model_names = ['BernoulliNB', 'RandomForestClassifier', 'SVC', 'LogisticRegression']
    return trained_models, model_names

def evaluate_models(models, model_names, X_test, y_test):
    """Evaluate models and return a DataFrame with their performance metrics."""
    accuracy = []
    precision = []
    recall = []
    f1_score = []

    for model in models:
        y_pred = model.predict(X_test)
        accuracy.append(model.score(X_test, y_test))
        report = classification_report(y_test, y_pred, output_dict=True)
        precision.append(report['macro avg']['precision'])
        recall.append(report['macro avg']['recall'])
        f1_score.append(report['macro avg']['f1-score'])

        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix for {type(model).__name__}")
        print(cm)
        print()

    results = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    })

    return results

def save_model_and_vectorizer(model, vectorizer, model_path='ml_model.pkl', vectorizer_path='tfidf.pkl'):
    """Save the trained model and the vectorizer to disk."""
    with open(model_path, 'wb') as model_file, open(vectorizer_path, 'wb') as vectorizer_file:
        pickle.dump(model, model_file)
        pickle.dump(vectorizer, vectorizer_file)

def load_model_and_vectorizer(model_path='ml_model.pkl', vectorizer_path='tfidf.pkl'):
    """Load the trained model and the vectorizer from disk."""
    with open(model_path, 'rb') as model_file, open(vectorizer_path, 'rb') as vectorizer_file:
        model = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer