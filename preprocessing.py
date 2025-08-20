import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords (only needed first time)
nltk.download('stopwords')

def clean_review(review):
    return ' '.join(word for word in review.split() if word not in set(stopwords.words('english')))

def preprocess_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # Clean reviews
    data['text'] = data['text'].astype(str).apply(clean_review)

    # Replace labels: pos -> 1, neg -> 0
    data['sentiment'] = data['sentiment'].replace(['pos', 'neg'], [1, 0])

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=2500)
    X = vectorizer.fit_transform(data['text']).toarray()
    y = data['sentiment']

    return X, y, vectorizer
