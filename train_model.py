from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

def train_and_save_model(X, y, vectorizer, model_file="model.pkl", vectorizer_file="vectorizer.pkl"):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save model & vectorizer
    pickle.dump(model, open(model_file, 'wb'))
    pickle.dump(vectorizer, open(vectorizer_file, 'wb'))

    return model, X_test, y_test
