from preprocessing import preprocess_data
from train_model import train_and_save_model
from evaluate import evaluate_model

if __name__ == "__main__":
    print("Step 1: Preprocessing data...")
    X, y, vectorizer = preprocess_data("data/Movie_Review.csv")

    print("Step 2: Training model...")
    model, X_test, y_test = train_and_save_model(X, y, vectorizer)

    print("Step 3: Evaluating model...")
    evaluate_model(model, X_test, y_test)
