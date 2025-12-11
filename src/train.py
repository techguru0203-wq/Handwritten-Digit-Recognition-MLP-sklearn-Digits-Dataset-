# Train script for handwritten digit recognition using sklearn's digits dataset
import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def main():
    digits = load_digits()
    X = digits.data  # (n_samples, 64)
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(clf, 'model.joblib')
    print('Saved model to model.joblib')

if __name__ == '__main__':
    main()
