# Simple prediction demo for the trained digits model
import joblib
from sklearn.datasets import load_digits
import numpy as np

def main():
    clf = joblib.load('model.joblib')
    digits = load_digits()
    X = digits.data
    y = digits.target
    sample = X[:10]
    preds = clf.predict(sample)
    print('Predictions for first 10 samples:', preds)
    print('Ground truth:', y[:10])

if __name__ == '__main__':
    main()
