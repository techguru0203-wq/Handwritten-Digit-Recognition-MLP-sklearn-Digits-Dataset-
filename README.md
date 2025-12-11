# Handwritten Digit Recognition (sklearn digits)

This project demonstrates handwritten digit classification using the classic `digits` dataset from scikit-learn.  
An MLP neural network is trained to recognize digits (0–9) from 8×8 pixel grayscale images.

---

## 🚀 Features
- Uses sklearn’s built-in digits dataset  
- MLPClassifier with two hidden layers  
- Training script (`train.py`) that saves `model.joblib`  
- Prediction script (`predict.py`) to test model on sample digits  
- Easy to run, no external dataset required  

---

## 📂 Project Structure

```md
├── src/
│ ├── train.py
│ └── predict.py
├── requirements.txt
└── README.md
```


---

## 🔧 Installation
```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```
## 🧠 Train Model
```bash
python src/train.py
```

## 🔍 Run Predictions
```bash
python src/predict.py
```

