#Evaluation of the existing project dataset

## Protocol
- Original rows: 5,000
- Unique review texts: 3,961
- Duplicate rows removed: 1,039
- Deduplicated rows: 3,961
- Stratified train/test split: 80/20, random_state=42
- Same TF-IDF representation and Logistic Regression classifier for both models
- Combined model adds all 22 numeric features defined in project features.py
- Required metrics: accuracy, precision, recall, macro-F1, MCC
- Additional plots: ROC, precision–recall, label distribution, rating distribution, review length
- Confusion matrices, predictions CSV, and audit CSV included

## Results
                             model  accuracy  precision  recall  macro_f1  mcc  roc_auc_auxiliary  average_precision_auxiliary
                         Text only       1.0        1.0     1.0       1.0  1.0                1.0                          1.0
Text + project engineered features       1.0        1.0     1.0       1.0  1.0                1.0                          1.0

## Interpretation
This is a more careful evaluation of the existing synthetic dataset.It is not independent validation on real customer reviews. Labels: 0=genuine, 1=fake.
