# Baseline Model Evaluation Report

## SVM Classifier

```
              precision    recall  f1-score   support

          IE       0.83      0.95      0.89       164
          IG       0.89      0.91      0.90       196
          PU       0.88      0.72      0.79        50
         RIE       0.85      0.55      0.67        31
         RIG       0.78      0.62      0.69        34
        RST1       1.00      1.00      1.00        33
        RST2       1.00      1.00      1.00        29

    accuracy                           0.88       537
   macro avg       0.89      0.82      0.85       537
weighted avg       0.88      0.88      0.87       537
```

## RandomForest Classifier

```
              precision    recall  f1-score   support

          IE       0.70      0.76      0.73       164
          IG       0.68      0.93      0.78       196
          PU       0.80      0.40      0.53        50
         RIE       0.00      0.00      0.00        31
         RIG       0.80      0.12      0.21        34
        RST1       0.97      1.00      0.99        33
        RST2       0.96      0.93      0.95        29

    accuracy                           0.73       537
   macro avg       0.70      0.59      0.60       537
weighted avg       0.70      0.73      0.68       537
```

## XGBoost Classifier

```
              precision    recall  f1-score   support

          IE       0.81      0.91      0.86       164
          IG       0.85      0.96      0.90       196
          PU       0.88      0.60      0.71        50
         RIE       0.75      0.19      0.31        31
         RIG       0.63      0.50      0.56        34
        RST1       1.00      1.00      1.00        33
        RST2       0.97      1.00      0.98        29

    accuracy                           0.84       537
   macro avg       0.84      0.74      0.76       537
weighted avg       0.84      0.84      0.83       537
```

