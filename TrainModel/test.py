from sklearn.metrics import confusion_matrix, log_loss, classification_report

y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 1, 0, 1, 1, 1]
y_pred_proba = [0.4, 0.8, 0.3, 0.9, 0.7, 0.8]

cm = confusion_matrix(y_true, y_pred)
loss = log_loss(y_true, y_pred_proba)
cls_report = classification_report(y_true, y_pred)
print(cls_report)
