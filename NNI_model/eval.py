from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np


model.eval()


true_labels = []
pred_probs = []  

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = opt_model(inputs)
        probs = torch.softmax(logits, dim=1)  

        true_labels.extend(labels.tolist())
        pred_probs.extend(probs.tolist())

true_labels = np.array(true_labels)
pred_probs = np.array(pred_probs)

acc = accuracy_score(true_labels, np.argmax(pred_probs, axis=1))
f1 = f1_score(true_labels, np.argmax(pred_probs, axis=1), average='macro')
auc = roc_auc_score(true_labels, pred_probs, multi_class='ovo')

print(f'Test accuracy: {acc:.4f}')
print(f'Test F1 score: {f1:.4f}')
print(f'Test AUC score: {auc:.4f}')

