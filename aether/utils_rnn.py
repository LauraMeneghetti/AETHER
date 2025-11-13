import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

def collate_fun(batch):
    """
    Function preparing data to become input of the BiLSTM.
    
    :param torch.Tensor batch: List of (sequence, label) tuples
    :return: padded sequences, sorted labels, lengths of the sequences and idx
    :rtype: torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    """
    sequences, labels, seq_lengths = zip(*batch)  # Separate sequences and labels

    # Convert sequences to PyTorch tensors and sort by descending length
    seq_lengths = torch.tensor([len(seq) for seq in sequences])  # Original lengths
    sorted_indices = torch.argsort(seq_lengths, descending=True)  # Sort indices

    sequences_sorted = [torch.tensor(sequences[i], dtype=torch.float32) for i in sorted_indices]  # Sort sequences
    labels_sorted = torch.tensor([labels[i] for i in sorted_indices])  # Sort labels
    seq_lengths_sorted = seq_lengths[sorted_indices]  # Sort lengths

    # Pad sequences to max length in batch
    padded_sequences = pad_sequence(sequences_sorted, batch_first=True, padding_value=0)

    return padded_sequences, labels_sorted, seq_lengths_sorted, sorted_indices


def collate_fn_loo(batch):
    """
    Function preparing data to become input of the BiLSTM.
    
    :param torch.Tensor batch: List of (sequence, label) tuples
    :return: padded sequences, sorted labels, lengths of the sequences and idx
    :rtype: torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    """
    sequences, labels, seq_lengths, patient_ids = zip(*batch)  # Separate sequences and labels

    # Convert sequences to PyTorch tensors and sort by descending length
    seq_lengths = torch.tensor([len(seq) for seq in sequences])  # Original lengths
    sorted_indices = torch.argsort(seq_lengths, descending=True)  # Sort indices

    sequences_sorted = [torch.tensor(sequences[i], dtype=torch.float32) for i in sorted_indices]  # Sort sequences
    labels_sorted = torch.tensor([labels[i] for i in sorted_indices])  # Sort labels
    seq_lengths_sorted = seq_lengths[sorted_indices]  # Sort lengths

    # Pad sequences to max length in batch
    padded_sequences = pad_sequence(sequences_sorted, batch_first=True, padding_value=0)

    return padded_sequences, labels_sorted, seq_lengths_sorted, sorted_indices  # Return sorted and padded batch




def evaluate_model(model, data_loader, device):
    """
    Function performing the model's forward pass to gain the predictions,
    class probabilities and ordered real labels.

    :param nn.Module model: input model
    :param DataLoader data_loader: data loader we are considering for evaluation
    :param torch.device device: device

    :return: accurcay of the model, predictions, related class probabilities and 
             ordered labels
    :rtype: float, np.array, np.array, np.array
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for X_batch, y_batch, seq_len_batch, sorted_idx in data_loader:
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            X_batch = X_batch.unsqueeze(-1) # (batch, seq_len) -> (batch, seq_len, 1)
            
            outputs = model(X_batch, seq_len_batch)
            outs = torch.sigmoid(outputs).squeeze(-1)
            predictions = (outs > 0.5).long()

            # reorder idx to come back to th original order (useful to compare with ml models)
            original_idx_per_batch = torch.argsort(sorted_idx)

            predictions_original_order = predictions[original_idx_per_batch]
            probs_original_order = outs[original_idx_per_batch]
            labels_original_order = y_batch[original_idx_per_batch]

            all_preds.append(predictions_original_order.cpu())
            all_probs.append(probs_original_order.cpu())
            all_labels.append(labels_original_order.cpu())

    all_preds_np = torch.cat(all_preds, dim=0).numpy()
    all_labels_np = torch.cat(all_labels, dim=0).numpy()
    all_probs_np = torch.cat(all_probs, dim=0).numpy()
 
    # accuracy score
    acc = accuracy_score(all_labels_np, all_preds_np)

    return acc, all_preds_np, all_labels_np, all_probs_np


def find_best_threshold(y_true, y_probs, method="f1"):
    """
    Finds the best decision threshold for binary classification.
    
    :param np.array y_true: ground truth labels (0/1)
    :param np.array y_probs: predicted probabilities for class 1 (apnea)
    :param str method: Criteria to choose the best threshold ("f1", "pr", "youden")

    :return: Best threshold based on the selected method.
    :rtype: float
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
    
    if method == "f1":
        # Compute F1-score for all thresholds
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)  # Avoid division by zero
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Best threshold based on F1-score: {best_threshold:.4f}")

    elif method == "pr":
        # Compute best threshold where precision and recall are balanced
        best_threshold = thresholds[np.argmin(np.abs(precisions - recalls))]
        print(f"Best threshold where Precision ≈ Recall: {best_threshold:.4f}")

    elif method == "youden":
        # Youden’s J statistic = TPR - FPR (maximizes difference)
        youden_index = tpr - fpr
        best_threshold = roc_thresholds[np.argmax(youden_index)]
        print(f"Best threshold based on Youden’s J statistic: {best_threshold:.4f}")

    else:
        raise ValueError("Invalid method. Choose 'f1', 'pr', or 'youden'.")

    return best_threshold



def calculate_metrics(y_true, y_pred, y_probs):
    """
    Calculate several classification metrics.

    :param np.array y_true: ground truth labels (0/1)
    :param np.array y_pred: model predictions
    :param np.array y_probs: predicted probabilities for class 1 (apnea)

    :return: a dictionary with all the results obtained
    :rtype: dict(float)
    """

    # confusion matrix and derived metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (TPR)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0           # Positive Predictive Value (Precision)
    acc = (tp + tn) / len(y_true)
    f1 = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0

    try:
        roc_auc = roc_auc_score(y_true, y_probs)  # AUC-ROC
    except ValueError:
        roc_auc = "N/A (Only one class present)"
    
    try:
        pr_auc = average_precision_score(y_true, y_probs) # PR-AUC
    except ValueError:
        pr_auc = "N/A (Only  one class present)"

    print("\n" + "="*70)
    print("REPORT OF THE CLASSIFICATION METRICS")
    print("="*70)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, zero_division=0, target_names=['Non-Apnea', 'Apnea']))

    print("\n--- Confusion Matrix ---")
    print(cm)
    
    print("\n--- Derived Metrics from Confusion Matrix ---")
    print(f'Specificity (True Negative Rate): {specificity:.4f}')
    print(f'Negative Predictive Value (NPV): {npv:.4f}')
    print(f'Sensitivity (Recall/TPR): {sensitivity:.4f}')
    print(f'Precision (PPV): {ppv:.4f}')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    print("\n--- Area Under Curve (AUC) ---")
    print(f"AUC-ROC Score: {roc_auc:.4f}")
    print(f"Precision-Recall AUC Score: {pr_auc:.4f}")
    
    # ROC CURVE

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    
    if isinstance(roc_auc, float):
        plt.figure(figsize=(8, 6))
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label=f"ROC Curve (Area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate - FPR")
        plt.ylabel("True Positive Rate - TPR / Sensitivity")
        plt.title("ROC CURVE")
        plt.legend(loc="lower right")
        
        # Salvataggio e visualizzazione
        # Uso 'roc_curve_output.png' come nome del file generico
        plt.savefig('roc_curve_output.png') 
        plt.show()
        
    else:
        print("\nImpossible to generate ROC curve (data not valid).")

    print("="*70 + "\n")

    return {
        'accuracy': acc,
        'f1_score': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc_roc': roc_auc,
        'auc_pr': pr_auc,
        'confusion_matrix': cm.tolist(),
        'ppv': ppv,
        'npv': npv,
    }
