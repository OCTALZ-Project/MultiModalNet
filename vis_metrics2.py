import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns
from sklearn.preprocessing import label_binarize

# Ana fold dizini (bu script ile aynı dizinde çalıştırılırsa)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLD_ROOT = "FINETUNE/2025-07-20-043921_kfold_dataset_p_octa_b_oct"  # Gerekirse değiştirin

# Fold klasörlerini bul (sadece bir seviye altındaki klasörler)
fold_dirs = []
for d in os.listdir(FOLD_ROOT):
    full_path = os.path.join(FOLD_ROOT, d)
    if d.startswith("fold_") and os.path.isdir(full_path):
        fold_dirs.append(full_path)
fold_dirs = sorted(fold_dirs)

# Her fold için acc1 değerlerini topla
all_fold_accs = []
max_epochs = 0
for fold_path in fold_dirs:
    metrics_path = os.path.join(fold_path, "model", "epoch_metrics.txt")
    if not os.path.exists(metrics_path):
        continue
    val_accs = []
    with open(metrics_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "acc1:" in line:
            acc = float(line.strip().split(":")[1])
            val_accs.append(acc)
    if val_accs:
        all_fold_accs.append(val_accs)
        if len(val_accs) > max_epochs:
            max_epochs = len(val_accs)

# Kısa fold'ları pad et (eksik epoch varsa son değeriyle doldur)
for i in range(len(all_fold_accs)):
    if len(all_fold_accs[i]) < max_epochs:
        all_fold_accs[i] += [all_fold_accs[i][-1]] * (max_epochs - len(all_fold_accs[i]))

# Ortalamaları hesapla
all_fold_accs = np.array(all_fold_accs)
mean_accs = np.mean(all_fold_accs, axis=0)
epochs = list(range(len(mean_accs)))

# Maksimum accuracy ve epochunu bul
max_idx = int(np.argmax(mean_accs))
max_acc = mean_accs[max_idx]
print(f"Max mean accuracy: {max_acc:.2f} at epoch {max_idx}")

# Ortalama, maksimum ve std accuracy değerlerini 1 üzerinden yazdır
acc_mean = np.mean(mean_accs) / 100 if np.max(mean_accs) > 1.0 else np.mean(mean_accs)
acc_max = np.max(mean_accs) / 100 if np.max(mean_accs) > 1.0 else np.max(mean_accs)
acc_std = np.std(mean_accs) / 100 if np.max(mean_accs) > 1.0 else np.std(mean_accs)
print(f"Mean of mean accuracies across epochs (0-1): {acc_mean:.4f}")
print(f"Max of mean accuracies across epochs (0-1): {acc_max:.4f}")
print(f"Std of mean accuracies across epochs (0-1): {acc_std:.4f}")

# Grafik ve confusion matrix dosyalarını FOLD_ROOT içine kaydet
save_dir = FOLD_ROOT

# Ortalama accuracy grafiği
plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_accs, marker='o', color='b', label='Mean Validation Accuracy')
plt.scatter([max_idx], [max_acc], color='red', zorder=5, label=f'Max: {max_acc:.2f} (Epoch {max_idx})')
plt.xlabel("Epoch")
plt.ylabel("Mean Validation Accuracy (acc1)")
plt.title("Mean Validation Accuracy per Epoch Across Folds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "mean_validation_accuracy_per_epoch.png"))

# Her fold'un acc1 grafiği
plt.figure(figsize=(10, 6))
for i, val_accs in enumerate(all_fold_accs):
    plt.plot(epochs, val_accs, marker='o', label=f'Fold {i+1}')
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (acc1)")
plt.title("Validation Accuracy per Epoch for Each Fold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "validation_accuracy_per_fold.png"))

# Maksimum accuracy epochunda confusion matrix
all_cms = []
class_count = None
for fold_path in fold_dirs:
    metrics_path = os.path.join(fold_path, "model", "epoch_metrics.txt")
    if not os.path.exists(metrics_path):
        continue
    # O epochun true ve predicted label'larını oku
    with open(metrics_path, "r") as f:
        lines = f.readlines()
    epoch_counter = -1
    true_labels = None
    pred_labels = None
    for line in lines:
        m = re.match(r"Epoch (\d+)", line)
        if m:
            epoch_counter = int(m.group(1))
        if epoch_counter == max_idx:
            if line.startswith("True labels:"):
                true_labels = [int(x) for x in line.strip().split(":")[1].split(",") if x.strip()]
            if line.startswith("Predicted labels:"):
                pred_labels = [int(x) for x in line.strip().split(":")[1].split(",") if x.strip()]
        if true_labels is not None and pred_labels is not None:
            break
    if true_labels is not None and pred_labels is not None:
        cm = confusion_matrix(true_labels, pred_labels)
        all_cms.append(cm)
        if class_count is None:
            class_count = cm.shape[0]

if all_cms:
    # Ortalama confusion matrix (direkt sayıların toplamı)
    sum_cm = np.sum(np.stack([cm for cm in all_cms]), axis=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(sum_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Summed Confusion Matrix at Epoch {max_idx} (Max Mean Accuracy)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summed_confusion_matrix_max_epoch.png"))
    
    # Her fold için ayrı ayrı confusion matrix kaydet
    for i, cm in enumerate(all_cms):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix Fold {i+1} at Epoch {max_idx}")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"conf_matrix_fold_{i+1}.png"))
        plt.close()
else:
    print("Confusion matrix verisi bulunamadı!")

# ROC curve (proba ile, binary ve multi-class destekli)
num_classes = class_count if class_count is not None else 2
all_true = []
all_proba = []
for fold_path in fold_dirs:
    metrics_path = os.path.join(fold_path, "model", "epoch_metrics.txt")
    if not os.path.exists(metrics_path):
        continue
    with open(metrics_path, "r") as f:
        lines = f.readlines()
    # Fold içindeki en iyi epoch'u bul
    epoch_accs = []
    epoch_indices = []
    for idx, line in enumerate(lines):
        m = re.match(r"acc1:\s*([0-9.eE+-]+)", line)
        if m:
            acc = float(m.group(1))
            epoch_accs.append(acc)
            # Epoch indexini bulmak için bir önceki Epoch satırını bul
            for back in range(1, 10):
                if idx-back >= 0 and lines[idx-back].startswith("Epoch "):
                    epoch_indices.append(int(lines[idx-back].split()[1]))
                    break
    if not epoch_accs:
        continue
    best_idx = int(np.argmax(epoch_accs))
    best_epoch = epoch_indices[best_idx]
    # O epochun true, pred, proba'sını bul
    epoch_counter = -1
    true_labels = None
    proba_rows = []
    in_proba = False
    for line in lines:
        m = re.match(r"Epoch (\d+)", line)
        if m:
            if epoch_counter == best_epoch and in_proba:
                in_proba = False
            epoch_counter = int(m.group(1))
        if epoch_counter == best_epoch:
            if line.startswith("True labels:"):
                true_labels = [int(x) for x in line.strip().split(":")[1].split(",") if x.strip()]
            if line.startswith("Predicted probas"):
                in_proba = True
                continue
            if in_proba:
                if line.startswith("Epoch ") or line.startswith("-") or line.strip() == "":
                    in_proba = False
                else:
                    proba_row = [float(x) for x in line.strip().split(",") if x.strip()]
                    proba_rows.append(proba_row)
    if true_labels is not None and proba_rows:
        min_len = min(len(true_labels), len(proba_rows))
        all_true.extend(true_labels[:min_len])
        all_proba.extend(proba_rows[:min_len])

if all_true and all_proba:
    all_true = np.array(all_true)
    all_proba = np.array(all_proba)
    # Precision, Recall ve F1-score hesapla
    class_report = classification_report(all_true, np.argmax(all_proba, axis=1), output_dict=True)
    print("Classification Report:")
    metrics_to_print = ["precision", "recall", "f1-score"]
    for metric in metrics_to_print:
        cn_value = class_report.get("1", {}).get(metric, 0.0)  # CN için değer
        ad_value = class_report.get("0", {}).get(metric, 0.0)  # AD için değer
        weighted_value = class_report.get("weighted avg", {}).get(metric, 0.0)  # Weighted değer
        print(f"{metric.capitalize()} - CN: {cn_value:.2f}, AD: {ad_value:.2f}, Weighted: {weighted_value:.2f}")
    # ROC curve doğrudan tüm foldlardan toplanan (concatenate edilen) true label ve proba ile hesaplanır
    if num_classes == 2:
        # Binary: pozitif sınıfın proba'sı
        fpr, tpr, _ = roc_curve(all_true, all_proba[:,1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve at Epoch {max_idx} (Max Mean Accuracy)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curve_max_epoch.png"))
        
    else:
        # Multi-class: her sınıf için ROC
        all_true_bin = label_binarize(all_true, classes=list(range(num_classes)))
        plt.figure(figsize=(7, 6))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(all_true_bin[:,i], all_proba[:,i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (One-vs-Rest) at Epoch {max_idx}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curve_max_epoch.png"))
        
else:
    print("ROC için yeterli veri yok!")
