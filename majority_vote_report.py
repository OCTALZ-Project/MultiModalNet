import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def run_majority_vote_report_v2(input_path, out_report_path=None):
    """
    input_path: Tek bir fold klasörü (içinde model/ ve logs/ var) veya birden fazla fold içeren ana klasör olabilir.
    out_report_path: Raporun kaydedileceği yol. None ise input_path içine kaydeder.
    """
    def process_fold(fold_dir, report):
        pred_path = os.path.join(fold_dir, "model", "final_validation_all_predictions.csv")
        if not os.path.exists(pred_path):
            report.write(f"{fold_dir}: Prediction file not found!\n\n")
            return None
        df = pd.read_csv(pred_path)
        def extract_item_id(subject_id):
            return str(subject_id).split('_')[0]
        df['item_id'] = df['subject_id'].apply(extract_item_id)
        item_groups = df.groupby('item_id')
        majority_vote_rows = []
        for item_id, group in item_groups:
            pred_label = Counter(group["predicted_label"]).most_common(1)[0][0]
            true_label = group["true_label"].iloc[0]
            majority_vote_rows.append({
                "item_id": item_id,
                "true_label": true_label,
                "predicted_label": pred_label
            })
        mv_df = pd.DataFrame(majority_vote_rows)
        mv_pred_path = os.path.join(fold_dir, "model", "majority_vote_predictions_by_item.csv")
        mv_df.to_csv(mv_pred_path, index=False)
        labels = sorted(list(set(mv_df["true_label"]) | set(mv_df["predicted_label"])))
        report.write(f"--- {os.path.basename(fold_dir)} Majority Vote Results (item_id bazında) ---\n")
        report.write(classification_report(mv_df["true_label"], mv_df["predicted_label"], labels=labels))
        report.write("\n")
        cm = confusion_matrix(mv_df["true_label"], mv_df["predicted_label"], labels=labels)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{os.path.basename(fold_dir)} Majority Vote Confusion Matrix (item_id bazında)")
        cm_path = os.path.join(fold_dir, "model", "majority_vote_confusion_matrix_by_item.png")
        plt.savefig(cm_path)
        plt.close()
        report.write(f"Confusion matrix saved to: {cm_path}\n")
        report.write(f"Majority vote predictions saved to: {mv_pred_path}\n\n")
        return mv_df

    # Tek fold mu yoksa ana klasör mü?
    if os.path.isdir(os.path.join(input_path, "model")):
        # Tek fold
        folds = [input_path]
        if out_report_path is None:
            out_report_path = os.path.join(input_path, "majority_vote_report.txt")
    else:
        # Ana klasör, fold_x klasörlerini bul
        folds = [os.path.join(input_path, d) for d in os.listdir(input_path)
                 if os.path.isdir(os.path.join(input_path, d)) and d.startswith("fold_")]
        folds.sort()
        if out_report_path is None:
            out_report_path = os.path.join(input_path, "majority_vote_report.txt")

    all_results = []
    # Her fold için kendi raporunu çıkar
    for fold_dir in folds:
        fold_report_path = os.path.join(fold_dir, "majority_vote_report.txt") if len(folds) > 1 else out_report_path
        with open(fold_report_path, "w") as report:
            mv_df = process_fold(fold_dir, report)
            if mv_df is not None:
                mv_df['fold'] = os.path.basename(fold_dir)
                all_results.append(mv_df)
    # Genel rapor (tüm foldlar birleştirilerek)
    if len(all_results) > 1:
        with open(out_report_path, "w") as report:
            report.write("\n====================\nGENEL RAPOR (Tüm Foldlar Birleşik)\n====================\n")
            all_df = pd.concat(all_results, ignore_index=True)
            labels = sorted(list(set(all_df["true_label"]) | set(all_df["predicted_label"])))
            report.write(classification_report(all_df["true_label"], all_df["predicted_label"], labels=labels))
            report.write("\n")
            cm = confusion_matrix(all_df["true_label"], all_df["predicted_label"], labels=labels)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("GENEL Majority Vote Confusion Matrix (item_id bazında)")
            cm_path = os.path.join(os.path.dirname(out_report_path), "majority_vote_confusion_matrix_by_item_ALL.png")
            plt.savefig(cm_path)
            plt.close()
            report.write(f"GENEL confusion matrix saved to: {cm_path}\n")
    print(f"Majority vote raporu ve confusion matrixler tamamlandı.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='K-fold veya tek fold majority vote raporu ve confusion matrix oluşturucu')
    parser.add_argument('--input_path', type=str, required=True, help='Fold klasörü veya ana klasör (fold_x dizinlerinin bulunduğu yer)')
    parser.add_argument('--out_report_path', type=str, default=None, help='Çıktı rapor dosya yolu')
    args = parser.parse_args()
    run_majority_vote_report_v2(args.input_path, args.out_report_path)
