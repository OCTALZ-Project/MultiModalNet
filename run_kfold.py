import subprocess
import os
import re
import numpy as np
import shutil
import sys
import argparse
from datetime import datetime

def parse_accuracy_from_report(report_path):
    """Parses accuracy from the classification report file."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Example line: "Final validation results (Accuracy: 0.9876) for ..."
                # or "Accuracy: 0.9876" from --eval mode output
                match = re.search(r"Accuracy: ([\d.]+)", line)
                if match:
                    return float(match.group(1))
    except FileNotFoundError:
        print(f"Error: Report file not found at {report_path}")
    except Exception as e:
        print(f"Error parsing accuracy from {report_path}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--majority_vote', action='store_true', default=False, help='Run majority vote report after each fold and at the end')
    args, unknown = parser.parse_known_args()

    # Base command arguments (excluding fold-specific ones)
    base_executable = 'python' # or 'python3' depending on your environment
    script_to_run = 'main_finetune.py'
    
    # Base dataset path
    base_data_path = "/home/omerfarukaydin/Data/folds/kfold_dataset_for_3_years"

    # Automatically determine number of classes from the first fold's train directory
    def get_num_classes_from_fold(fold_dir):
        train_dir = os.path.join(fold_dir, 'train')
        if not os.path.isdir(train_dir):
            # Try without 'train' subdir (if structure is flat)
            train_dir = fold_dir
        if not os.path.isdir(train_dir):
            raise RuntimeError(f"Train directory not found: {train_dir}")
        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        return len(class_dirs)

    # Dynamically find all fold directories
    fold_dirs = [d for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d)) and d.startswith('fold_')]
    # Extract fold numbers and sort
    fold_tuples = []
    for d in fold_dirs:
        try:
            fold_num = int(re.findall(r'fold_(\d+)', d)[0])
            fold_tuples.append((fold_num, d))
        except Exception:
            continue
    fold_tuples.sort()  # sort by fold number
    if not fold_tuples:
        print(f"No fold directories found in {base_data_path}")
        return

    # Get number of classes from the first fold
    first_fold_path = os.path.join(base_data_path, fold_tuples[0][1])
    num_classes = get_num_classes_from_fold(first_fold_path)
    print(f"Detected number of classes: {num_classes}")

    base_args = [
        '--batch_size', '6',
        '--epochs', '100',
        '--blr', '5e-4',
        '--layer_decay', '0.75',
        '--weight_decay', '0.05',
        '--clip_grad', '1.0',
        '--warmup_epochs', '5',
        '--nb_classes', str(num_classes),
        '--bscan_model_global_pool', 'avg',
        '--num_workers', '4',
        '--pin_mem',
        '--projection_maps_model_dropout', '0.15',
        '--projection_maps_model', 'resnet101',
        '--bscan_model_dropout', '0.15',
        '--bscan_model_name', 'vit_large_patch16',
        '--loss_type', 'focal_loss',
        '--use_tensorboard'
    ]

    # Base output and log directory (added _py to distinguish from shell script outputs)
    now_str = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    base_output_log_prefix = f"FINETUNE/{now_str}_3_years_b_oct_resnet101_vitL"

    # Dynamically find all fold directories
    fold_dirs = [d for d in os.listdir(base_data_path) if os.path.isdir(os.path.join(base_data_path, d)) and d.startswith('fold_')]
    # Extract fold numbers and sort
    fold_tuples = []
    for d in fold_dirs:
        try:
            fold_num = int(re.findall(r'fold_(\d+)', d)[0])
            fold_tuples.append((fold_num, d))
        except Exception:
            continue
    fold_tuples.sort()  # sort by fold number
    if not fold_tuples:
        print(f"No fold directories found in {base_data_path}")
        return
    all_fold_accuracies = []
    fold_report_paths = {}
    fold_prediction_csvs = {}

    print(f"Starting k-fold cross-validation for {len(fold_tuples)} folds...")
    print(f"Base output directory: {base_output_log_prefix}\n")

    for fold_num, fold_dir_name in fold_tuples:
        print(f"----------------------------------------------------")
        print(f"Processing FOLD {fold_num}")
        print(f"----------------------------------------------------")

        current_data_path = os.path.join(base_data_path, fold_dir_name)
        current_output_dir = os.path.join(base_output_log_prefix, f"{fold_dir_name}", "model")
        current_log_dir = os.path.join(base_output_log_prefix, f"{fold_dir_name}", "logs")

        # Create output and log directories if they don't exist
        os.makedirs(current_output_dir, exist_ok=True)
        os.makedirs(current_log_dir, exist_ok=True)

        # Check if a checkpoint-best.pth exists for this fold to potentially resume or use for eval
        # For training, main_finetune.py handles loading 'checkpoint-best.pth' if --resume is appropriately managed or if it's the final eval stage.
        # If you want to *only* evaluate, you'd need to ensure 'checkpoint-best.pth' is in current_output_dir and add '--eval' and '--resume' current_output_dir/checkpoint-best.pth to command.
        # For now, assuming standard training and final evaluation as per main_finetune.py logic.

        command = [
            base_executable, script_to_run,
            '--data_path', current_data_path,
            '--output_dir', current_output_dir,
            '--log_dir', current_log_dir
        ] + base_args
        
        print(f"Executing command for FOLD {fold_num}:")
        print(' '.join(command))
        print("")

        try:
            # Run the command, show output in real-time
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
            
            for line in process.stdout:
                print(line, end='') # Print fold script output in real-time

            process.wait() # Wait for the subprocess to complete

            if process.returncode != 0:
                print(f"Error running command for FOLD {fold_num}. Return code: {process.returncode}")
                print(f"Skipping accuracy parsing for this fold.")
                continue # Skip to the next fold

            # After successful run, determine the report file path
            # main_finetune.py saves "final_val_classification_report.txt" after training
            # or "eval_classification_report.txt" if run with --eval
            report_filename = "final_val_classification_report.txt"
            report_file_path = os.path.join(current_output_dir, report_filename)
            fold_report_paths[fold_num] = report_file_path # Store path for later summary

            # Also track prediction CSVs
            pred_csv = os.path.join(current_output_dir, "final_validation_all_predictions.csv")
            if os.path.exists(pred_csv):
                fold_prediction_csvs[fold_num] = pred_csv

            if os.path.exists(report_file_path):
                accuracy = parse_accuracy_from_report(report_file_path)
                if accuracy is not None:
                    all_fold_accuracies.append(accuracy)
                    print(f"Fold {fold_num} reported accuracy: {accuracy:.4f}")
                else:
                    print(f"Could not parse accuracy from {report_file_path} for FOLD {fold_num}")
            else:
                print(f"Report file not found: {report_file_path} for FOLD {fold_num}")

        except Exception as e:
            print(f"An unexpected error occurred while processing FOLD {fold_num}: {e}")
            print(f"Skipping to next fold.")

        print("")
        print(f"----------------------------------------------------")
        print(f"Finished processing FOLD {fold_num}")
        print(f"----------------------------------------------------")
        print("")

    print("\n====================================================")
    print("K-Fold Cross-Validation Summary")
    print("====================================================\n")

    if all_fold_accuracies:
        mean_accuracy = np.mean(all_fold_accuracies)
        std_accuracy = np.std(all_fold_accuracies)
        print(f"Number of folds processed successfully for accuracy: {len(all_fold_accuracies)}/{len(fold_tuples)}")
        print("\nIndividual Fold Accuracies:")
        for i, (fold_num, _) in enumerate(fold_tuples):
            acc = None
            if fold_num in fold_report_paths:
                acc = parse_accuracy_from_report(fold_report_paths[fold_num])
            print(f"  Fold {fold_num}: {acc if acc is not None else 'N/A'} (Report: {fold_report_paths.get(fold_num, 'N/A')})")
        print(f"\nMean Accuracy across folds: {mean_accuracy:.4f}")
        print(f"Standard Deviation of Accuracy across folds: {std_accuracy:.4f}")
        consolidated_report_path = os.path.join(base_output_log_prefix, "consolidated_kfold_report.txt")
        with open(consolidated_report_path, 'w', encoding='utf-8') as outfile:
            outfile.write("K-Fold Cross-Validation Consolidated Report\n")
            outfile.write("=============================================\n\n")
            for fold_num, _ in fold_tuples:
                report_path = fold_report_paths.get(fold_num)
                if report_path and os.path.exists(report_path):
                    outfile.write(f"--- Fold {fold_num} Results (from: {report_path}) ---\n")
                    try:
                        with open(report_path, 'r', encoding='utf-8') as infile:
                            shutil.copyfileobj(infile, outfile)
                        outfile.write("\n\n")
                    except Exception as e:
                        outfile.write(f"Error reading report for fold {fold_num}: {e}\n\n")
                else:
                    outfile.write(f"--- Report for Fold {fold_num} not found or not processed ---\n\n")
            outfile.write("\n--- Overall Metrics ---\n")
            outfile.write(f"Mean Accuracy: {mean_accuracy:.4f}\n")
            outfile.write(f"Standard Deviation of Accuracy: {std_accuracy:.4f}\n")
            # Add prediction CSV summary
            outfile.write("\n--- Prediction CSVs per Fold ---\n")
            for fold_num, _ in fold_tuples:
                pred_csv = fold_prediction_csvs.get(fold_num)
                if pred_csv and os.path.exists(pred_csv):
                    outfile.write(f"Fold {fold_num}: {pred_csv}\n")
                else:
                    outfile.write(f"Fold {fold_num}: No prediction CSV found.\n")
        print(f"\nConsolidated report saved to: {consolidated_report_path}")
    else:
        print("\nNo accuracies were recorded. Overall results cannot be calculated.")
        print("Please check the output of each fold for errors.")
    print("\nAll k-fold finetuning runs completed.")

    # Her fold bittikten sonra majority vote raporu oluştur (opsiyonel)
    if args.majority_vote:
        mv_script = os.path.join(os.path.dirname(__file__), 'majority_vote_report.py')
        for fold_num, fold_dir_name in fold_tuples:
            fold_dir = os.path.join(base_output_log_prefix, fold_dir_name)
            print(f"\nRunning majority_vote_report.py for {fold_dir}")
            subprocess.run([
                sys.executable, mv_script,
                '--input_path', fold_dir
            ], check=True)
        # Tüm foldlar bitince genel majority vote raporu oluştur
        print(f"\nRunning majority_vote_report.py for all folds in {base_output_log_prefix}")
        subprocess.run([
            sys.executable, mv_script,
            '--input_path', base_output_log_prefix
        ], check=True)

if __name__ == '__main__':
    # Ensure that 'main_finetune.py' is in the same directory or in PATH,
    # and 'final_model.pth' is accessible.
    main()