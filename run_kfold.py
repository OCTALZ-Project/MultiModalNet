import subprocess
import os
import re
import numpy as np
import shutil
import sys

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
    # Base command arguments (excluding fold-specific ones)
    base_executable = 'python' # or 'python3' depending on your environment
    script_to_run = 'main_finetune.py'
    
    base_args = [
        '--batch_size', '32',
        '--epochs', '100',
        '--blr', '5e-4',
        '--layer_decay', '0.75',
        '--weight_decay', '0.05',
        '--clip_grad', '1.0',
        '--warmup_epochs', '5',
        '--nb_classes', '2',
        '--bscan_model_global_pool', 'avg',
        '--num_workers', '4',
        '--pin_mem',
        '--projection_maps_model_dropout', '0.15',
        '--projection_maps_model', 'resnet152',
        '--bscan_model_dropout', '0.15',
        '--bscan_model_name', 'resnet152',
        '--use_tensorboard'
    ]

    # Base dataset path
    base_data_path = "/home/omerfarukaydin/Desktop/ad-reduced-cn-5-fold-dataset"

    # Base output and log directory (added _py to distinguish from shell script outputs)
    base_output_log_prefix = "FINETUNE/2025-06-02-resnet152-both-ad-cn-OCT-augmented-both"

    # Number of folds
    num_folds = 5  # Set to 5 for your kfold_dataset structure

    all_fold_accuracies = []
    fold_report_paths = {}
    fold_prediction_csvs = {}

    print(f"Starting k-fold cross-validation for {num_folds} folds...")
    print(f"Base output directory: {base_output_log_prefix}\n")

    for i in range(1, num_folds+1):
        fold_num = i
        print(f"----------------------------------------------------")
        print(f"Processing FOLD {fold_num}")
        print(f"----------------------------------------------------")

        current_data_path = os.path.join(base_data_path, f"fold_{fold_num}")
        # Output structure: ./13_05_2025_avg_pool_py/fold_1/model
        current_output_dir = os.path.join(base_output_log_prefix, f"fold_{fold_num}", "model")
        # Log structure: ./13_05_2025_avg_pool_py/fold_1/logs
        current_log_dir = os.path.join(base_output_log_prefix, f"fold_{fold_num}", "logs")

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
        print(f"Number of folds processed successfully for accuracy: {len(all_fold_accuracies)}/{num_folds}")
        print("\nIndividual Fold Accuracies:")
        for i, acc in enumerate(all_fold_accuracies):
            print(f"  Fold {i+1}: {acc:.4f} (Report: {fold_report_paths.get(i+1, 'N/A')})")
        print(f"\nMean Accuracy across folds: {mean_accuracy:.4f}")
        print(f"Standard Deviation of Accuracy across folds: {std_accuracy:.4f}")
        consolidated_report_path = os.path.join(base_output_log_prefix, "consolidated_kfold_report.txt")
        with open(consolidated_report_path, 'w', encoding='utf-8') as outfile:
            outfile.write("K-Fold Cross-Validation Consolidated Report\n")
            outfile.write("=============================================\n\n")
            for fold_idx in range(0, num_folds):
                report_path = fold_report_paths.get(fold_idx)
                if report_path and os.path.exists(report_path):
                    outfile.write(f"--- Fold {fold_idx} Results (from: {report_path}) ---\n")
                    try:
                        with open(report_path, 'r', encoding='utf-8') as infile:
                            shutil.copyfileobj(infile, outfile)
                        outfile.write("\n\n")
                    except Exception as e:
                        outfile.write(f"Error reading report for fold {fold_idx}: {e}\n\n")
                else:
                    outfile.write(f"--- Report for Fold {fold_idx} not found or not processed ---\n\n")
            outfile.write("\n--- Overall Metrics ---\n")
            outfile.write(f"Mean Accuracy: {mean_accuracy:.4f}\n")
            outfile.write(f"Standard Deviation of Accuracy: {std_accuracy:.4f}\n")
            # Add prediction CSV summary
            outfile.write("\n--- Prediction CSVs per Fold ---\n")
            for fold_idx in range(0, num_folds):
                pred_csv = fold_prediction_csvs.get(fold_idx)
                if pred_csv and os.path.exists(pred_csv):
                    outfile.write(f"Fold {fold_idx}: {pred_csv}\n")
                else:
                    outfile.write(f"Fold {fold_idx}: No prediction CSV found.\n")
        print(f"\nConsolidated report saved to: {consolidated_report_path}")
    else:
        print("\nNo accuracies were recorded. Overall results cannot be calculated.")
        print("Please check the output of each fold for errors.")
    print("\nAll k-fold finetuning runs completed.")

    # Her fold bittikten sonra majority vote raporu oluştur
    mv_script = os.path.join(os.path.dirname(__file__), 'majority_vote_report.py')
    for i in range(1, num_folds+1):
        fold_dir = os.path.join(base_output_log_prefix, f"fold_{i}")
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