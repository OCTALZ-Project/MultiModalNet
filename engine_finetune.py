import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    # Note: When using larger models like ViT-Large, consider using
    # gradient accumulation (--accum_iter > 1) to handle higher memory requirements

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (modalities_tuple, targets_cpu) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        octa_samples_dev = modalities_tuple[0].to(device, non_blocking=True)
        bscan_samples_dev = modalities_tuple[1].to(device, non_blocking=True)
        targets_dev = targets_cpu.to(device, non_blocking=True)

        current_octa_input = octa_samples_dev
        current_bscan_input = bscan_samples_dev
        current_targets = targets_dev

        if mixup_fn is not None:
            # Apply mixup to OCTA modality. B-Scan uses the primary sample's B-Scan.
            # Targets are mixed accordingly by mixup_fn.
            octa_mixed, targets_mixed = mixup_fn(octa_samples_dev, targets_dev)
            current_octa_input = octa_mixed
            # current_bscan_input remains bscan_samples_dev (from original primary sample)
            current_targets = targets_mixed
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True): # Ensure autocast is enabled
            outputs = model(current_octa_input, current_bscan_input)
            loss = criterion(outputs, current_targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        # loss_scaler should handle mixed precision if enabled.
        # The original NativeScaler handles parameters and update_grad.
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        
        min_lr = 10.
        max_lr = 0.
        # Check learning rate from param groups
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)


        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, num_classes, class_names=None, detailed_report=False, return_predictions=False): # detailed_report ve return_predictions eklendi
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    all_preds = []
    all_targets = []
    all_subject_ids = []

    for batch_idx, (modalities_tuple, target_cpu) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        octa_images_dev = modalities_tuple[0].to(device, non_blocking=True)
        bscan_images_dev = modalities_tuple[1].to(device, non_blocking=True)
        target_dev = target_cpu.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            output = model(octa_images_dev, bscan_images_dev)
            loss = criterion(output, target_dev)

        acc1, _ = accuracy(output, target_dev, topk=(1, 5))

        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target_dev.cpu().numpy())
        
        # Subject ID'leri topla
        if return_predictions:
            batch_size = target_cpu.shape[0]
            start_idx = batch_idx * data_loader.batch_size
            for i in range(batch_size):
                sample_idx = start_idx + i
                if sample_idx < len(data_loader.dataset.ids):
                    all_subject_ids.append(data_loader.dataset.ids[sample_idx])

        batch_size = octa_images_dev.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    # print(f"* Processed {len(all_targets)} samples for evaluation.") # Bu satırı detaylı raporda bırakabiliriz.

    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)
    overall_accuracy = accuracy_score(all_targets_np, all_preds_np)

    eval_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    eval_stats['overall_accuracy_sklearn'] = overall_accuracy

    if detailed_report:
        print(f"* Processed {len(all_targets)} samples for detailed evaluation.")
        print(f"* Overall Accuracy (sklearn): {overall_accuracy:.4f}")

        target_names_for_report = class_names
        if target_names_for_report is None:
            target_names_for_report = [str(i) for i in range(num_classes)]

        print("\nClassification Report:")
        report_str = classification_report(all_targets_np, all_preds_np, target_names=target_names_for_report, digits=4, zero_division=0)
        print(report_str)
        eval_stats['classification_report_str'] = report_str

        cm = confusion_matrix(all_targets_np, all_preds_np, labels=list(range(num_classes)))
        print("\nConfusion Matrix:")
        print("\n", cm)
        eval_stats['confusion_matrix'] = cm
    
    # Her zaman temel acc1 ve loss'u yazdır (veya sadece detailed_report=False ise)
    # Şimdilik metric_logger'dan gelenleri her zaman yazdıralım.
    print(f"* Acc@1 (from metric_logger) {eval_stats.get('acc1', 0):.3f} loss {eval_stats.get('loss', 0):.3f}")
    if detailed_report: # Sadece detaylı raporda sklearn acc'sini ayrıca belirtelim
        print(f"* Overall Accuracy (from sklearn) {eval_stats.get('overall_accuracy_sklearn', 0):.3f}")

    # Return predictions if requested
    if return_predictions:
        eval_stats['predictions'] = all_preds_np
        eval_stats['targets'] = all_targets_np
        eval_stats['subject_ids'] = all_subject_ids

    return eval_stats