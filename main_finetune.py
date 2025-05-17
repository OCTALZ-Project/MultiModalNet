import argparse
import datetime
import json
import numpy as np
import os
import time
import shutil
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import timm
assert timm.__version__ >= "0.3.2"
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from model_multimodal import MultiModalNet
from dataset_loader import OCTAMultiModalDataset

from engine_finetune import train_one_epoch, evaluate

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('Multi-modal fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--bscan_model_name', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of ViT model component to use (e.g., vit_base_patch16, vit_large_patch16)')
    parser.add_argument('--projection_maps_model', default='resnet50', type=str, choices=['resnet50', 'resnet101', 'alexnet'],
                        help='ResNet architecture to use for OCTA encoding (resnet50 or resnet101)')
    parser.add_argument('--projection_maps_model_dropout', type=float, default=0.0,
                        help='Dropout rate applied after ResNet feature extraction (default: 0.0, no dropout)')
    parser.add_argument('--projection_maps_model_finetune', default='', type=str,
                        help='Finetune ResNet from this checkpoint path. If empty, uses ImageNet weights.')
    parser.add_argument('--bscan_model_finetune', default='',
                        help='Finetune ViT from MAE checkpoint (or other ViT pretrain)')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--bscan_model_drop_path_rate', type=float, default=0.1, metavar='PCT',
                        help='ViT Drop path rate (default: 0.1)')
    parser.add_argument('--bscan_model_global_pool', type=str, default='avg',
                        choices=['', 'avg', 'avgmax', 'max', 'token', 'map'],
                        help="ViT global pooling type: 'avg' for average pooling, "
                             "'token' for CLS token, or 'false' to use CLS token output.")
    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay for ViT part (default: 0.75, 1.0 to disable)')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data_path', default='DATASET/', type=str,
                        help='dataset path (should point to the parent of train/val/test)')
    parser.add_argument('--nb_classes', default=3, type=int,
                        help='number of the classification types (e.g., 3 for NORMAL, AMD, DR)')

    parser.add_argument('--output_dir', default='./output_dir_multimodal',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_multimodal/logs',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    

    parser.add_argument('--save_args', type=str, nargs='+', 
                        default=['batch_size', 'epochs', 'bscan_model_name', 'bscan_model_finetune', 'projection_maps_model', 'projection_maps_model_dropout', 
                                'projection_maps_model_finetune', 'bscan_model_drop_path_rate', 'bscan_model_global_pool', 'clip_grad', 'bscan_model_drop_path_rate', 
                                'weight_decay', 'lr', 'blr', 'data_path', 'nb_classes', 'seed', 'resume', 'use_tensorboard'],
                        help='List of argument names to save to args.txt file (empty list to save all)')
    
    # Add TensorBoard logging control
    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help='Use TensorBoard for logging training progress')
    return parser

def plot_confusion_matrix(cm, class_names, output_path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)*0.9), max(6, len(class_names)*0.7)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print(f"Confusion matrix saved to {output_path}")
    except Exception as e:
        print(f"Error saving confusion matrix: {e}")
    plt.close(fig)

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(int(args.input_size / 0.875)),
        transforms.CenterCrop(args.input_size),
        normalize,
    ])

    train_data_path = os.path.join(args.data_path, 'train')
    val_data_path = os.path.join(args.data_path, 'val')
    test_data_path = os.path.join(args.data_path, 'test')

    dataset_train = OCTAMultiModalDataset(
        data_dir=train_data_path,
        num_classes=args.nb_classes,
        octa_transform=transform_train,
        bscan_transform=transform_train
    )
    dataset_val = OCTAMultiModalDataset(
        data_dir=val_data_path,
        num_classes=args.nb_classes,
        octa_transform=transform_val,
        bscan_transform=transform_val
    )
    dataset_test = OCTAMultiModalDataset(
        data_dir=test_data_path,
        num_classes=args.nb_classes,
        octa_transform=transform_val,
        bscan_transform=transform_val
    )

    class_names = None
    if hasattr(dataset_val, 'idx_to_class') and dataset_val.idx_to_class:
        try:
            class_names = [dataset_val.idx_to_class[i] for i in range(args.nb_classes)]
            if len(class_names) != args.nb_classes or any(name is None for name in class_names): # Check all names are derived
                 raise ValueError("Mismatch in derived class names.")
        except (KeyError, ValueError) as e:
            print(f"Warning: Could not derive all class names correctly from dataset.idx_to_class ({e}). Expected {args.nb_classes}.")
            class_names = [f"Class_{i}" for i in range(args.nb_classes)]
    else:
        print("Warning: dataset_val.idx_to_class not found or empty. Using generic class names for report.")
        class_names = [f"Class_{i}" for i in range(args.nb_classes)]
    print(f"Using class names for evaluation report: {class_names}")

    global_rank = 0
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    log_writer = None
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = None
        if args.use_tensorboard:
            log_writer = SummaryWriter(log_dir=args.log_dir)
            print(f"TensorBoard logging enabled, logs will be saved to {args.log_dir}")
        else:
            print("TensorBoard logging disabled")
        
        # Save selected arguments to a text file
        args_file = os.path.join(args.log_dir, "args.txt")
        with open(args_file, 'w') as f:
            # Add timestamp at the beginning
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Training started: {current_time}\n")
            
            # Add resume information if available
            if args.resume:
                f.write(f"Resumed from: {args.resume}\n")
            f.write("-" * 40 + "\n")
            
            # Check if save_args is empty (in which case save all arguments)
            if not args.save_args:
                for arg in sorted(vars(args)):
                    f.write(f"{arg}: {getattr(args, arg)}\n")
            else:
                # Save only the selected arguments
                for arg in sorted(args.save_args):
                    if hasattr(args, arg):
                        f.write(f"{arg}: {getattr(args, arg)}\n")
                    else:
                        f.write(f"{arg}: NOT_FOUND\n")
        print(f"Saved selected arguments to {args_file}")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # bscan_model_gp_value = args.bscan_model_global_pool
    # if isinstance(args.bscan_model_global_pool, str) and args.bscan_model_global_pool.lower() == 'false':
    #     bscan_model_gp_value = False
    # elif isinstance(args.bscan_model_global_pool, str) and args.bscan_model_global_pool.lower() == 'avg':
    #     bscan_model_gp_value = 'avg' # Keep as string for models_vit if it expects 'avg'
    # elif isinstance(args.bscan_model_global_pool, str) and args.bscan_model_global_pool.lower() == 'token':
    #     bscan_model_gp_value = True # MAE models_vit uses True for token, False for no token/head.
    #                                 # This needs to align with how your MultiModalNet and models_vit.py handle it.
    #                                 # Assuming 'token' means using the CLS token, which is default for MAE's vit global_pool=True.
    
    model = MultiModalNet(
        num_classes=args.nb_classes,
        projection_maps_model_finetune_path=args.projection_maps_model_finetune,
        bscan_model_finetune_path=args.bscan_model_finetune,
        bscan_model_name=args.bscan_model_name,
        bscan_model_global_pool=args.bscan_model_global_pool,
        bscan_model_drop_path_rate=args.bscan_model_drop_path_rate,
        bscan_model_input_size=args.input_size,
        projection_maps_model=args.projection_maps_model,
        projection_maps_model_dropout=args.projection_maps_model_dropout
    )
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('Number of trainable params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: %.2e" % (args.lr * 256 / eff_batch_size if eff_batch_size > 0 else args.lr))
    print("Actual lr: %.2e" % args.lr)
    print("Accumulate grad iterations: %d" % args.accum_iter)
    print("Effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    param_groups = model_without_ddp.get_param_groups(
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        layer_decay_rate=args.layer_decay if args.layer_decay < 1.0 else None
    )
    optimizer = torch.optim.AdamW(param_groups)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print("Criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        if args.resume:
            misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=None, loss_scaler=None)
        else:
             print("Running evaluation. Weights assumed to be loaded from --projection_maps_model_finetune or --bscan_model_finetune if specified.")

        print("\n" + "="*30 + " Evaluation Mode " + "="*30)
        test_stats = evaluate(data_loader_val, model, device, 
                              num_classes=args.nb_classes, class_names=class_names, 
                              detailed_report=True) # <<< DETAILED REPORT TRUE
        
        # Accuracy yazdırma (acc1 veya sklearn'den gelen)
        eval_acc = test_stats.get('overall_accuracy_sklearn', test_stats.get('acc1', 0))
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {eval_acc * 100:.2f}%") # Yüzde olarak
        
        if args.output_dir and global_rank == 0:
            cm_to_plot = test_stats.get('confusion_matrix')
            if cm_to_plot is not None:
                cm_filename = os.path.join(args.output_dir, "eval_confusion_matrix.png")
                plot_confusion_matrix(cm_to_plot, class_names, cm_filename)
            
            report_str = test_stats.get('classification_report_str')
            if report_str:
                report_filename = os.path.join(args.output_dir, "eval_classification_report.txt")
                with open(report_filename, "w") as f:
                    f.write(f"Evaluation results for {args.data_path} (validation set)\n")
                    f.write(f"Accuracy: {eval_acc:.4f}\n")
                    f.write("="*50 + "\n")
                    f.write(report_str)
                print(f"Classification report saved to {report_filename}")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_epoch_stats = {} # En iyi epoch'un istatistiklerini saklamak için
    
    # Initialize tracking for best 2 models
    best1_accuracy = 0.0
    best1_epoch = -1
    best2_accuracy = 0.0
    best2_epoch = -1

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        
        # Eğitim sırasında sadece temel metrikler için evaluate çağır
        val_stats = evaluate(data_loader_val, model, device,
                             num_classes=args.nb_classes, class_names=class_names, 
                             detailed_report=False) # <<< DETAILED REPORT FALSE
        
        current_accuracy_val = val_stats.get('overall_accuracy_sklearn', val_stats.get('acc1', 0))
        print(f"Validation Accuracy (epoch {epoch}): {current_accuracy_val * 100:.2f}% (Loss: {val_stats.get('loss', 0):.4f})")
        
        if args.output_dir and global_rank == 0:
            # Always save as last model
            last_model_path = os.path.join(args.output_dir, "checkpoint-last.pth")
            
            # Check if current model is better than best1
            if current_accuracy_val > best1_accuracy:
                # Save current model as best
                best_model_path = os.path.join(args.output_dir, "checkpoint-best.pth")
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, 
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch="best")
                
                # Update best tracking
                best1_accuracy = current_accuracy_val
                best1_epoch = epoch
                best_epoch_stats = val_stats
                print(f"New best validation accuracy: {best1_accuracy*100:.2f}%. Model saved as checkpoint-best.pth")
            
            # Optionally also save the current model for resuming later if needed
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, 
                optimizer=optimizer, loss_scaler=loss_scaler, epoch="last")
            # But immediately delete the "last" checkpoint if we don't need it
            if os.path.exists(last_model_path) and os.path.exists(os.path.join(args.output_dir, "checkpoint-best.pth")):
                os.remove(last_model_path)

        # Update max_accuracy to reflect best1
        max_accuracy = best1_accuracy
        print(f'Best validation accuracy so far: {best1_accuracy*100:.2f}% (epoch {best1_epoch})')

        if log_writer is not None and args.use_tensorboard:
            log_writer.add_scalar('perf/val_acc1', current_accuracy_val, epoch)
            log_writer.add_scalar('perf/val_loss', val_stats.get('loss',0), epoch)
            # train_loss'u da loglayalım
            log_writer.add_scalar('perf/train_loss', train_stats.get('loss',0), epoch)

        # log_stats için JSON uyumlu olanları alalım
        val_stats_for_log = {k: v for k, v in val_stats.items() if not isinstance(v, np.ndarray) and k != 'classification_report_str'}
        log_stats_dict = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats_for_log.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and global_rank == 0:
            if log_writer is not None and args.use_tensorboard:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_dict) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # Eğitimin sonunda Final Evaluation (detaylı rapor ile)
    print("\n" + "="*30 + " Final Evaluation on Validation Set " + "="*30)
    if args.output_dir and global_rank == 0:
        best_model_path = os.path.join(args.output_dir, "checkpoint-best.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} for final evaluation.")
            checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
            state_dict_to_load = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            new_state_dict = {}
            for k, v in state_dict_to_load.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model_without_ddp.load_state_dict(new_state_dict)
        else:
            print("Best model checkpoint not found. Evaluating with the last model state.")
    
    if args.distributed:
        torch.distributed.barrier()

    # concat test set and val set (dataset_val + dataset_test)
    dataset_final_evaluation = torch.utils.data.ConcatDataset([dataset_val, dataset_test])
    sampler_final_evaluation = torch.utils.data.SequentialSampler(dataset_final_evaluation)
    data_loader_final_evaluation = torch.utils.data.DataLoader(
        dataset_final_evaluation, sampler=sampler_final_evaluation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    final_test_stats = evaluate(data_loader_final_evaluation, model, device, 
                                num_classes=args.nb_classes, class_names=class_names, 
                                detailed_report=True) # <<< DETAILED REPORT TRUE
    
    final_acc = final_test_stats.get('overall_accuracy_sklearn', final_test_stats.get('acc1', 0))
    print(f"Final accuracy of the (best/last) model on the {len(dataset_final_evaluation)} validation images: {final_acc * 100:.2f}%")
    
    if args.output_dir and global_rank == 0:
        cm_to_plot = final_test_stats.get('confusion_matrix')
        if cm_to_plot is not None:
            cm_filename = os.path.join(args.output_dir, "final_val_confusion_matrix.png")
            plot_confusion_matrix(cm_to_plot, class_names, cm_filename)
        
        report_str = final_test_stats.get('classification_report_str')
        if report_str:
            report_filename = os.path.join(args.output_dir, "final_val_classification_report.txt")
            with open(report_filename, "w") as f:
                f.write(f"Final validation results (Accuracy: {final_acc:.4f}) for {args.data_path}\n")
                f.write("="*50 + "\n")
                f.write(report_str)
            print(f"Final validation classification report saved to {report_filename}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)