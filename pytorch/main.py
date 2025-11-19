#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, TomatoDataset, TomatoInferenceDataset
from model import PointNet, DGCNN, DGCNNSeg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from tqdm import tqdm


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def _create_confusion(num_classes):
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def _update_confusion(confusion, preds, labels, num_classes, ignore_label):
    preds = preds.detach().cpu().view(-1)
    labels = labels.detach().cpu().view(-1)
    if ignore_label is not None:
        mask = labels != ignore_label
        preds = preds[mask]
        labels = labels[mask]
    if preds.numel() == 0:
        return confusion
    hist = torch.bincount(labels * num_classes + preds, minlength=num_classes ** 2)
    confusion += hist.cpu().numpy().reshape(num_classes, num_classes)
    return confusion


def _confusion_metrics(confusion):
    total = confusion.sum()
    overall_acc = float(np.trace(confusion) / total) if total > 0 else 0.0
    intersection = np.diag(confusion)
    gt = confusion.sum(axis=1)
    pred = confusion.sum(axis=0)
    union = gt + pred - intersection
    iou = intersection.astype(np.float64) / np.maximum(union, 1)
    iou[union == 0] = np.nan
    miou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
    return overall_acc, miou, iou


def run_segmentation_epoch(model, loader, device, criterion, num_classes, ignore_label, optimizer=None):
    is_train = optimizer is not None
    model.train(mode=is_train)
    confusion = _create_confusion(num_classes)
    total_loss = 0.0
    count = 0.0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        pbar = tqdm(loader, desc='Training' if is_train else 'Validation')
        for data, label in pbar:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            data = data.float().to(device)                        # (B, N, 3)
            label = label.long().to(device)                      # (B, N)
            data = data.permute(0, 2, 1)
            logits = model(data)
            loss = criterion(logits, label)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            preds = logits.max(dim=1)[1]
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size
            confusion = _update_confusion(confusion, preds, label, num_classes, ignore_label)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'batch_size': batch_size
            })
    avg_loss = total_loss / max(count, 1)
    overall_acc, miou, iou = _confusion_metrics(confusion)
    return avg_loss, overall_acc, miou, iou


def _predict_single_scan(points, model, device, num_points):
    """Run sliding-window inference over a variable-length point cloud."""
    total = points.shape[0]
    preds = np.zeros(total, dtype=np.int64)
    start = 0
    while start < total:
        end = min(start + num_points, total)
        chunk = points[start:end]
        pad = num_points - chunk.shape[0]
        if pad > 0:
            pad_points = np.repeat(chunk[-1][None, :], pad, axis=0)
            chunk = np.concatenate([chunk, pad_points], axis=0)
        tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
        tensor = tensor.permute(0, 2, 1)
        with torch.no_grad():
            logits = model(tensor)
        pred_chunk = logits.max(dim=1)[1].cpu().numpy().reshape(-1)
        preds[start:end] = pred_chunk[:end-start]
        start = end
    return preds

def train(args, io):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points),
                              batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False,
                             num_workers=args.num_workers)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False,
                             num_workers=args.num_workers)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


def train_segmentation(args, io):
    if args.model != 'dgcnn':
        raise ValueError('Segmentation currently supports the DGCNN backbone only.')
    train_dataset = TomatoDataset(root=args.tomato_root, split='train', num_points=args.num_points, augment=True)
    val_dataset = TomatoDataset(root=args.tomato_root, split='val', num_points=args.num_points, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNNSeg(args, output_channels=args.num_classes).to(device)
    model = nn.DataParallel(model)

    if args.use_sgd:
        io.cprint("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        io.cprint("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    ignore_index = args.ignore_label if args.ignore_label >= 0 else None
    
    # Setup class weights if provided
    class_weights = None
    if args.class_weights is not None:
        if len(args.class_weights) != args.num_classes:
            raise ValueError(f'class_weights must have {args.num_classes} values')
        class_weights = torch.FloatTensor(args.class_weights).to(device)
        io.cprint(f'Using class weights: {args.class_weights}')
    
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights) if ignore_index is not None else nn.CrossEntropyLoss(weight=class_weights)

    best_miou = 0.0
    for epoch in range(args.epochs):
        scheduler.step()
        train_loss, train_acc, train_miou, train_iou = run_segmentation_epoch(
            model, train_loader, device, criterion, args.num_classes, ignore_index, optimizer=opt)
        val_loss, val_acc, val_miou, val_iou = run_segmentation_epoch(
            model, val_loader, device, criterion, args.num_classes, ignore_index, optimizer=None)

        io.cprint('Train %d, loss: %.6f, acc: %.6f, mIoU: %.6f' %
                  (epoch, train_loss, train_acc, train_miou))
        io.cprint('Val   %d, loss: %.6f, acc: %.6f, mIoU: %.6f, IoU per class: %s' %
                  (epoch, val_loss, val_acc, val_miou, np.array2string(val_iou, precision=4, suppress_small=True)))

        if val_miou >= best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)


def eval_segmentation(args, io):
    if args.model != 'dgcnn':
        raise ValueError('Segmentation currently supports the DGCNN backbone only.')
    val_dataset = TomatoDataset(root=args.tomato_root, split='val', num_points=args.num_points, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNNSeg(args, output_channels=args.num_classes).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    ignore_index = args.ignore_label if args.ignore_label >= 0 else None
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss()

    val_loss, val_acc, val_miou, val_iou = run_segmentation_epoch(
        model, val_loader, device, criterion, args.num_classes, ignore_index, optimizer=None)
    io.cprint('Test :: loss: %.6f, acc: %.6f, mIoU: %.6f, IoU per class: %s' %
              (val_loss, val_acc, val_miou, np.array2string(val_iou, precision=4, suppress_small=True)))


def predict_segmentation(args, io):
    if args.model != 'dgcnn':
        raise ValueError('Segmentation currently supports the DGCNN backbone only.')
    if not args.model_path:
        raise ValueError('Provide --model_path for prediction.')
    dataset = TomatoInferenceDataset(root=args.tomato_root, split=args.predict_split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNNSeg(args, output_channels=args.num_classes).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    base_output = os.path.join(args.prediction_dir, args.predict_split)
    os.makedirs(base_output, exist_ok=True)

    with torch.no_grad():
        for points, rel_paths in loader:
            coords = points.squeeze(0).cpu().numpy()
            rel_path = rel_paths[0]
            preds = _predict_single_scan(coords, model, device, args.num_points)
            out_rel = os.path.splitext(rel_path)[0] + '_pred.npz'
            out_path = os.path.join(base_output, out_rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(out_path, coords=coords, preds=preds)
            io.cprint('Saved predictions to %s' % out_path)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'tomato'])
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'seg'],
                        help='Training task')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--tomato_root', type=str, default='dataset/tomato',
                        help='Root directory for the tomato dataset')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of semantic classes for segmentation')
    parser.add_argument('--ignore_label', type=int, default=-1,
                        help='Label id to ignore when computing loss/metrics')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--predict', action='store_true',
                        help='Run prediction on the specified dataset split')
    parser.add_argument('--predict_split', type=str, default='test',
                        help='Tomato split to use when predicting')
    parser.add_argument('--prediction_dir', type=str, default='predictions',
                        help='Directory to store prediction files')
    parser.add_argument('--class_weights', type=float, nargs='+', default=None,
                        help='Class weights for loss function (e.g., --class_weights 0.23 2.52 0.25)')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if args.dataset == 'tomato' and args.task == 'cls':
        io.cprint('Dataset tomato only supports segmentation, switching task to seg.')
        args.task = 'seg'
    if args.task == 'cls' and args.dataset != 'modelnet40':
        raise ValueError('Classification is only configured for ModelNet40.')
    if args.task == 'seg' and args.dataset != 'tomato':
        raise ValueError('Segmentation is only configured for the tomato dataset.')

    if args.predict:
        if args.task != 'seg' or args.dataset != 'tomato':
            raise ValueError('Prediction is currently implemented only for tomato segmentation.')
        predict_segmentation(args, io)
        sys.exit(0)

    if not args.eval:
        if args.task == 'cls':
            train(args, io)
        else:
            train_segmentation(args, io)
    else:
        if args.task == 'cls':
            test(args, io)
        else:
            eval_segmentation(args, io)
