import argparse
import copy
import logging
import math
import time

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from preact_resnet import PreActResNet18
from utils import *


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt):
    delta = torch.zeros_like(X).cuda()
    delta[:, 0, :, :].uniform_(-epsilon[0][0][0].item(), epsilon[0][0][0].item())
    delta[:, 1, :, :].uniform_(-epsilon[1][0][0].item(), epsilon[1][0][0].item())
    delta[:, 2, :, :].uniform_(-epsilon[2][0][0].item(), epsilon[2][0][0].item())
    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(clamp(X + delta, lower_limit, upper_limit))
        loss = F.cross_entropy(output, y)
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
        grad = delta.grad.detach()
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'piecewise'])
    parser.add_argument('--lr-max', default=0.21, type=float)
    parser.add_argument('--attack', default='fgsm', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=5, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=int)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='cifar_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--overfit-check', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    start_start_time = time.time()
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        transform=transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std)
        ]))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=2,
        pin_memory=True, shuffle=True, drop_last=False)

    epsilon = (args.epsilon / 255.) / std
    pgd_alpha = (args.pgd_alpha / 255.) / std

    model = PreActResNet18().cuda()
    model.apply(initialize_weights)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)

    model, opt = amp.initialize(model, opt, opt_level="O2", loss_scale=1.0, master_weights=False)

    criterion = nn.CrossEntropyLoss()

    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.attack == 'free':
        assert args.epochs % args.attack_iters == 0
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs

    if args.lr_schedule == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.

    prev_robust_acc = 0.
    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, batch in enumerate(dataloader):
            X, y = batch

            X = X.to(device).float()
            y = y.to(device).long()

            if i == 0:
                first_batch = batch
            lr = lr_schedule(epoch + (i + 1) / len(dataloader))
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'pgd':
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, opt)

            elif args.attack == 'fgsm':
                if args.fgsm_init == 'zero':
                    delta = torch.zeros_like(X, requires_grad=True)
                elif args.fgsm_init == 'random':
                    delta = torch.zeros_like(X).cuda()
                    delta[:, 0, :, :].uniform_(-epsilon[0][0][0].item(), epsilon[0][0][0].item())
                    delta[:, 1, :, :].uniform_(-epsilon[1][0][0].item(), epsilon[1][0][0].item())
                    delta[:, 2, :, :].uniform_(-epsilon[2][0][0].item(), epsilon[2][0][0].item())
                    delta.requires_grad = True
                elif args.fgsm_init == 'previous':
                    delta.requires_grad = True
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + args.fgsm_alpha * epsilon * torch.sign(grad), -epsilon, epsilon)
                delta = delta.detach()

            elif args.attack == 'free':
                delta.requires_grad = True
                for j in range(args.attack_iters):
                    epoch_iters = epoch * args.attack_iters + (i * args.attack_iters + j + 1) / len(dataloader)
                    lr = lr_schedule(epoch_iters)
                    opt.param_groups[0].update(lr=lr)
                    output = model(clamp(X + delta[:X.size(0)], lower_limit, upper_limit))
                    loss = F.cross_entropy(output, y)
                    opt.zero_grad()
                    with amp.scale_loss(loss, opt) as scaled_loss:
                        scaled_loss.backward()
                    grad = delta.grad.detach()
                    delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    delta.grad.zero_()
            elif args.attack == 'none':
                delta = torch.zeros_like(X)

            output = model(clamp(X + delta[:X.size(0)], lower_limit, upper_limit))
            loss = criterion(output, y)
            if args.attack != 'free':
                opt.zero_grad()
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        if args.overfit_check:
            # Check current PGD robustness of model using random minibatch
            X, y = first_batch
            X = X.to(device).float()
            y = y.to(device).long()

            pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, opt)
            with torch.no_grad():
                output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
            robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
            if robust_acc - prev_robust_acc < -0.5:
                break
            prev_robust_acc = robust_acc
        best_state_dict = copy.deepcopy(model.state_dict())

        train_time = time.time()
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
            epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
    torch.save(best_state_dict, args.fname + '.pth')
    logger.info('Total time: %.4f', train_time - start_start_time)


if __name__ == "__main__":
    main()
