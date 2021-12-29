"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
Note: Our implementation is different from ADDA paper in several respects. We do not use separate networks for
source and target domain, nor fix classifier head. Besides, we do not adopt asymmetric objective loss function
of the feature extractor. We achieve promising results on digits datasets (reported by ADDA paper).
But on other benchmarks, ADDA-grl may achieve better results.
"""
# TODO：本代码复现内容和原文有几处不符合
"""
没有将source和target的encoder独立分开，
loss优化方法没有采用GAN，而是采用梯度反转
没有固定分类器头，应该先在source上获得预训练模型，固定分类器头，然后优化target的encoder
"""
# TODO: target CNN的初始化参数是？target CNN和head的优化间隔？
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import List

sys.path.append('../../..')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.adda import ImageClassifier, DomainAdversarialLoss
from dalib.translation.cyclegan.util import set_requires_grad
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, binary_accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance

sys.path.append('.')
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.arch, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True  # 使cuDNN只使用确定性的卷积算法
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True  # 使cuDNN对多个卷积算法进行基准测试并选择最快的

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    # 貌似对于office数据集，task val和test dataset是一样的
    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    if "resnet50" in args.arch:
        source_backbone = utils.get_model(args.arch, pretrain=not args.scratch)  # source的feature extractor
        source_bottleneck = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                          nn.Linear(source_backbone.out_features, args.bottleneck_dim),
                                          nn.BatchNorm1d(args.bottleneck_dim),
                                          nn.ReLU())
        source_CNN = nn.Sequential(source_backbone, source_bottleneck)

        target_backbone = utils.get_model(args.arch, pretrain=not args.scratch)
        target_CNN = nn.Sequential(target_backbone,
                                   nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                   nn.Linear(target_backbone.out_features, args.bottleneck_dim),
                                   nn.BatchNorm1d(args.bottleneck_dim),
                                   nn.ReLU()
                                   )
        classifier_head = nn.Linear(args.bottleneck_dim, num_classes)
    elif "vgg" in args.arch:
        pass  # TODO:vgg网络待增加

    domain_discri = DomainDiscriminator(in_feature=args.bottleneck_dim, hidden_size=args.discrim_dim).to(device)
    domain_adv_loss = DomainAdversarialLoss().to(device)

    # define optimizer and lr scheduler
    optimizer_head = SGD(classifier_head.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                         nesterov=True)
    optimizer_s_cnn = SGD(
        [{"params": source_backbone.parameters(), "lr": 0.1 * args.lr if not args.scratch else args.lr},
         {"params": source_bottleneck.parameters(), "lr": args.lr}],
        args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer_d = SGD(domain_discri.get_parameters(), args.lr_d, momentum=args.momentum, weight_decay=args.weight_decay,
                      nesterov=True)  # 领域判别器的优化器
    optimizer_t_cnn = SGD(target_CNN.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
    # LambdaLR是用来自定义学习率调整策略的，会将参数的原始学习率乘上一个因子
    lr_scheduler_head = LambdaLR(optimizer_head,
                                 lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_s_cnn = LambdaLR(optimizer_s_cnn,
                                  lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_t_cnn = LambdaLR(optimizer_t_cnn,
                                  lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_d = LambdaLR(optimizer_d, lambda x: args.lr_d * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('source_CNN_best'), map_location='cpu')
        source_CNN.load_state_dict(checkpoint)
        checkpoint = torch.load(logger.get_checkpoint_path('classifier_head_best'), map_location='cpu')
        classifier_head.load_state_dict(checkpoint)
        checkpoint = torch.load(logger.get_checkpoint_path('target_CNN_best'), map_location='cpu')
        target_CNN.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        # feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, source_CNN, device)
        target_feature = collect_feature(train_target_loader, target_CNN, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        temp_model = nn.Sequential(target_CNN, classifier_head)
        acc1 = utils.validate(test_loader, temp_model, args, device)
        print(acc1)
        return

    # 将source encoder和head在source上训练
    best_acc1 = 0.
    print("begin to train the source model on the source dataset")
    for epoch in range(args.epochs1):
        optimizer_list = [optimizer_s_cnn, optimizer_head]
        lr_scheduler_list = [lr_scheduler_s_cnn, lr_scheduler_head]
        print("lr source cnn:", lr_scheduler_s_cnn.get_lr())
        print("lr classifier head:", lr_scheduler_head.get_lr())
        train_source(source_CNN, classifier_head, train_source_iter, optimizer_list, lr_scheduler_list, epoch, args)

        temp_model = nn.Sequential(target_CNN, classifier_head)
        acc1 = utils.validate(val_loader, temp_model, args, device)
        torch.save(source_CNN.state_dict(), logger.get_checkpoint_path('source_CNN_latest'))
        torch.save(classifier_head.state_dict(), logger.get_checkpoint_path('classifier_head_latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('source_CNN_latest'), logger.get_checkpoint_path('source_CNN_best'))
            shutil.copy(logger.get_checkpoint_path('classifier_head_latest'),
                        logger.get_checkpoint_path('classifier_head_best'))
        best_acc1 = max(acc1, best_acc1)

    # 改为target_CNN和source_classifier的feature提取层对抗训练
    best_acc1 = 0
    print("begin to train the target CNN and source CNN using adversarial manner")
    for epoch in range(args.epochs2):
        optimizer_list = [optimizer_t_cnn, optimizer_d]
        lr_scheduler_list = [lr_scheduler_t_cnn, lr_scheduler_d]
        print("lr target cnn:", lr_scheduler_t_cnn.get_lr())
        print("lr discriminator:", lr_scheduler_d.get_lr())
        train_adversarial(source_CNN, target_CNN, domain_discri, domain_adv_loss, train_source_iter,
                          train_target_iter, optimizer_list, lr_scheduler_list, epoch, args)
        # 拼接模型，做测试
        temp_model = nn.Sequential(target_CNN, classifier_head)
        acc1 = utils.validate(val_loader, temp_model, args, device)
        torch.save(target_CNN.state_dict(), logger.get_checkpoint_path('target_CNN_latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('target_CNN_latest'), logger.get_checkpoint_path('target_CNN_best'))
        best_acc1 = max(acc1, best_acc1)

    print("val_best_acc1 = {:3.1f}".format(best_acc1))
    # evaluate on test set
    target_CNN.load_state_dict(torch.load(logger.get_checkpoint_path('target_CNN_best')))
    classifier_head.load_state_dict(torch.load(logger.get_checkpoint_path('classifier_head_best')))
    test_model = nn.Sequential(target_CNN, classifier_head)
    acc1 = utils.validate(test_loader, test_model, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train_source(source_cnn: nn.Module, head: nn.Module, train_source_iter: ForeverDataIterator, optimizer: List,
                 lr_scheduler: List, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')  # 对AverageMeter直接使用str()返回该指标的信息
    data_time = AverageMeter('Data', ':5.2f')
    losses_s = AverageMeter('Cls Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    progress = ProgressMeter(args.iters_per_epoch, [batch_time, data_time, losses_s, cls_accs],
                             prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)
        data_time.update(time.time() - end)

        source_cnn.train()
        head.train()
        set_requires_grad(source_cnn, True)
        set_requires_grad(head, False)

        y_s = head(source_cnn(x_s))
        loss = F.cross_entropy(y_s, labels_s)

        for i in optimizer:
            i.zero_grad()
        loss.backward()
        for i in optimizer:
            i.step()
        for i in lr_scheduler:
            i.step()

        losses_s.update(loss, x_s.size(0))
        cls_accs.update(accuracy(y_s, labels_s)[0].item(), x_s.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train_adversarial(source_cnn: nn.Module, target_cnn: nn.Module, domain_discri: DomainDiscriminator,
                      domain_adv: DomainAdversarialLoss, train_source_iter: ForeverDataIterator,
                      train_target_iter: ForeverDataIterator, optimizer: List, lr_sceduler: List, epoch: int,
                      args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')  # 对AverageMeter直接使用str()返回该指标的信息
    data_time = AverageMeter('Data', ':5.2f')  # 处理
    losses_cnn = AverageMeter('Cls Loss', ':6.2f')
    losses_discriminator = AverageMeter('Discriminator Loss', ':6.2f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(args.iters_per_epoch,
                             [batch_time, data_time, losses_discriminator, domain_accs],
                             prefix="Epoch: [{}]".format(epoch))
    set_requires_grad(source_cnn, False)
    source_cnn.eval()

    end = time.time()
    for i in range(args.iters_per_epoch):
        # 更新判别器
        for i in range(args.k1):
            x_s, labels_s = next(train_source_iter)
            x_t, _ = next(train_target_iter)
            x_s = x_s.to(device)
            x_t = x_t.to(device)
            data_time.update(time.time() - end)

            target_cnn.eval()
            domain_discri.train()
            set_requires_grad(target_cnn, False)
            set_requires_grad(domain_discri, True)

            f_s = source_cnn(x_s)
            f_t = target_cnn(x_t)
            f = torch.cat((f_s, f_t), dim=0)
            d = domain_discri(f)
            d_s, d_t = d.chunk(2, dim=0)
            loss_dis = 0.5 * (domain_adv(d_s, 'source') + domain_adv(d_t, 'target'))

            optimizer[1].zero_grad()
            loss_dis.backward()
            optimizer[1].step()
            lr_sceduler[1].step()

        # 更新Target CNN
        for i in range(args.k2):
            x_s, labels_s = next(train_source_iter)
            x_t, _ = next(train_target_iter)
            x_s = x_s.to(device)
            x_t = x_t.to(device)
            data_time.update(time.time() - end)

            target_cnn.train()
            domain_discri.eval()
            set_requires_grad(target_cnn, True)
            set_requires_grad(domain_discri, False)

            f = target_cnn(x_t)
            d = domain_discri(f)
            loss_cnn = domain_adv(d, 'source')

            optimizer[0].zero_grad()
            loss_cnn.backward()
            optimizer[0].step()
            lr_sceduler[0].step()

            losses_cnn.update(loss_cnn.item(), x_s.size(0))
            losses_discriminator.update(loss_dis.item(), x_s.size(0))
            domain_acc = 0.5 * (binary_accuracy(d_s, torch.ones_like(d_s)) + binary_accuracy(d_t, torch.zeros_like(d_t)))
            domain_accs.update(domain_acc.item(), x_s.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADDA for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('-k1', type=int, default=1)
    parser.add_argument('-k2', type=int, default=10)
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-discrim_dim', nargs='+', type=int, default=[1024, 2048, 3072])
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=0.1, type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate of the classifier', dest='lr')
    parser.add_argument('--lr-d', default=0.01, type=float,
                        help='initial learning rate of the domain discriminator')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs1', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epochs2', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='adda',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
