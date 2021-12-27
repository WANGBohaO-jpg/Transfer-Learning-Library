"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
Note: Our implementation is different from ADDA paper in several respects. We do not use separate networks for
source and target domain, nor fix classifier head. Besides, we do not adopt asymmetric objective loss function
of the feature extractor. We achieve promising results on digits datasets (reported by ADDA paper).
But on other benchmarks, ADDA-grl may achieve better results.
"""
# TODO：本代码复现内容和原文有几处不符合
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

sys.path.append('../../..')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.adda import ImageClassifier, DomainAdversarialLoss
from dalib.modules.gl import WarmStartGradientLayer
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
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)  # source的feature extractor
    pool_layer = nn.Identity() if args.no_pool else None
    # source完整的网络
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    # 领域判别器
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)

    # define loss function
    domain_adv = DomainAdversarialLoss().to(device)
    # TODO:看上去像梯度反转层 最大迭代次数1000是什么意思？
    gl = WarmStartGradientLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                    nesterov=True)  # 分类器的优化器
    optimizer_d = SGD(domain_discri.get_parameters(), args.lr_d, momentum=args.momentum, weight_decay=args.weight_decay,
                      nesterov=True)  # 领域判别器的优化器
    # LambdaLR是用来自定义学习率调整策略的
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_d = LambdaLR(optimizer_d, lambda x: args.lr_d * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.pool_layer, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr classifier:", lr_scheduler.get_lr())
        print("lr discriminator:", lr_scheduler_d.get_lr())
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_discri, domain_adv, gl, optimizer,
              lr_scheduler, optimizer_d, lr_scheduler_d, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: ImageClassifier,
          domain_discri: DomainDiscriminator, domain_adv: DomainAdversarialLoss, gl,
          optimizer: SGD, lr_scheduler: LambdaLR, optimizer_d: SGD, lr_scheduler_d: LambdaLR,
          epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')  # 对AverageMeter直接使用str()返回该指标的信息
    data_time = AverageMeter('Data', ':5.2f')
    losses_s = AverageMeter('Cls Loss', ':6.2f')
    losses_transfer = AverageMeter('Transfer Loss', ':6.2f')
    losses_discriminator = AverageMeter('Discriminator Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(args.iters_per_epoch,
                             [batch_time, data_time, losses_s, losses_transfer, losses_discriminator, cls_accs,
                              domain_accs], prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # Step 1: Train the classifier, freeze the discriminator
        model.train()
        domain_discri.eval()
        set_requires_grad(model, True)
        set_requires_grad(domain_discri, False)
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)  # 模型预测结果切分成两部分，
        loss_s = F.cross_entropy(y_s, labels_s)  # source域数据的分类误差

        # adversarial training to fool the discriminator
        d = domain_discri(gl(f))  # 梯度反转层
        d_s, d_t = d.chunk(2, dim=0)
        # 对抗损失 TODO：feature提取器的损失没有修改
        loss_transfer = 0.5 * (domain_adv(d_s, 'target') + domain_adv(d_t, 'source'))

        optimizer.zero_grad()
        (loss_s + loss_transfer * args.trade_off).backward()
        optimizer.step()
        lr_scheduler.step()

        # Step 2: Train the discriminator
        model.eval()
        domain_discri.train()
        set_requires_grad(model, False)
        set_requires_grad(domain_discri, True)
        d = domain_discri(f.detach())
        d_s, d_t = d.chunk(2, dim=0)
        loss_discriminator = 0.5 * (domain_adv(d_s, 'source') + domain_adv(d_t, 'target'))

        optimizer_d.zero_grad()
        loss_discriminator.backward()
        optimizer_d.step()
        lr_scheduler_d.step()

        losses_s.update(loss_s.item(), x_s.size(0))
        losses_transfer.update(loss_transfer.item(), x_s.size(0))
        losses_discriminator.update(loss_discriminator.item(), x_s.size(0))

        cls_acc = accuracy(y_s, labels_s)[0]
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_acc = 0.5 * (binary_accuracy(d_s, torch.ones_like(d_s)) + binary_accuracy(d_t, torch.zeros_like(d_t)))
        domain_accs.update(domain_acc.item(), x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADDA for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
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
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
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
