from copy import deepcopy
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import csv
import math
import time
import random
import pickle
import argparse
from contextlib import nullcontext
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler

from util.icbhi_diffusion_dataset import ICBHIDataset
from util.metrics import evaluate_confusion
from util.augmentation import SpecAugment
from util.misc import adjust_learning_rate, warmup_learning_rate, set_optimizer, update_moving_average
from util.misc import AverageMeter, accuracy, save_model, update_json
from models import get_backbone_class, Projector
from method import PatchMixLoss, PatchMixConLoss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer('alpha', alpha if alpha is not None else torch.ones(4))

    def forward(self, logits, target):
        logp = torch.log_softmax(logits.float(), dim=-1)
        lp = logp.gather(1, target.long().view(-1, 1)).squeeze(1)
        alpha = self.alpha.to(logits.device)[target.long()]
        return (-(alpha * (1.0 - lp.exp()).pow(self.gamma) * lp)).mean()


def autocast_context(args):
    if args.amp == 'fp32':
        return nullcontext()
    dtype = torch.float16 if args.amp == 'fp16' else torch.bfloat16
    return torch.autocast(device_type='cuda', dtype=dtype)


@torch.no_grad()
def update_ema(shadow, source, decay):
    src_params = dict(source.named_parameters())
    for name, target_param in shadow.named_parameters():
        target_param.lerp_(src_params[name].detach(), 1.0 - decay)
    src_buffers = dict(source.named_buffers())
    for name, target_buffer in shadow.named_buffers():
        source_buffer = src_buffers[name].detach()
        if target_buffer.is_floating_point():
            target_buffer.lerp_(source_buffer, 1.0 - decay)
        else:
            target_buffer.copy_(source_buffer)


def parse_args():
    parser = argparse.ArgumentParser('argument for supervised training')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--train_real_only', action='store_true')
    parser.add_argument('--imbalance', type=str, default='none',
                        choices=['none', 'weighted_ce', 'balanced_sampler'],
                        help='real-training class imbalance control')
    parser.add_argument('--real_gen_dir', type=str, default='./data/train_5secs/real_with_gen')
    parser.add_argument('--N', type=int, default=None,
                        help='fill-to-N target recorded for synthetic-mix experiments')
    parser.add_argument('--synth_version', type=str, default='',
                        help='synthetic pool version recorded in run metadata')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='path of model checkpoint to resume')
    parser.add_argument('--eval', action='store_true',
                        help='only evaluation with pretrained encoder and classifier')
    parser.add_argument('--two_cls_eval', action='store_true',
                        help='evaluate with two classes')
    
    # optimization
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--warm_epochs', type=int, default=0,
                        help='warmup epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='explicit number of LR warmup epochs; overrides the legacy warmup length')
    parser.add_argument('--loss', choices=['ce', 'focal'], default='ce')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='focal-loss gamma when --loss focal')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--amp', choices=['fp32', 'fp16', 'bf16'], default='fp16',
                        help='mixed-precision mode; fp16 preserves legacy default behavior')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='separate EMA shadow model decay; zero disables it')
    parser.add_argument('--llrd', type=float, default=1.0,
                        help='AST layer-wise learning-rate decay; values below 1 enable LLRD')
    parser.add_argument('--head_lr_mult', type=float, default=1.0,
                        help='learning-rate multiplier for AST normalization and classifier heads')
    parser.add_argument('--drop_path', type=float, default=0.0,
                        help='maximum stochastic-depth probability across AST blocks')
    parser.add_argument('--rms_norm', action='store_true',
                        help='normalize each cycle to the train-set median RMS (gain capped at 10x)')
    parser.add_argument('--view', choices=['mel', 'scalo', 'both'], default='mel',
                        help='AST input view: Kaldi fbank, cached CWT, or paired two-channel early fusion')
    parser.add_argument('--scalo_cache_dir', type=str, default=None,
                        help='root with train/, test/, and stats.json created by precompute_scalogram_cache.py')
    parser.add_argument('--scalo_stats_json', type=str, default=None,
                        help='optional explicit CWT normalization metadata path')
    # dataset
    parser.add_argument('--dataset', type=str, default='icbhi')
    parser.add_argument('--data_folder', type=str, default='./data/')
    parser.add_argument('--val_mode', choices=['none', 'holdout'], default='none',
                        help='use a patient-grouped validation subset carved from official train')
    parser.add_argument('--val_manifest', type=str, default=None,
                        help='CSV listing validation filenames; required for --val_mode holdout')
    parser.add_argument('--save_probs', action='store_true',
                        help='save final official-test probability matrices for best-val and last checkpoints')
    parser.add_argument('--final_test', action='store_true',
                        help='open the official test set after training (use for the locked confirmation recipe only)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    # icbhi dataset
    parser.add_argument('--class_split', type=str, default='lungsound',
                        help='lungsound: (normal, crackles, wheezes, both), diagnosis: (healthy, chronic diseases, non-chronic diseases)')
    parser.add_argument('--n_cls', type=int, default=0,
                        help='set k-way classification problem for class')
    parser.add_argument('--m_cls', type=int, default=0,
                        help='set k-way classification problem for domain (meta)')
    parser.add_argument('--test_fold', type=str, default='official', choices=['official', '0', '1', '2', '3', '4'],
                        help='test fold to use official 60-40 split or 80-20 split from RespireNet')
    parser.add_argument('--sample_rate', type=int,  default=16000, 
                        help='sampling rate when load audio data, and it denotes the number of samples per one second')
    parser.add_argument('--desired_length', type=int,  default=8, 
                        help='fixed length size of individual cycle')
    parser.add_argument('--n_mels', type=int, default=128,
                        help='the number of mel filter banks')
    parser.add_argument('--pad_types', type=str,  default='repeat', 
                        help='zero: zero-padding, repeat: padding with duplicated samples, aug: padding with augmented samples')
    parser.add_argument('--resz', type=float, default=1, 
                        help='resize the scale of mel-spectrogram')
    parser.add_argument('--raw_augment', type=int, default=0, 
                        help='control how many number of augmented raw audio samples')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                        help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])
    parser.add_argument('--nospec', action='store_true')

    # model
    parser.add_argument('--model', type=str, default='ast')
    parser.add_argument('--pretrained', action='store_true')
    
    parser.add_argument('--pretrained_ckpt', type=str, default=None,
                        help='path to pre-trained encoder model')
    parser.add_argument('--from_sl_official', action='store_true',
                        help='load from supervised imagenet-pretrained model (official PyTorch)')
    parser.add_argument('--ma_update', action='store_true',
                        help='whether to use moving average update for model')
    parser.add_argument('--ma_beta', type=float, default=0,
                        help='moving average value')
    # for AST
    parser.add_argument('--audioset_pretrained', action='store_true',
                        help='load from imagenet- and audioset-pretrained model')
    # for SSAST
    parser.add_argument('--ssast_task', type=str, default='ft_avgtok', 
                        help='pretraining or fine-tuning task', choices=['ft_avgtok', 'ft_cls'])
    parser.add_argument('--fshape', type=int, default=16, 
                        help='fshape of SSAST')
    parser.add_argument('--tshape', type=int, default=16, 
                        help='tshape of SSAST')
    parser.add_argument('--ssast_pretrained_type', type=str, default='Patch', 
                        help='pretrained ckpt version of SSAST model')

    parser.add_argument('--method', type=str, default='ce')
    parser.add_argument('--adversarial_ft', action='store_true')    
    # Meta Domain CL & Patch-Mix CL loss
    parser.add_argument('--proj_dim', type=int, default=768)
    parser.add_argument('--mix_beta', default=1.0, type=float,
                        help='patch-mix interpolation coefficient')
    parser.add_argument('--time_domain', action='store_true',
                        help='patchmix for the specific time domain')

    parser.add_argument('--temperature', type=float, default=0.06)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--lambda_dis', type=float, default=0.2,
                        help='weight for adversarial domain loss in Patch-Mix CL AFT runs')
    parser.add_argument('--negative_pair', type=str, default='all',
                        help='the method for selecting negative pair', choices=['all', 'diff_label'])
    parser.add_argument('--target_type', type=str, default='project1_project2block',
                        help='how to make target representation', choices=['project_flow', 'grad_block1', 'grad_flow1', 'project_block1', 'grad_block2', 'grad_flow2', 'project_block2', 'project_block_all', 'representation_all', 'grad_block', 'grad_flow', 'project_block'])
    
    # Meta for SCL
    parser.add_argument('--meta_mode', type=str, default='none',
                        help='the meta information for selecting', choices=['none', 'mixed'])
                        
    args = parser.parse_args()
    if args.warmup_epochs > 0:
        args.warm = True
        args.warm_epochs = args.warmup_epochs

    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.model_name = '{}_{}_{}'.format(args.dataset, args.model, args.method) if args.meta_mode == 'none' else '{}_{}_{}_{}'.format(args.dataset, args.model, args.method, args.meta_mode)
    if args.tag:
        args.model_name += '_{}'.format(args.tag)

    if args.method in ['patchmix', 'patchmix_cl']:
        assert args.model in ['ast', 'ssast']
    if args.loss == 'focal':
        assert args.method == 'ce' and not args.adversarial_ft, 'focal loss is currently supported only for non-adversarial CE runs'
    if args.imbalance != 'none':
        assert args.method == 'ce' and args.train_real_only and not args.adversarial_ft, (
            'Imbalance controls are currently defined only for real-only, non-adversarial CE runs'
        )
    
    if args.adversarial_ft:
        args.save_folder = os.path.join(args.save_dir, 'aft', args.model_name)
    else:
        args.save_folder = os.path.join(args.save_dir, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        if args.warm_epochs <= 0:
            args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate

    if args.adversarial_ft:
        if args.meta_mode in ['mixed']:
            args.m_cls = 2
            
    if args.dataset == 'icbhi':
        if args.class_split == 'lungsound':
            if args.adversarial_ft:
                #single
                if args.meta_mode == 'mixed':
                    args.meta_cls_list = ['Real', 'Generated']
            if args.n_cls == 4:
                args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
            elif args.n_cls == 2:
                args.cls_list = ['normal', 'abnormal']
                
        elif args.class_split == 'diagnosis':
            if args.n_cls == 3:
                args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
            elif args.n_cls == 2:
                args.cls_list = ['healthy', 'unhealthy']
            else:
                raise NotImplementedError
        
    else:
        raise NotImplementedError
    
    if args.n_cls == 0 and args.m_cls !=0:
        args.n_cls = args.m_cls
        args.cls_list = args.meta_cls_list

    return args


def set_loader(args):
    if args.dataset == 'icbhi':        
        args.h = int(args.desired_length * 100 - 2)
        args.w = 128
        #args.h, args.w = 798, 128
        train_transform = [transforms.ToTensor(),
                            transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        val_transform = [transforms.ToTensor(),
                        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
        ##
        train_transform = transforms.Compose(train_transform)
        val_transform = transforms.Compose(val_transform)

        if args.rms_norm:
            import soundfile as sf
            from glob import glob
            rms_values = []
            train_wavs = sorted(glob(os.path.join(args.data_folder, 'training', 'real', '*.wav')))
            if args.val_mode == 'holdout':
                with open(args.val_manifest, newline='', encoding='utf-8') as f:
                    val_names = {row['filename'] for row in csv.DictReader(f)}
                train_wavs = [p for p in train_wavs if os.path.basename(p) not in val_names]
            for wav_path in train_wavs:
                signal, _sr = sf.read(wav_path, dtype='float32')
                if signal.ndim > 1:
                    signal = signal.mean(axis=1)
                rms_values.append(float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)))))
            if not rms_values:
                raise ValueError('Cannot compute train-set RMS target: no official training WAVs found')
            args.rms_target = float(np.median(rms_values))
            print(f'RMS normalization target (official train median)={args.rms_target:.8f}')

        train_dataset = ICBHIDataset(train_flag=True, transform=train_transform, args=args, real=args.train_real_only, print_flag=True)
        val_dataset = ICBHIDataset(train_flag=False, transform=val_transform, args=args, real=True, print_flag=True)

        args.class_nums = args.n_cls
    else:
        raise NotImplemented    
    
    train_labels = []
    for path in train_dataset.data_glob:
        stem = os.path.basename(path).rsplit('.wav', 1)[0]
        label_token = stem.split('_')[-2] if '_index=' in stem else stem.split('_')[-1]
        train_labels.append(int(label_token))
    class_counts = np.bincount(train_labels, minlength=args.n_cls).astype(np.int64)
    if len(train_labels) != len(train_dataset) or np.any(class_counts == 0):
        raise AssertionError(f'Invalid training class counts from dataset paths: {class_counts.tolist()}')
    args.train_class_counts = class_counts.tolist()
    args.class_weights = None
    sampler = None
    shuffle = True
    if args.imbalance == 'weighted_ce':
        weights = np.sqrt(len(train_labels) / (args.n_cls * class_counts.astype(np.float64)))
        weights /= weights.mean()
        args.class_weights = weights.tolist()
    elif args.imbalance == 'balanced_sampler':
        per_item = np.asarray([1.0 / class_counts[label] for label in train_labels], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(per_item, dtype=torch.double),
                                        num_samples=len(train_labels), replacement=True)
        shuffle = False

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle,
                                           sampler=sampler, num_workers=args.num_workers, pin_memory=True,
                                           drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    
    return train_loader, val_loader, args
    

def set_model(args):
    kwargs = {}
    if args.model == 'ast':
        kwargs['input_fdim'] = int(args.h * args.resz)
        kwargs['input_tdim'] = int(args.w * args.resz)
        kwargs['label_dim'] = args.n_cls
        kwargs['imagenet_pretrain'] = args.from_sl_official
        kwargs['audioset_pretrain'] = args.audioset_pretrained
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL
        if args.adversarial_ft:
            kwargs['domain_label_dim'] = args.m_cls
    elif args.model == 'ssast':
        kwargs['label_dim'] = args.n_cls
        kwargs['fshape'], kwargs['tshape'] = args.fshape, args.tshape
        kwargs['fstride'], kwargs['tstride'] = 10, 10
        kwargs['input_tdim'] = 798
        kwargs['task'] = args.ssast_task
        kwargs['pretrain_stage'] = not args.audioset_pretrained
        kwargs['load_pretrained_mdl_path'] = args.ssast_pretrained_type
        kwargs['mix_beta'] = args.mix_beta  # for Patch-MixCL
        if args.adversarial_ft:
            kwargs['domain_label_dim'] = args.m_cls # not debugging yet

    model = get_backbone_class(args.model)(**kwargs)
    if args.view == 'both':
        if args.model != 'ast':
            raise ValueError('The two-view fusion input is currently implemented only for AST')
        # Preserve the existing single-channel AST response exactly. The
        # second channel starts at zero and is learned during fine-tuning.
        projection = model.v.patch_embed.proj
        if projection.in_channels != 1:
            raise ValueError(f'Expected a one-channel pretrained patch embed, got {projection.in_channels}')
        two_channel = nn.Conv2d(
            2, projection.out_channels, kernel_size=projection.kernel_size,
            stride=projection.stride, padding=projection.padding,
            dilation=projection.dilation, groups=projection.groups,
            bias=projection.bias is not None, padding_mode=projection.padding_mode)
        with torch.no_grad():
            two_channel.weight[:, 0].copy_(projection.weight[:, 0])
            two_channel.weight[:, 1].zero_()
            if projection.bias is not None:
                two_channel.bias.copy_(projection.bias)
        model.v.patch_embed.proj = two_channel
    if args.drop_path > 0:
        if args.model != 'ast' or not hasattr(model, 'v') or not hasattr(model.v, 'blocks'):
            raise ValueError('--drop_path is currently implemented for the AST backbone only')
        from timm.models.layers import DropPath
        block_count = len(model.v.blocks)
        for index, block in enumerate(model.v.blocks):
            rate = args.drop_path * index / max(block_count - 1, 1)
            block.drop_path = DropPath(drop_prob=rate) if rate > 0 else nn.Identity()
    if args.adversarial_ft:
        class_classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
        data_discriminator = nn.Linear(model.final_feat_dim, args.m_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.domain_mlp_head)
    else:
        classifier = nn.Linear(model.final_feat_dim, args.n_cls) if args.model not in ['ast', 'ssast'] else deepcopy(model.mlp_head)
    projector = Projector(model.final_feat_dim, args.proj_dim) if args.method in ['patchmix_cl'] else nn.Identity()
    
    
    class_weight = None
    if args.imbalance == 'weighted_ce':
        if not args.class_weights:
            raise ValueError('weighted_ce requires class weights computed from the training dataset')
        class_weight = torch.tensor(args.class_weights, dtype=torch.float32)
        print(f'Class counts: {args.train_class_counts}; normalized sqrt-inverse weights: {args.class_weights}')
    if args.loss == 'focal':
        if not args.train_class_counts:
            raise ValueError('focal loss requires class counts from the training data')
        counts = np.asarray(args.train_class_counts, dtype=np.float64)
        alpha = 1.0 / np.maximum(counts, 1.0)
        alpha = alpha / alpha.mean()
        criterion = FocalLoss(args.gamma, torch.tensor(alpha, dtype=torch.float32))
        print(f'Focal loss gamma={args.gamma}; normalized inverse-frequency alpha={alpha.tolist()}')
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=args.label_smoothing)
        
    if args.adversarial_ft:
        criterion2 = nn.CrossEntropyLoss()
        
           
    if args.model not in ['ast', 'ssast'] and args.from_sl_official:
        model.load_sl_official_weights()
        print('pretrained model loaded from PyTorch ImageNet-pretrained')

    # load SSL pretrained checkpoint for linear evaluation
    if args.pretrained and args.pretrained_ckpt is not None:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu', weights_only=False)
        state_dict = ckpt['model']

        # HOTFIX: always use dataparallel during SSL pretraining
        new_state_dict = {}
        for k, v in state_dict.items():
            if "module." in k:
                k = k.replace("module.", "")
            if "backbone." in k:
                k = k.replace("backbone.", "")
            if not 'mlp_head' in k: #del mlp_head
                new_state_dict[k] = v
        state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        
        if ckpt.get('classifier', None) is not None:
            classifier.load_state_dict(ckpt['classifier'], strict=True)

        print('pretrained model loaded from: {}'.format(args.pretrained_ckpt))
    
    if args.method == 'ce':
        if args.adversarial_ft:
            criterion = [criterion.cuda(), criterion2.cuda()]
        else:
            criterion = [criterion.cuda()]
    elif args.method == 'patchmix':
        
        if args.adversarial_ft:
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda(), PatchMixLoss(criterion=criterion2).cuda()]
        else:
            criterion = [criterion.cuda(), PatchMixLoss(criterion=criterion).cuda()]
    elif args.method == 'patchmix_cl':
        criterion = [criterion.cuda(), PatchMixConLoss(temperature=args.temperature).cuda()]
        if args.adversarial_ft:
            criterion.append(criterion2.cuda())
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.cuda()
    if args.adversarial_ft:
        classifier = [class_classifier.cuda(), data_discriminator.cuda()]
    else:
        classifier.cuda()
    projector.cuda()
    
    
    if args.adversarial_ft:
        optim_params = list(model.parameters()) + list(classifier[0].parameters()) + list(classifier[-1].parameters()) + list(projector.parameters())
    else:
        optim_params = list(model.parameters()) + list(classifier.parameters()) + list(projector.parameters())
    if args.llrd < 1.0:
        if args.model != 'ast' or args.optimizer != 'adam' or args.adversarial_ft:
            raise ValueError('LLRD is implemented only for non-adversarial AST with Adam')
        backbone = model.v
        groups = []
        assigned = set()
        embed = list(backbone.patch_embed.parameters())
        for name in ('pos_embed', 'cls_token', 'dist_token'):
            value = getattr(backbone, name, None)
            if isinstance(value, nn.Parameter):
                embed.append(value)
        if embed:
            groups.append({'params': embed, 'lr': args.learning_rate * args.llrd ** 12})
            assigned.update(id(p) for p in embed)
        blocks = list(backbone.blocks)
        for i, block in enumerate(blocks):
            params = list(block.parameters())
            groups.append({'params': params, 'lr': args.learning_rate * args.llrd ** (len(blocks)-1-i)})
            assigned.update(id(p) for p in params)
        head = list(backbone.norm.parameters()) + list(classifier.parameters()) + list(projector.parameters())
        assigned.update(id(p) for p in head)
        other = [p for p in model.parameters() if id(p) not in assigned]
        head.extend(other)
        groups.append({'params': head, 'lr': args.learning_rate * args.head_lr_mult})
        optimizer = torch.optim.Adam(groups, weight_decay=args.weight_decay)
        print('AST LLRD groups:', [round(g['lr'], 10) for g in groups])
    else:
        optimizer = set_optimizer(args, optim_params)
    
    return model, classifier, projector, criterion, optimizer

def train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler=None,
          ema_model=None, ema_classifier=None):
    model.train()
    if args.adversarial_ft:
        classifier[0].train()
        classifier[1].train()
    else:
        classifier.train()
    projector.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    domain_correct = 0
    domain_total = 0

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        # The gradient-reversal coefficient is only updated by the legacy
        # moving-average path.  AST's ordinary CE path still passes it through
        # the model API, so give that path the neutral value explicitly.
        lamb = None
        # data load
        data_time.update(time.time() - end)
        images = images.cuda(non_blocking=True)
        if args.adversarial_ft:
            class_labels = labels[0].cuda(non_blocking=True)
            meta_labels = labels[1].cuda(non_blocking=True)
        else:
            labels = labels.cuda(non_blocking=True)
        bsz = class_labels.shape[0] if args.adversarial_ft else labels.shape[0]
        
        
        if args.ma_update:
            # store the previous iter checkpoint
            with torch.no_grad():
                if args.adversarial_ft:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier[0].state_dict()), deepcopy(classifier[1].state_dict()), deepcopy(projector.state_dict())]
                    p = float(idx + epoch * len(train_loader)) / args.epochs + 1 / len(train_loader)
                    lamb = 2. / (1. + np.exp(-10 * p)) - 1
                else:
                    ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict()), deepcopy(projector.state_dict())]
                    lamb = None

        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        with autocast_context(args):
            if args.method == 'ce':
                
                if args.nospec:
                    features = model(images, args=args, alpha=lamb, training=True)
                else:
                    features = model(args.transforms(images), args=args, alpha=lamb, training=True)
                if args.adversarial_ft:
                    #features = (features, domain_features) # domain_features -> ReverseLayerF                    
                    
                    output = classifier[0](features[0])
                    class_loss = criterion[0](output, class_labels)
                                   
                    meta_output = classifier[1](features[1])
                    meta_loss = criterion[1](meta_output, meta_labels)
                    domain_correct += (meta_output.argmax(dim=1) == meta_labels).sum().item()
                    domain_total += meta_labels.numel()
                    
                    loss = class_loss + args.alpha * meta_loss
                
                else:
                    output = classifier(features)
                    loss = criterion[0](output, labels)
                    

            elif args.method == 'patchmix':
                mix_images, labels_a, labels_b, lam, index = model(args.transforms(images), y=class_labels if args.adversarial_ft else labels, 
                    y2=meta_labels if args.adversarial_ft else None, da_index=None, args=args, alpha=lamb, patch_mix=True, time_domain=args.time_domain, training=True)
                
                if args.adversarial_ft:
                    output = classifier[0](mix_images[0])
                    class_loss = criterion[1](output, labels_a[0], labels_b[0], lam)
                    
                    meta_output = classifier[1](mix_images[1])
                    meta_loss = criterion[2](meta_output, labels_a[1], labels_b[1], lam)
                    
                    loss = class_loss + args.alpha * meta_loss
                    
                
                else:
                    output = classifier(mix_images)
                    loss = criterion[1](output, labels_a, labels_b, lam)

            elif args.method == 'patchmix_cl':
                input_images = images if args.nospec else args.transforms(images)
                if args.adversarial_ft:
                    features = model(input_images, args=args, alpha=lamb, training=True)
                    output = classifier[0](features[0])
                    class_loss = criterion[0](output, class_labels)
                    domain_logits = classifier[1](features[1])
                    domain_loss = criterion[2](domain_logits, meta_labels)
                    domain_correct += (domain_logits.argmax(dim=1) == meta_labels).sum().item()
                    domain_total += meta_labels.numel()
                    feature_for_proj = features[0]
                    mix_images, labels_a, labels_b, lam, index = model(
                        input_images, y=class_labels, y2=meta_labels, args=args,
                        patch_mix=True, time_domain=args.time_domain)
                    mix_features = mix_images
                    mix_labels_a = labels_a[0]
                    mix_labels_b = labels_b[0]
                    loss = class_loss + args.lambda_dis * domain_loss
                else:
                    features = model(input_images)
                    output = classifier(features)
                    class_loss = criterion[0](output, labels)
                    feature_for_proj = features
                    mix_images, labels_a, labels_b, lam, index = model(
                        input_images, y=labels, args=args,
                        patch_mix=True, time_domain=args.time_domain)
                    mix_features = mix_images
                    mix_labels_a = labels_a
                    mix_labels_b = labels_b
                    loss = class_loss

                if args.target_type == 'grad_block':
                    proj1 = deepcopy(feature_for_proj.detach())
                elif args.target_type == 'grad_flow':
                    proj1 = feature_for_proj
                elif args.target_type == 'project_block':
                    proj1 = deepcopy(projector(feature_for_proj).detach())
                elif args.target_type == 'project_flow':
                    proj1 = projector(feature_for_proj)
                proj2 = projector(mix_features)
                loss += args.alpha * criterion[1](
                    proj1, proj2, mix_labels_a, mix_labels_b, lam, index, args)

        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], class_labels if args.adversarial_ft else labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if ema_model is not None:
            update_ema(ema_model, model, args.ema_decay)
            update_ema(ema_classifier, classifier, args.ema_decay)
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.ma_update:
            with torch.no_grad():
                # exponential moving average update
                model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
                if args.adversarial_ft:
                    classifier[0] = update_moving_average(args.ma_beta, classifier[0], ma_ckpt[1])
                    classifier[1] = update_moving_average(args.ma_beta, classifier[1], ma_ckpt[2])
                else:
                    classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])
                projector_ema_state = ma_ckpt[3] if args.adversarial_ft else ma_ckpt[2]
                projector = update_moving_average(args.ma_beta, projector, projector_ema_state)

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    if args.adversarial_ft and args.method in ['ce', 'patchmix_cl']:
        return losses.avg, top1.avg, (domain_correct / max(domain_total, 1))
    return losses.avg, top1.avg

def validate(val_loader, model, classifier, criterion, args, best_acc, best_model=None):
    save_bool = False
    model.eval()
    
    if args.adversarial_ft:
        classifier = classifier[0]
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    confusion = torch.zeros((args.n_cls, args.n_cls), dtype=torch.int64, device='cpu')
    probability_rows = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            

            with autocast_context(args):
                features = model(images, args=args, training=False)
                output = classifier(features)
                loss = criterion[0](output, labels)
            if getattr(args, 'save_probs', False):
                probability_rows.append(torch.softmax(output.float(), dim=-1).cpu().numpy())

            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            flat = (labels * args.n_cls + preds).detach().cpu()
            confusion += torch.bincount(flat, minlength=args.n_cls * args.n_cls).reshape(args.n_cls, args.n_cls)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    
    cm = confusion.numpy()
    current_metrics = evaluate_confusion(cm, two_cls_eval=args.two_cls_eval and args.n_cls == 4)
    micro_metrics = evaluate_confusion(cm)
    sp = 100.0 * current_metrics['Sp']
    se = 100.0 * current_metrics['Se']
    sc = 100.0 * current_metrics['Score']
    metrics = {
        'Se': se,
        'Sp': sp,
        'Score': sc,
        'accuracy': 100.0 * current_metrics['accuracy'],
        'recalls': (100.0 * np.asarray(current_metrics['recall'])).tolist(),
        'confusion': current_metrics['confusion'],
        'micro_metrics': {
            'Se': 100.0 * micro_metrics['Se_micro'],
            'Sp': 100.0 * micro_metrics['Sp_micro'],
            'Score': 100.0 * micro_metrics['Score_micro'],
            'Acc': 100.0 * micro_metrics['accuracy_micro'],
        },
    }
    if getattr(args, 'save_probs', False):
        metrics['probabilities'] = np.concatenate(probability_rows, axis=0)

    if sc > best_acc[-1] and se > 5:
        save_bool = True
        best_acc = [sp, se, sc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[-1]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool, metrics, losses.avg, top1.avg


def main():
    args = parse_args()
    with open(os.path.join(args.save_folder, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    # Runtime transform objects are not configuration values and cannot be
    # serialized to JSON. Keep their construction parameters in args, but omit
    # the instantiated object from the recorded run metadata.
    run_config = {key: value for key, value in vars(args).items() if key != 'transforms'}
    run_config['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')
    with open(os.path.join(args.save_folder, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=4)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    
    best_model = None
    if args.dataset == 'icbhi':
        best_acc = [0, 0, 0]  # Specificity, Sensitivity, Score
    
    if not args.nospec:
        args.transforms = SpecAugment(args)
    train_loader, val_loader, args = set_loader(args)
    # Dataset-derived imbalance weights are unavailable until after loader creation.
    run_config = {key: value for key, value in vars(args).items() if key != 'transforms'}
    run_config['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')
    for metadata_name in ('train_args.json', 'config.json'):
        with open(os.path.join(args.save_folder, metadata_name), 'w') as f:
            json.dump(run_config, f, indent=4)
    model, classifier, projector, criterion, optimizer = set_model(args)
    ema_model = ema_classifier = None
    if args.ema_decay > 0:
        if args.adversarial_ft:
            raise ValueError('separate EMA shadow weights currently support non-adversarial runs only')
        ema_model = deepcopy(model).eval()
        ema_classifier = deepcopy(classifier).eval()
        for module in (ema_model, ema_classifier):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, weights_only=False)
            args.start_epoch = checkpoint['epoch'] 
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        args.start_epoch = 1

    # use mix_precision:
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == 'fp16')
    
    print('*' * 20)
    print('Checkpoint Name: {}'.format(args.model_name))
     
    if not args.eval:
        print('Training for {} epochs on {} dataset'.format(args.epochs, args.dataset))
        metrics_path = os.path.join(args.save_folder, 'metrics.csv')
        metric_fields = [
            'epoch', 'eval_split', 'is_best', 'train_loss', 'train_acc', 'test_loss', 'test_acc',
            'Se', 'Sp', 'Score', 'Se_micro', 'Sp_micro', 'Score_micro', 'accuracy_micro',
            'se_binary', 'accuracy',
            'recall_normal', 'recall_crackle', 'recall_wheeze', 'recall_both', 'epoch_seconds',
            'train_domain_acc',
        ] + [f'cm_{i}{j}' for i in range(args.n_cls) for j in range(args.n_cls)]
        metrics_mode = 'a' if args.resume and os.path.exists(metrics_path) else 'w'
        metrics_file = open(metrics_path, metrics_mode, newline='')
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=metric_fields)
        if metrics_mode == 'w':
            metrics_writer.writeheader()
        for epoch in range(args.start_epoch, args.epochs+1):
            adjust_learning_rate(args, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            
            train_result = train(train_loader, model, classifier, projector, criterion, optimizer, epoch, args, scaler,
                                 ema_model, ema_classifier)
            loss, acc = train_result[:2]
            train_domain_acc = train_result[2] if len(train_result) == 3 else ''
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2-time1, acc))
            
            # eval for one epoch
            eval_model = ema_model if ema_model is not None else model
            eval_classifier = ema_classifier if ema_classifier is not None else classifier
            best_acc, best_model, save_bool, current_metrics, test_loss, test_acc = validate(
                val_loader, eval_model, eval_classifier, criterion, args, best_acc, best_model)
            micro_metrics = evaluate_confusion(np.asarray(current_metrics['confusion']))
            metrics_row = {
                'epoch': epoch,
                'eval_split': 'validation' if args.val_mode == 'holdout' else 'official_test',
                'train_loss': loss,
                'train_acc': float(acc),
                'test_loss': test_loss,
                'test_acc': float(test_acc.item() if torch.is_tensor(test_acc) else test_acc),
                'Se': current_metrics['Se'],
                'Sp': current_metrics['Sp'],
                'Score': current_metrics['Score'],
                'Se_micro': 100.0 * micro_metrics['Se_micro'],
                'Sp_micro': 100.0 * micro_metrics['Sp_micro'],
                'Score_micro': 100.0 * micro_metrics['Score_micro'],
                'accuracy_micro': 100.0 * micro_metrics['accuracy_micro'],
                'se_binary': 100.0 * evaluate_confusion(
                    np.asarray(current_metrics['confusion']), two_cls_eval=True)['Se'],
                'accuracy': current_metrics['accuracy'],
                'epoch_seconds': time2 - time1,
                'is_best': save_bool,
                'train_domain_acc': train_domain_acc,
            }
            metrics_row.update({f'recall_{name}': value for name, value in zip(
                ['normal', 'crackle', 'wheeze', 'both'], current_metrics['recalls'])})
            metrics_row.update({f'cm_{i}{j}': current_metrics['confusion'][i][j]
                                for i in range(args.n_cls) for j in range(args.n_cls)})
            metrics_writer.writerow(metrics_row)
            metrics_file.flush()
            
            # save a checkpoint of model and classifier when the best score is updated
            if save_bool:            
                save_file = os.path.join(args.save_folder, 'best.pth')
                print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc[2], epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.adversarial_ft else classifier)
                
            if epoch % args.save_freq == 0:
                save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
                save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.adversarial_ft else classifier)

        metrics_file.close()
        # Retain compact last-epoch weights for unbiased last-epoch evaluation and
        # seed ensembles. Optimizer state is unnecessary for inference.
        last_file = os.path.join(args.save_folder, 'last.pth')
        last_model = ema_model if ema_model is not None else model
        last_classifier = ema_classifier if ema_classifier is not None else (classifier[0] if args.adversarial_ft else classifier)
        torch.save({
            'model': last_model.state_dict(),
            'classifier': last_classifier.state_dict(),
            'epoch': epoch,
        }, last_file)
        # Keep only one resumable full checkpoint for the best epoch.
        save_file = os.path.join(args.save_folder, 'best.pth')
        model.load_state_dict(best_model[0])
        classifier[0].load_state_dict(best_model[1]) if args.adversarial_ft else classifier.load_state_dict(best_model[1])
        save_model(model, optimizer, args, epoch, save_file, classifier[0] if args.adversarial_ft else classifier)
        if args.val_mode == 'holdout':
            # Screening outputs are validation-only. Official test evaluation is
            # opt-in for a locked confirmation run after recipe selection.
            official_test_loader = None
            if args.final_test:
                args.eval_official_test = True
                _, official_test_loader, args = set_loader(args)
            test_report = {'selection_rule': 'best checkpoint selected by validation Score', 'seed': args.seed}
            for checkpoint_name in ('best', 'last'):
                checkpoint_path = os.path.join(args.save_folder, f'{checkpoint_name}.pth')
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model'])
                (classifier[0] if args.adversarial_ft else classifier).load_state_dict(checkpoint['classifier'])
                if args.save_probs:
                    val_metrics = validate(val_loader, model, classifier, criterion, args, [0, 0, 0])[3]
                    np.save(os.path.join(args.save_folder, f'probs_val_{checkpoint_name}.npy'),
                            val_metrics['probabilities'])
                if args.final_test:
                    test_metrics = validate(official_test_loader, model, classifier, criterion, args, [0, 0, 0])[3]
                    test_metrics.pop('probabilities', None)
                    test_report[checkpoint_name] = test_metrics
                if args.save_probs and args.final_test:
                    # Probabilities are generated in a deterministic, ordered pass.
                    model.eval()
                    test_head = classifier[0] if args.adversarial_ft else classifier
                    test_head.eval()
                    outputs = []
                    with torch.no_grad():
                        for images, _labels in official_test_loader:
                            images = images.cuda(non_blocking=True)
                            with autocast_context(args):
                                features = model(images, args=args, training=False)
                                logits = test_head(features[0] if args.adversarial_ft else features)
                            outputs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
                    np.save(os.path.join(args.save_folder, f'probs_test_{checkpoint_name}.npy'), np.concatenate(outputs))
            if args.final_test:
                with open(os.path.join(args.save_folder, 'official_test_final.json'), 'w', encoding='utf-8') as f:
                    json.dump(test_report, f, indent=2)
        elif args.save_probs:
            # In the lean protocol val_loader is the official test loader and
            # is used only for per-epoch reporting.  Preserve the final model's
            # ordered probabilities for pre-registered ensemble evaluation.
            last_path = os.path.join(args.save_folder, 'last.pth')
            last_ckpt = torch.load(last_path, map_location='cpu', weights_only=False)
            model.load_state_dict(last_ckpt['model'])
            (classifier[0] if args.adversarial_ft else classifier).load_state_dict(last_ckpt['classifier'])
            _, _, _, last_metrics, _, _ = validate(
                val_loader, model, classifier, criterion, args, [0, 0, 0])
            probabilities = last_metrics.pop('probabilities')
            np.save(os.path.join(args.save_folder, 'probs_test_last.npy'), probabilities)
            np.savetxt(os.path.join(args.save_folder, 'confusion_test_last.csv'),
                       last_metrics['confusion'], fmt='%d', delimiter=',')
            with open(os.path.join(args.save_folder, 'official_test_final.json'), 'w', encoding='utf-8') as f:
                json.dump({'headline_checkpoint': 'last epoch', 'last': last_metrics}, f, indent=2)
    else:
        print('Testing the pretrained checkpoint on {} dataset'.format(args.dataset))
        best_acc, _, _, _, _, _ = validate(val_loader, model, classifier, criterion, args, best_acc)

    if args.adversarial_ft:
        update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results_aft.json'))
    else:
        update_json('%s' % args.model_name, best_acc, path=os.path.join(args.save_dir, 'results.json'))
    print('Checkpoint {} finished'.format(args.model_name))
    
if __name__ == '__main__':
    main()
