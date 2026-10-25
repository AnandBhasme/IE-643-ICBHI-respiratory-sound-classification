# removed curses import
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio
import torchaudio


def get_meta_infor(stetho, real_or_gen, args):
    
    if args.meta_mode == 'none':
        meta_label = -1
    
    
    elif args.meta_mode == 'mixed':
        if real_or_gen == 'real':
            meta_label = 0
        else:
            meta_label = 1
    
                   
    return meta_label


class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, real, print_flag=True):
        train_data_folder = os.path.join(args.data_folder, 'training', 'real') if real else os.path.join(args.data_folder, 'training', args.real_gen_dir)
        test_data_folder = os.path.join(args.data_folder, 'test', 'real')
        self.train_flag = train_flag
        
        if self.train_flag:
            self.data_folder = train_data_folder
        else:
            self.data_folder = test_data_folder
        self.transform = transform
        self.args = args
        
        # parameters for spectrograms
        self.sample_rate = self.args.sample_rate
        self.n_mels = self.args.n_mels        
        self.view = getattr(self.args, 'view', 'mel')
        if self.view not in ('mel', 'scalo', 'both'):
            raise ValueError(f'Unsupported spectrogram view: {self.view}')
        if self.view != 'mel' and self.args.raw_augment:
            raise ValueError('CWT views currently require --raw_augment 0 so cached features remain paired')
        self.scalo_dir = None
        self.scalo_mean = self.scalo_std = None
        if self.view in ('scalo', 'both'):
            cache_root = os.path.abspath(getattr(self.args, 'scalo_cache_dir', ''))
            stats_path = getattr(self.args, 'scalo_stats_json', None) or os.path.join(cache_root, 'stats.json')
            if not cache_root or not os.path.isfile(stats_path):
                raise FileNotFoundError('CWT view needs --scalo_cache_dir and its stats.json')
            import json
            with open(stats_path, encoding='utf-8') as f:
                scalo_stats = json.load(f)
            self.scalo_mean = float(scalo_stats['train_mean_db'])
            self.scalo_std = float(scalo_stats['train_std_db'])
            self.scalo_dir = os.path.join(cache_root, 'train' if self.train_flag else 'test')
        
        self.data_glob = sorted(glob(self.data_folder+'/*/*.wav')) if not real else sorted(glob(self.data_folder+'/*.wav'))
        if getattr(args, 'val_mode', 'none') == 'holdout' and not getattr(args, 'eval_official_test', False):
            if not args.val_manifest:
                raise ValueError('--val_mode holdout requires --val_manifest')
            import csv
            with open(args.val_manifest, newline='', encoding='utf-8') as f:
                selected = {row['filename'] for row in csv.DictReader(f)}
            train_manifest = os.path.join(args.data_folder, 'splits', 'train_manifest.csv')
            with open(train_manifest, newline='', encoding='utf-8') as f:
                official_train_names = {row['filename'] for row in csv.DictReader(f)}
            train_names = official_train_names - selected
            # Holdout validation is carved from official training recordings. Keep
            # official test files out of every screening epoch.
            if not train_flag:
                self.data_folder = os.path.join(args.data_folder, 'training', 'real')
                all_train_paths = sorted(glob(self.data_folder + '/*.wav'))
                self.data_glob = [p for p in all_train_paths if os.path.basename(p) in selected]
            else:
                # For a synthetic-mix run, retain generated examples while
                # filtering the real component to train-minus-validation.
                self.data_glob = [p for p in self.data_glob if
                                  ('_index=' in os.path.basename(p)) or
                                  os.path.basename(p) in train_names]
        if self.train_flag:
            diagnostic_paths = [path for path in self.data_glob if '/diag/' in path.replace('\\', '/')]
            if diagnostic_paths:
                raise AssertionError(
                    'Diagnostic-only audio under artifacts/diag/ must never enter a training dataset: '
                    + ', '.join(diagnostic_paths[:5])
                )
        
        print('Total length of dataset is', len(self.data_glob))
                                    
        # ==========================================================================
        """ convert fbank """
        self.audio_images = []
        for index in self.data_glob: #for the training set, 4142
            _, file_id = os.path.split(index)
            
            file_id_tmp = file_id.split('.wav')[0]
            stetho = file_id_tmp.split('_')[4]
            if 'index' in file_id_tmp:
                label = file_id_tmp.split('_')[-2]
                real_or_gen = 'gen'
                meta_label = get_meta_infor(stetho, real_or_gen, self.args)
            else:
                label = file_id_tmp.split('_')[-1]
                real_or_gen = 'real'
            meta_label = get_meta_infor(stetho, real_or_gen, self.args)
            import soundfile as sf
            audio_np, sr = sf.read(index)
            if getattr(self.args, 'rms_norm', False):
                mono = audio_np if audio_np.ndim == 1 else audio_np.mean(axis=1)
                rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
                target = float(self.args.rms_target)
                gain = min(10.0, target / (rms + 1e-8))
                audio_np = audio_np * gain
            audio = torch.from_numpy(audio_np).float()
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.transpose(0, 1)
            audio_image = []
            for aug_idx in range(self.args.raw_augment+1):
                if aug_idx > 0:
                    if self.train_flag:
                        audio = augment_raw_audio(np.asarray(audio.squeeze(0)), self.sample_rate, self.args)
                        audio = cut_pad_sample_torchaudio(torch.tensor(audio), self.args)                
                mel_image = None
                if self.view in ('mel', 'both'):
                    mel_image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels)
                if self.view in ('scalo', 'both'):
                    cache_path = os.path.join(self.scalo_dir, f'{file_id}.npy')
                    if not os.path.isfile(cache_path):
                        raise FileNotFoundError(f'Missing cached CWT feature: {cache_path}')
                    scalo_image = np.load(cache_path, mmap_mode='r').astype(np.float32)
                    scalo_image = (scalo_image - self.scalo_mean) / max(self.scalo_std, 1e-8)
                    scalo_image = scalo_image[..., None]
                if self.view == 'mel':
                    image = mel_image
                elif self.view == 'scalo':
                    image = scalo_image
                else:
                    if mel_image.shape[:2] != scalo_image.shape[:2]:
                        raise ValueError(f'Mel/CWT frame grids differ for {file_id}: {mel_image.shape} vs {scalo_image.shape}')
                    image = np.concatenate([mel_image, scalo_image], axis=-1)
                audio_image.append(image)
            self.audio_images.append((audio_image, int(label))) if self.args.meta_mode == 'none' else self.audio_images.append((audio_image, int(label), meta_label))
            
            
        '''
        self.class_nums = np.zeros(args.n_cls)
        self.class_nums[sample[1]] += 1
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        if print_flag:
            print('total number of audio data: {}'.format(len(self.data_glob)))
            print('*' * 25)
            print('For the Label Distribution')
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))
        '''
        # ==========================================================================

    def __getitem__(self, index):
        if self.args.meta_mode == 'none':
            audio_images, label = self.audio_images[index][0], self.audio_images[index][1]
        else:
            audio_images, label, meta_label = self.audio_images[index][0], self.audio_images[index][1], self.audio_images[index][2]

        if self.args.raw_augment and self.train_flag:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        
        if self.train_flag:
            if self.args.meta_mode == 'none':
                return audio_image, torch.tensor(label)
            else:
                return audio_image, (torch.tensor(label), torch.tensor(meta_label))
        else:
            return audio_image, torch.tensor(label)

    def __len__(self):
        return len(self.data_glob)
