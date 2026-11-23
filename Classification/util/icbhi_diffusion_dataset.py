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
            
            
        # ------------------------------------------------------------------
        # Novelty phase: extra training items listed in a CSV manifest.
        #   kind=synth    -> a generated log-mel (already in AST-normalised
        #                    units) read from row `index` of the .npy pool `source`
        #   kind=real_dup -> a duplicate of the real training cycle `source`
        #                    (classical oversampling control)
        self.n_base_items = len(self.audio_images)
        extra_manifest = getattr(self.args, 'extra_manifest', None)
        if self.train_flag and extra_manifest:
            if self.view != 'mel' or self.args.raw_augment:
                raise ValueError('--extra_manifest requires --view mel and --raw_augment 0')
            import csv as _csv
            with open(extra_manifest, newline='', encoding='utf-8') as f:
                extra_rows = list(_csv.DictReader(f))
            name_to_item = {os.path.basename(path): item for path, item in zip(self.data_glob, self.audio_images)}
            pools = {}
            teachers = {}
            # item index -> soft class target (only for synthetic rows with a 'teacher' column)
            self.soft_targets = {}
            n_synth = n_dup = 0
            for row in extra_rows:
                label = int(row['label'])
                if row['kind'] == 'synth':
                    if '/diag/' in row['source']:
                        raise AssertionError('Diagnostic-only pools must never enter training')
                    if row['source'] not in pools:
                        pools[row['source']] = np.load(row['source'], mmap_mode='r')
                    image = np.asarray(pools[row['source']][int(row['index'])], dtype=np.float32)[..., None]
                    if image.shape != (498, 128, 1):
                        raise ValueError(f'Unexpected synthetic log-mel shape {image.shape}')
                    meta_label = get_meta_infor(None, 'gen', self.args)
                    item = ([image], label) if self.args.meta_mode == 'none' else ([image], label, meta_label)
                    n_synth += 1
                    if row.get('teacher'):
                        if row['teacher'] not in teachers:
                            teachers[row['teacher']] = np.load(row['teacher'])
                        self.soft_targets[len(self.audio_images)] = np.asarray(
                            teachers[row['teacher']][int(row['index'])], dtype=np.float32)
                elif row['kind'] == 'real_dup':
                    if row['source'] not in name_to_item:
                        raise KeyError(f"real_dup source {row['source']} is not in the active real training set")
                    item = name_to_item[row['source']]
                    if item[1] != label:
                        raise ValueError(f"Label mismatch for duplicated cycle {row['source']}")
                    n_dup += 1
                else:
                    raise ValueError(f"Unknown extra-manifest kind: {row['kind']}")
                self.audio_images.append(item)
                self.data_glob.append(f"extra::{row['kind']}::{row['source']}::{row.get('index', '')}")
            print(f'Extra training items from {extra_manifest}: synthetic={n_synth}, duplicated real={n_dup}')
            print('Training class counts with extras:',
                  np.bincount([item[1] for item in self.audio_images], minlength=4).tolist())

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
            if getattr(self.args, 'soft_extra', False):
                # [label, target_0..3]: hard one-hot for real items; for synthetic items the
                # teacher soft label mixed with the intended class by --soft_beta.
                target = np.zeros(4, dtype=np.float32)
                target[label] = 1.0
                soft = getattr(self, 'soft_targets', {}).get(index)
                if soft is not None:
                    beta = float(self.args.soft_beta)
                    target = (1.0 - beta) * target + beta * soft
                return audio_image, torch.from_numpy(np.concatenate([[np.float32(label)], target]).astype(np.float32))
            if self.args.meta_mode == 'none':
                return audio_image, torch.tensor(label)
            else:
                return audio_image, (torch.tensor(label), torch.tensor(meta_label))
        else:
            return audio_image, torch.tensor(label)

    def __len__(self):
        return len(self.data_glob)
