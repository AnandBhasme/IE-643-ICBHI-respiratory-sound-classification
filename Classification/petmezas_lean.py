#!/usr/bin/env python3
"""Paper-informed Petmezas CNN-LSTM on the official split, lean fixed-epoch protocol."""
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class CycleSpectrograms(Dataset):
    def __init__(self, rows, wav_root):
        self.rows, self.wav_root = rows, Path(wav_root)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        wav, sr = sf.read(self.wav_root / row['filename'], dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 4000:
            from math import gcd
            d = gcd(int(sr), 4000)
            wav = resample_poly(wav, 4000 // d, int(sr) // d).astype(np.float32)
        target = 10800
        if len(wav) >= target:
            wav = wav[:target]
        else:
            wav = np.resize(wav, target).astype(np.float32)
        x = torch.from_numpy(wav.copy())
        spec = torch.stft(x, n_fft=256, hop_length=144, win_length=256,
                          window=torch.hann_window(256), center=True,
                          return_complex=True).abs().square()[:129]
        spec = torch.log10(spec + 1e-10)
        spec = F.interpolate(spec[None, None], size=(50, 75), mode='bilinear', align_corners=False)[0]
        lo, hi = spec.amin(), spec.amax()
        spec = (spec - lo) / (hi - lo).clamp_min(1e-8)
        image = spec.expand(3, -1, -1).contiguous()
        return image, int(row['label'])


class PetmezasCnnLstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(.2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(.2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout2d(.2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(54 * 128, 1024)
        self.drop4 = nn.Dropout(.2)
        self.fc_out = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.bn1(self.conv2(x)))))
        x = self.drop3(self.pool3(self.bn2(F.relu(self.conv4(F.relu(self.conv3(x)))))))
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x, _ = self.lstm(x)
        x = x.reshape(b, -1)
        return self.fc_out(self.drop4(F.relu(self.fc1(x))))


def load_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def confusion_metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    supports = cm.sum(axis=1)
    se = cm[1:, 1:].trace() / cm[1:, :].sum()
    sp = cm[0, 0] / supports[0]
    acc = np.trace(cm) / cm.sum()
    micro_sp = (2 + acc) / 3
    return {
        'official': {'Se': 100*se, 'Sp': 100*sp, 'Score': 50*(se+sp)},
        'petmezas_style_micro': {'Se': 100*acc, 'Sp': 100*micro_sp,
                                 'Score': 50*(acc+micro_sp), 'Acc': 100*(1+acc)/2},
        'accuracy': 100*acc, 'class_recall': (100*np.diag(cm)/supports).tolist(),
        'confusion': cm.tolist(), 'support': supports.tolist(),
    }


def predict(model, loader, device, collect_loss=False):
    model.eval(); all_y=[]; all_prob=[]; losses=[]
    with torch.no_grad():
        for x, y in loader:
            x=x.to(device); y=y.to(device)
            logits=model(x)
            all_y.extend(y.cpu().numpy().tolist())
            all_prob.extend(torch.softmax(logits, -1).cpu().numpy().tolist())
            if collect_loss:
                losses.append(F.cross_entropy(logits, y, reduction='sum').item())
    probs=np.asarray(all_prob, dtype=np.float32); labels=np.asarray(all_y, dtype=np.int64)
    preds=probs.argmax(1)
    cm=np.bincount(labels*4+preds, minlength=16).reshape(4,4)
    return (float(sum(losses)/len(labels)) if collect_loss else None), confusion_metrics(cm), probs


def train_seed(args, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
    split=Path(args.split_dir)
    train_all=load_rows(split/'train_manifest.csv')
    train_rows=train_all
    test_rows=load_rows(split/'test_manifest.csv')
    root=Path(args.data_root)
    train_loader=DataLoader(CycleSpectrograms(train_rows,root/'training/real'),batch_size=32,shuffle=True,
                            num_workers=args.workers,pin_memory=True,drop_last=False)
    test_loader=DataLoader(CycleSpectrograms(test_rows,root/'test/real'),batch_size=64,shuffle=False,
                           num_workers=args.workers,pin_memory=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=PetmezasCnnLstm().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    out=Path(args.output_root)/f'petmezas_lean_seed{seed}'
    out.mkdir(parents=True,exist_ok=True)
    config={
        'method':'petmezas_cnn_lstm','train_real_only':True,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
        'paper_specified': {'resample_hz':4000,'cycle_seconds':2.7,'image_hw':[50,75],
           'conv_layers':4,'pool_layers':3,'batch_norm_layers':2,'dropout':.2,
           'lstm_layers':1,'lstm_hidden':128,'dense_hidden':1024,'lr':1e-4,'focal_gamma':2.0},
        'assumptions': {'stft_n_fft':256,'stft_hop':144,'window':'Hann','image':'log10 power, per-image min-max, replicated grayscale RGB',
           'short_cycle_padding':'repeat samples','conv_kernel':'3x3 same padding','optimizer':'Adam',
           'batch_size':32,'epochs':args.epochs,'schedule':'fixed; no validation split or early stopping',
           'test_use':'reporting only; last epoch is headline; best test epoch is secondary'},
        'seed':seed,'train_cycles':len(train_rows),'validation_cycles':0,'test_cycles':len(test_rows),
    }
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    best_score=-float('inf'); best_epoch=0; best_path=out/'best_test_secondary.pth'; history=[]
    metric_fields=['epoch','eval_split','is_best','train_loss','train_acc','test_loss','test_acc',
                   'Se','Sp','Score','Se_micro','Sp_micro','Score_micro','accuracy_micro',
                   'se_binary','accuracy','recall_normal','recall_crackle','recall_wheeze','recall_both',
                   'epoch_seconds','train_domain_acc']+[f'cm_{i}{j}' for i in range(4) for j in range(4)]
    t0=time.time()
    for epoch in range(1,args.epochs+1):
        epoch_start=time.time()
        model.train(); running=0.0; n=0
        for x,y in train_loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x); logp=F.log_softmax(logits,dim=-1); lp=logp.gather(1,y[:,None]).squeeze(1)
            loss=(-(1-lp.exp()).pow(2)*lp).mean()
            loss.backward(); optimizer.step(); running += loss.item()*len(y); n += len(y)
        test_loss,test_metrics,_=predict(model,test_loader,device,collect_loss=True)
        row={'epoch':epoch,'eval_split':'official_test','is_best':False,'train_loss':running/n,
             'train_acc':'','test_loss':test_loss,'test_acc':test_metrics['accuracy'],
             'Se':test_metrics['official']['Se'],'Sp':test_metrics['official']['Sp'],
             'Score':test_metrics['official']['Score'],
             'Se_micro':test_metrics['petmezas_style_micro']['Se'],
             'Sp_micro':test_metrics['petmezas_style_micro']['Sp'],
             'Score_micro':test_metrics['petmezas_style_micro']['Score'],
             'accuracy_micro':test_metrics['petmezas_style_micro']['Acc'],
             'se_binary':100.0*float(np.asarray(test_metrics['confusion'])[1:,1:].sum()/np.asarray(test_metrics['confusion'])[1:,:].sum()),
             'accuracy':test_metrics['accuracy'],'epoch_seconds':time.time()-epoch_start,'train_domain_acc':''}
        row.update({f'recall_{name}':value for name,value in zip(
            ['normal','crackle','wheeze','both'],test_metrics['class_recall'])})
        row.update({f'cm_{i}{j}':test_metrics['confusion'][i][j] for i in range(4) for j in range(4)})
        history.append(row)
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=metric_fields); w.writeheader(); w.writerows(history)
        torch.save({'model':model.state_dict(),'epoch':epoch},out/'last.pth')
        print(f'seed={seed} epoch={epoch} train_loss={running/n:.5f} test_loss={test_loss:.5f} '
              f"official={test_metrics['official']['Se']:.2f}/{test_metrics['official']['Sp']:.2f}/{test_metrics['official']['Score']:.2f}; "
              f"micro={test_metrics['petmezas_style_micro']['Se']:.2f}/{test_metrics['petmezas_style_micro']['Sp']:.2f}/"
              f"{test_metrics['petmezas_style_micro']['Score']:.2f}",flush=True)
        if test_metrics['official']['Score'] > best_score:
            best_score=test_metrics['official']['Score']; best_epoch=epoch
            row['is_best']=True
            history[-1]['is_best']=True
            with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=metric_fields); w.writeheader(); w.writerows(history)
            torch.save({'model':model.state_dict(),'epoch':epoch},best_path)
    test_report={'seed':seed,'epochs_trained':epoch,'best_epoch_by_test_score_secondary':best_epoch,
                 'elapsed_seconds':time.time()-t0,'headline_checkpoint':'last epoch',
                 'selection_note':'best-test epoch is secondary reporting only; it did not affect training'}
    for name,ckpt_path in [('best_test_secondary',best_path),('last',out/'last.pth')]:
        ckpt=torch.load(ckpt_path,map_location=device,weights_only=False)
        model.load_state_dict(ckpt['model'])
        _,metrics,probs=predict(model,test_loader,device)
        test_report[name]={'checkpoint_epoch':int(ckpt['epoch']),**metrics}
        np.save(out/f'probs_test_{name}.npy',probs)
        np.savetxt(out/f'confusion_test_{name}.csv',metrics['confusion'],fmt='%d',delimiter=',')
    (out/'official_test_final.json').write_text(json.dumps(test_report,indent=2)+'\n')
    print(f"seed={seed} completed fixed epochs={epoch}; last official "
          f"Se/Sp/Score={test_report['last']['official']['Se']:.2f}/"
          f"{test_report['last']['official']['Sp']:.2f}/"
          f"{test_report['last']['official']['Score']:.2f}",flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--split-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--epochs',type=int,default=40)
    a=p.parse_args()
    train_seed(a,a.seed)
