import os
import glob
import pandas as pd
import torchaudio
import torchaudio.transforms as T
import torch
import math
from tqdm import tqdm

class DummyArgs:
    def __init__(self):
        self.pad_types = 'repeat'
        self.desired_length = 5
        self.sample_rate = 16000

def _get_lungsound_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0:
        return 0
    elif crackle == 1 and wheeze == 0:
        return 1
    elif crackle == 0 and wheeze == 1:
        return 2
    elif crackle == 1 and wheeze == 1:
        return 3
    return 0

def cut_pad_sample_torchaudio(data, args):
    fade_samples_ratio = 16
    fade_samples = int(args.sample_rate / fade_samples_ratio)
    fade_out = T.Fade(fade_in_len=0, fade_out_len=fade_samples, fade_shape='linear')
    target_duration = args.desired_length * args.sample_rate

    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
        if data.dim() == 1:
            data = data.unsqueeze(0)
    else:
        if args.pad_types == 'zero':
            tmp = torch.zeros(1, target_duration, dtype=torch.float32)
            diff = target_duration - data.shape[-1]
            tmp[..., diff//2:data.shape[-1]+diff//2] = data
            data = tmp
        elif args.pad_types == 'repeat':
            ratio = math.ceil(target_duration / data.shape[-1])
            data = data.repeat(1, ratio)
            data = data[..., :target_duration]
            data = fade_out(data)
    
    return data

def main():
    args = DummyArgs()
    data_dir = r"F:\Workspace\Working\IE643_Project\Data\ICBHI_final_database\ICBHI_final_database"
    out_dir_train = r"F:\Workspace\Working\IE643_Project\Code\AFT\Classification\data\training\real"
    out_dir_test = r"F:\Workspace\Working\IE643_Project\Code\AFT\Classification\data\test\real"
    
    os.makedirs(out_dir_train, exist_ok=True)
    os.makedirs(out_dir_test, exist_ok=True)
    
    test_patients = [101, 103, 104, 106, 114, 118, 122, 124, 130, 133, 138, 139, 141, 145, 146, 147, 151, 154, 155, 156, 160, 162, 163, 167, 170, 172, 174, 176, 178, 179, 181, 186, 187, 188, 192, 193, 195, 198, 199, 200, 203, 205, 207, 211, 212, 213, 218, 224, 225]
    
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    
    for txt_path in tqdm(txt_files):
        fname = os.path.basename(txt_path).replace('.txt', '')
        if not fname[0].isdigit():
            continue
        
        patient_id = int(fname.split('_')[0])
        is_test = patient_id in test_patients
        out_dir = out_dir_test if is_test else out_dir_train
        
        ann = pd.read_csv(txt_path, sep='\t', names=['Start', 'End', 'Crackles', 'Wheezes'])
        wav_path = txt_path.replace('.txt', '.wav')
        
        import soundfile as sf
        data_np, sr = sf.read(wav_path)
        data = torch.from_numpy(data_np).float()
        if data.dim() == 1:
            data = data.unsqueeze(0) # shape (1, T)
        else:
            data = data.transpose(0, 1) # shape (C, T)
            
        if sr != args.sample_rate:
            resample = T.Resample(sr, args.sample_rate)
            data = resample(data)
            
        fade_samples_ratio = 16
        fade_samples = int(args.sample_rate / fade_samples_ratio)
        fade = T.Fade(fade_in_len=fade_samples, fade_out_len=fade_samples, fade_shape='linear')
        data = fade(data)
        
        for idx in ann.index:
            row = ann.loc[idx]
            start, end = row['Start'], row['End']
            
            start_ind = min(int(start * args.sample_rate), data.shape[1])
            end_ind = min(int(end * args.sample_rate), data.shape[1])
            audio_chunk = data[:, start_ind: end_ind]
            
            label = _get_lungsound_label(row['Crackles'], row['Wheezes'])
            
            padded_chunk = cut_pad_sample_torchaudio(audio_chunk, args)
            
            out_name = f"{fname}_event_{idx}_label_{label}.wav"
            out_path = os.path.join(out_dir, out_name)
            sf.write(out_path, padded_chunk.numpy().T, args.sample_rate)

if __name__ == '__main__':
    main()
