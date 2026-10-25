import os
import glob
import pandas as pd
import torch
import math
import torchaudio.transforms as T
from tqdm import tqdm
import soundfile as sf

def _get_lungsound_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0: return 0
    elif crackle == 1 and wheeze == 0: return 1
    elif crackle == 0 and wheeze == 1: return 2
    elif crackle == 1 and wheeze == 1: return 3
    return 0

def cut_pad_sample(data, sr, target_length=4):
    target_duration = target_length * sr
    if data.shape[-1] > target_duration:
        data = data[..., :target_duration]
    else:
        ratio = math.ceil(target_duration / data.shape[-1])
        data = data.repeat(1, ratio)[..., :target_duration]
    return data

def main():
    data_dir = r"F:\Workspace\Working\IE643_Project\Data\ICBHI_final_database\ICBHI_final_database"
    out_dir = r"F:\Workspace\Working\IE643_Project\Code\AFT\Diffwave\dataset\label3"
    os.makedirs(out_dir, exist_ok=True)
    
    # We only need the "both" class from the training set
    test_patients = [101, 103, 104, 106, 114, 118, 122, 124, 130, 133, 138, 139, 141, 145, 146, 147, 151, 154, 155, 156, 160, 162, 163, 167, 170, 172, 174, 176, 178, 179, 181, 186, 187, 188, 192, 193, 195, 198, 199, 200, 203, 205, 207, 211, 212, 213, 218, 224, 225]
    
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    
    for txt_path in tqdm(txt_files):
        fname = os.path.basename(txt_path).replace('.txt', '')
        if not fname[0].isdigit(): continue
        
        patient_id = int(fname.split('_')[0])
        if patient_id in test_patients:
            continue # We only generate synthetic data using training data!
            
        ann = pd.read_csv(txt_path, sep='\t', names=['Start', 'End', 'Crackles', 'Wheezes'])
        
        # Check if there's any 'both' label before loading audio
        has_both = False
        for _, row in ann.iterrows():
            if _get_lungsound_label(row['Crackles'], row['Wheezes']) == 3:
                has_both = True
                break
        if not has_both: continue
        
        wav_path = txt_path.replace('.txt', '.wav')
        data_np, sr = sf.read(wav_path)
        data = torch.from_numpy(data_np).float()
        if data.dim() == 1: data = data.unsqueeze(0)
        else: data = data.transpose(0, 1)
            
        if sr != 16000:
            resample = T.Resample(sr, 16000)
            data = resample(data)
        
        for idx in ann.index:
            row = ann.loc[idx]
            label = _get_lungsound_label(row['Crackles'], row['Wheezes'])
            if label == 3:
                start, end = row['Start'], row['End']
                start_ind = min(int(start * 16000), data.shape[1])
                end_ind = min(int(end * 16000), data.shape[1])
                chunk = data[:, start_ind: end_ind]
                padded = cut_pad_sample(chunk, 16000, target_length=4)
                
                out_name = f"{fname}_event_{idx}_label_3.wav"
                sf.write(os.path.join(out_dir, out_name), padded.numpy().T, 16000)

if __name__ == '__main__':
    main()
