import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json

class KEMDy20(Dataset):
    def __init__(self, path, text_feat_path, wav_feat_path):
        with open(path,'r') as f:
            self.preprocessed_data = json.load(f) #전처리된 데이터를 불러옴
        with open(text_feat_path, 'r') as f:
            KoBERT_output = json.load(f) #사전 학습된 KoBERT를 거친 output을 불러옴
        with open(wav_feat_path, 'r') as f:
            wav_output = json.load(f) #사전 학습된 Wav2Vec2를 거친 output을 불러옴
            
        # Normalize EDA / IBI / TEMP
        for idx, item in enumerate(self.preprocessed_data):
            self.preprocessed_data[idx]['KoBERT'] = torch.tensor(KoBERT_output[item['Filename']][0]).unsqueeze(0)
            self.preprocessed_data[idx]['Wav2Vec2'] = torch.tensor(wav_output[item['Filename']]).unsqueeze(0)
            eda_mean, eda_std = torch.mean(torch.tensor(item['EDA'])), torch.std(torch.tensor(item['EDA']))
            if len(item['EDA']) == 1: eda_std=1
            self.preprocessed_data[idx]['EDA'] = (torch.tensor(item['EDA']) - eda_mean) / (eda_std+1e-9)
            ibi_mean, ibi_std = torch.mean(torch.tensor(item['IBI'])), torch.std(torch.tensor(item['IBI']))
            if len(item['IBI']) == 1: ibi_std=1
            self.preprocessed_data[idx]['IBI'] = (torch.tensor(item['IBI']) - ibi_mean) / (ibi_std+1e-9)
            temp_mean, temp_std = torch.mean(torch.tensor(item['TEMP'])), torch.std(torch.tensor(item['TEMP']))
            if len(item['TEMP']) == 1: temp_std=1
            self.preprocessed_data[idx]['TEMP'] = (torch.tensor(item['TEMP']) - temp_mean) / (temp_std+1e-9)
            
    def __len__(self):
        return len(self.preprocessed_data)
    
    def __getitem__(self, index):
        return self.preprocessed_data[index]
    
    def collate_fn(self, batch):
        class_to_label = {'neutral':0,'angry':1,'disqust':2,'fear':3,'happy':4,'sad':5,'surprise':6} #라벨을 숫자로 변환
        text = torch.cat([data['KoBERT'] for data in batch], dim=0)
        audio = torch.cat([data['Wav2Vec2'] for data in batch], dim=0)
        eda = pad_sequence([data['EDA'] for data in batch], batch_first=True).unsqueeze(-1)
        ibi = pad_sequence([data['IBI'] for data in batch], batch_first=True).unsqueeze(-1)
        temp = pad_sequence([data['TEMP'] for data in batch], batch_first=True).unsqueeze(-1)
        if eda.shape[1] == 0:
            eda = torch.zeros([eda.shape[0], 1, eda.shape[2]])
        if ibi.shape[1] == 0:
            ibi = torch.zeros([ibi.shape[0], 1, ibi.shape[2]])
        if temp.shape[1] == 0:
            temp = torch.zeros([temp.shape[0], 1, temp.shape[2]])
        is_neutral = [1 if data['Emotion']=='neutral' else 0 for data in batch] #Fast-Thinking classifier를 위해 
        emotion = [data['Emotion'] for data in batch]
        label = [class_to_label[data['Emotion']] for data in batch]
        return {'Text': text, 'Audio':audio, 'EDA':eda, 'IBI':ibi, 'TEMP':temp, 'Is_neutral':is_neutral, 'Emotion':emotion,'label':label}