from torch import nn
import torch

class LSTM(nn.Module): #EDA, IBI, TEMP는 LSTM 모델을 거침
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.mlp = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.mlp(hidden[-1])
    
class Classifier(nn.Module): #분류 모델
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.classifier(x)

class FastThinking(nn.Module): #Fast-Thinking Clasifier 모델 
    def __init__(self, text_encoder, audio_encoder, feat_dim, hidden_dim, num_class):
        super(FastThinking, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.classifier = Classifier(self.feat_dim, self.hidden_dim, num_class)
        
    def forward(self, text, audio):
        text_feature = self.text_encoder(text) #text encoder: KoBERT
        audio_feature = self.audio_encoder(audio) #audio encoder : Wav2Vec
        
        feature_total = torch.cat([text_feature, audio_feature], dim=-1)
        return self.classifier(feature_total)

class SlowThinking(nn.Module): #Slow-Thinking Classifier 모델
    def __init__(self, text_encoder, audio_encoder, eda_encoder, ibi_encoder, temp_encoder, feat_dim, hidden_dim, num_class):
        super(SlowThinking, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.eda_encoder = eda_encoder
        self.ibi_encoder = ibi_encoder
        self.temp_encoder = temp_encoder
        self.multiclass_classifier = Classifier(self.feat_dim, self.hidden_dim, self.num_class)
        
    def forward(self, text, audio, eda, ibi, temp):
        text_feature = self.text_encoder(text)
        audio_feature = self.audio_encoder(audio)
        eda_feature = self.eda_encoder(eda)
        ibi_feature = self.ibi_encoder(ibi)
        temp_feature = self.temp_encoder(temp)
        
        feature_total = torch.cat([text_feature, audio_feature, eda_feature, ibi_feature, temp_feature], dim=-1)
        return self.multiclass_classifier(feature_total)
    