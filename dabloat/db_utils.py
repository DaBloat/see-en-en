import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import librosa
import numpy as np
import emoji

class CNNConfig(PretrainedConfig):
    model_type = "cnn"

    def __init__(self, input_length=512, num_labels=3, **kwargs):
        super().__init__(**kwargs)
        self.input_length = input_length
        self.num_labels = num_labels

class CNNForSequenceClassification(PreTrainedModel):
    config_class = CNNConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.conv1 = nn.Conv1d(1, 512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)

        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)

        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.drop4 = nn.Dropout(0.2)

        self.conv5 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(128)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.drop5 = nn.Dropout(0.2)

        # Compute flattened output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, config.input_length)
            x = self._forward_features(dummy)
            self.flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, config.num_labels)

        self.post_init()

    def _forward_features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.drop4(x)
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        x = self.drop5(x)
        return x

    def forward(self, input_values=None, labels=None):
        x = input_values.unsqueeze(1)  # (B, 1, T)
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn_fc1(self.fc1(x)))
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
    
    
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"input_values": self.X[idx], "labels": self.y[idx]}


class DaBloatCNNPipeline:
    def __init__(self):
        self.sentiment = {0:f"Negative {emoji.emojize('\N{unamused face}')}", 1:f"Neutral {emoji.emojize('\N{neutral face}')}", 2:f"Positive  {emoji.emojize('\N{smiling face with smiling eyes}')}"}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frame_length = 2048
        self.hop_length = 512
        self.training_args = TrainingArguments(
                            output_dir="./cnn_eval_output",
                            per_device_eval_batch_size=32,
                            dataloader_drop_last=False,
                            do_train=False,
                            do_eval=True,
                            report_to="none"
                        )
        self.config = CNNConfig.from_pretrained('dabloat-cnn-emotion-aug')
        self.weights = torch.load('./dabloat-cnn-emotion-aug.pth', map_location=self.device)
        self.model = CNNForSequenceClassification(config=self.config).to(self.device)
        self.model.load_state_dict(self.weights)
        self.predictor = Trainer(
                model=self.model,
                args=self.training_args,
        )
        
    def zero_crossing_rate(self, data, fr_len, hop_len):
        zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=fr_len, hop_length=hop_len)
        return np.squeeze(zcr)
    
    def root_mean_square(self, data, fr_len, hop_len):
        rms = librosa.feature.rms(y=data, frame_length=fr_len, hop_length=hop_len)
        return np.squeeze(rms)
    
    def mfcc(self, data, sr, flatten=True):
        mfcc = librosa.feature.mfcc(y=data, sr=sr)
        return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)
    
    def extract(self, data,sr=16000, fr_len=2048, hop_len=512):
        res = np.array([])
        result = np.hstack((
            res,
            self.zero_crossing_rate(data, fr_len, hop_len),
            self.root_mean_square(data, fr_len, hop_len),
            self.mfcc(data, sr)
        ))
        return result
    
    def process_test(self, path):
        data, sr = librosa.load(path, duration=2.5, offset=0.6)
        feat = self.extract(data)
        return np.pad(feat, (0, 2376 - feat.shape[0]), 'constant')
    
    def predict_sentiment(self, path):
        aud_data = [self.process_test(path)]
        dummy_emotion = [0]
        data_pred = EmotionDataset(aud_data, dummy_emotion)
        pred = self.predictor.predict(data_pred)
        summary = pred.predictions
        logits = torch.Tensor(summary)
        return {'prediction':summary, 'sentiment':self.sentiment[np.argmax(summary)], 'confidence':F.softmax(logits, dim=-1).numpy().tolist()[0]}
