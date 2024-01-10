import torch
import torch.nn as nn
from .layer import LSTM


class PredictLayer(nn.Module):

    def __init__(self, dim1, dim2, num_ans, dropout):
        super(PredictLayer, self).__init__()
        self.Joint = LSTM(dim1 + dim2, dim1, 2, dropout=0, residual_embeddings=True)
        self.predict = nn.Sequential(
            nn.Linear(dim1, dim1 // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(dim1 // 4, num_ans),
        )

    def forward(self, data1, data2):
        joint = torch.cat((data1, data2), dim=-1)
        joint = self.Joint(joint)
        return torch.softmax(self.predict(joint).squeeze(), dim=-1)