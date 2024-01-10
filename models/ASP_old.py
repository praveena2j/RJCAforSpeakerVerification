import torch
import torch.nn as nn
import torch.nn.functional as F
import sys



class Attentive_Statistics_Pooling(nn.Module):
    def __init__(self, dim):
        """ASP
        Paper: Attentive Statistics Pooling for Deep Speaker Embedding
        Link: https://arxiv.org/pdf/1803.10963.pdf
        Args:
            dim (pair): the size of attention weights
        """
        super(Attentive_Statistics_Pooling, self).__init__()
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.FloatTensor(dim, 1))

    def forward(self, x):
        """Computes Attentive Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim*2)
        """
        #x = x.permute(0, 2, 1)

        h = torch.tanh(self.sap_linear(x))
       
        w = torch.matmul(h, self.attention).squeeze(dim=2)
       
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
       
        mu = torch.sum(x * w, dim=1)
       
        rh = torch.sqrt( ( torch.sum((x**2) * w, dim=1) - mu**2 ).clamp(min=1e-5) )
        
        x = torch.cat((mu, rh), 1)
       
        return x