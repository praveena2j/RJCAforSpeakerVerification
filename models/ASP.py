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
        self.attention = nn.Sequential(
            nn.Conv1d(dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, dim, kernel_size=1),
            nn.Softmax(dim=2),
            )

    def forward(self, x):
        """Computes Attentive Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim*2)
        """
        x = x.permute(0, 2, 1)
        #t = x.size()[-1]
        #global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        w = self.attention(x)

        mu = torch.sum(x * w, dim=2)

        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)

        return x