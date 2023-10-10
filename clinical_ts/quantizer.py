__all__ = ['GumbelQuantize']

import torch
from torch import nn, einsum
import torch.nn.functional as F

#from https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/quantize.py
class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144

    num_hiddens: feature dimension of the incoming data
    n_embed: number of embedding tokens/clusters
    embedding_dim: embedding dimension
    straight_through: whether to use straight through estimator
    data_dim: dimensionality of the input data 1d/2d

    temperature and kld_scale can be adjusted via corresponding attributes
    """
    def __init__(self, num_hiddens, n_embed, embedding_dim, straight_through=False, data_dim=1):
        super().__init__()
        self.data_dim = data_dim

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        self.proj = nn.Conv1d(num_hiddens, n_embed, 1) if data_dim==1 else nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z):
        '''
        input:
        z: input data with shape (batch, num_hiddens, height, width) for data_dim=2 (input batch, num_hiddens, timesteps) for data_dim=1
        output:
        z_q: weighted sum of embedding vectors (batch, embedding_dim, height, width) or (batch, embedding_dim, timesteps) approaches quantized value in the limit of temperature goes to zero
        diff: kl divergence to the uniform prior (to be added to the loss)
        ind: picked indices (argmax) of shape b h w/ b t (integers between 0 and n_embed)
        '''
        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True

        logits = self.proj(z) # b n h w/ b n t (n=n_embed)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard) # b n h w/ b n t probability distribution (dim=1)
    
        if(self.data_dim == 1):
            z_q = einsum('b n t, n d -> b d t', soft_one_hot, self.embed.weight) #b d t weighted sum of embedding vectors (according to prob distribution)
        elif(self.data_dim == 2):
            z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight) #b d h w

        # + kl divergence to the prior loss  (uniform 1/n_embed)
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()    
        #ind = soft_one_hot.argmax(dim=1)
        return z_q, diff, soft_one_hot #ind
