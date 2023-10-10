__all__ = ['RNNEncoder', 'RNNEncoderConfig', 'RNNPredictor', 'RNNPredictorConfig', 'RNNHead', 'RNNHeadConfig']

import torch
import torch.nn as nn
from clinical_ts.basic_conv1d import _conv1d
import numpy as np

from collections.abc import Iterable
from .basic_conv1d import bn_drop_lin
from .cpc_template import EncoderBase, PredictorBase, HeadBase, EncoderBaseConfig, PredictorBaseConfig, HeadBaseConfig

from dataclasses import dataclass, field
from typing import List,Any

class RNNEncoder(EncoderBase):
    def __init__(self, hparams):
        '''RNN Encoder is actually just a conv encoder'''
        super().__init__(hparams)
        assert(len(hparams.encoder.strides)==len(hparams.encoder.kss) and len(hparams.encoder.strides)==len(hparams.encoder.features) and len(hparams.encoder.strides)==len(hparams.encoder.dilations))
        lst = []
        for i,(s,k,f,d) in enumerate(zip(hparams.encoder.strides,hparams.encoder.kss,hparams.encoder.features,hparams.encoder.dilations)):
            lst.append(_conv1d(hparams.base.input_channels if i==0 else hparams.encoder.features[i-1],f,kernel_size=k,stride=s,dilation=d,bn=hparams.encoder.normalization,layer_norm=hparams.encoder.layer_norm))
            if(hparams.encoder.multi_prediction and i==0):#local pool after first conv
                if(hparams.encoder.local_pool_max):
                    lst.append(torch.nn.MaxPool1d(kernel_size=hparams.encoder.local_pool_kernel_size,stride=hparams.encoder.local_pool_stride if hparams.encoder.local_pool_stride!=0 else hparams.encoder.local_pool_kernel_size,padding=(hparams.encoder.local_pool_kernel_size-1)//2))
                else:
                    lst.append(torch.nn.AvgPool1d(kernel_size=hparams.encoder.local_pool_kernel_size,stride=hparams.encoder.local_pool_stride if hparams.encoder.local_pool_stride!=0 else hparams.encoder.local_pool_kernel_size,padding=(hparams.encoder.local_pool_kernel_size-1)//2))        
        
        self.layers = nn.Sequential(*lst)
        self.downsampling_factor = (hparams.encoder.local_pool_stride if hparams.encoder.multi_prediction else 1)*np.prod(hparams.encoder.strides)
        
        self.timesteps_per_token = hparams.encoder.timesteps_per_token
        self.output_dim = hparams.encoder.features[-1]

    def forward(self, input, static=None):
        if(self.timesteps_per_token > 1):#patches a la vision transformer
            assert(input.size(2)%self.timesteps_per_token==0)
            size = input.size()
            input = input.transpose(1,2).reshape(size[0],size[2]//self.timesteps_per_token,-1).transpose(1,2) # output: bs, output_dim, seq//downsampling_factor
        input_encoded = self.layers(input).transpose(1,2)
        return input_encoded#bs,seq,feat
    
    def get_output_dim(self):
        return self.output_dim

    def get_output_length(self,length):
        return int(length//self.downsampling_factor+ (1 if length%self.downsampling_factor>0 else 0))

@dataclass
class RNNEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.cpc_rnn.RNNEncoder"

    #local pool after first conv
    multi_prediction:bool = False #local_pool named like this for consistency with MLP heads etc
    local_pool_max:bool = False
    local_pool_kernel_size: int = 0
    local_pool_stride: int = 0 #kernel_size if 0
    
    strides:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder strides (space-separated)")
    kss:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder kernel sizes (space-separated)")
    features:List[int]=field(default_factory=lambda: [512,512,512,512]) #help="encoder features (space-separated)")
    dilations:List[int]=field(default_factory=lambda: [1,1,1,1]) #help="encoder dilations (space-separated)")
    normalization:bool=True #help="disable encoder batch/layer normalization")
    layer_norm:bool=False#", action="store_true", help="encoder layer normalization")
    

class RNNPredictor(PredictorBase):
    def __init__(self, hparams, encoder_output_dim, encoder_output_length, encoder_static_output_dim):
        super().__init__(hparams, encoder_output_dim, encoder_output_length, encoder_static_output_dim)
        rnn_arch = nn.LSTM if not(hparams.predictor.gru) else nn.GRU
        self.rnn = rnn_arch(encoder_output_dim,hparams.predictor.model_dim//2 if not(hparams.predictor.causal) else hparams.predictor.model_dim,num_layers=hparams.predictor.n_layers,batch_first=True,bidirectional=not(hparams.predictor.causal))

        self.d_hidden = 2*hparams.predictor.n_layers if not(hparams.predictor.causal) else hparams.predictor.n_layers

        if(encoder_static_output_dim>0 and  hparams.predictor.static_input):
            self.mlp1 = nn.Sequential(nn.Linear(encoder_static_output_dim,hparams.predictor.model_dim*hparams.predictor.n_layers),nn.ReLU(inplace=True))
            self.mlp2 = nn.Sequential(nn.Linear(encoder_static_output_dim,hparams.predictor.model_dim*hparams.predictor.n_layers),nn.ReLU(inplace=True))
        self.static_input = hparams.predictor.static_input

    def forward(self, seq, static=None):
        if(static is not None and self.static_input):
            output_shape = (self.d_hidden,static.shape[0],-1)
            out, _ = self.rnn(seq,(self.mlp1(static).view(output_shape),self.mlp2(static).view(output_shape)))
        else:
            out, _ = self.rnn(seq)
        return out

@dataclass
class RNNPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.cpc_rnn.RNNPredictor"
    model_dim:int = 512 
    gru:bool=False # help="use GRU instead of LSTM")
    n_layers:int = 2 # help="number of RNN layers")
    static_input:bool=True #"do not use static information (if available) for initializing hidden states")

class RNNHead(HeadBase):
    def __init__(self, hparams, target_dim, encoder_output_length, encoder_static_output_dim):
        super().__init__(hparams, target_dim, encoder_output_length, encoder_static_output_dim)
        self.batch_first = hparams.head.batch_first
        self.local_pool = hparams.head.multi_prediction
        
        if(self.local_pool):#local pool
            self.local_pool_padding = (hparams.head.local_pool_kernel_size-1)//2
            self.local_pool_kernel_size = hparams.head.local_pool_kernel_size
            self.local_pool_stride = hparams.head.local_pool_stride
            
            if(hparams.head.local_pool_max):
                self.pool = torch.nn.MaxPool1d(kernel_size=hparams.head.local_pool_kernel_size,stride=hparams.head.local_pool_stride if hparams.head.local_pool_stride!=0 else hparams.head.local_pool_kernel_size,padding=(hparams.head.local_pool_kernel_size-1)//2)
            else:
                self.pool = torch.nn.AvgPool1d(kernel_size=hparams.head.local_pool_kernel_size,stride=hparams.head.local_pool_stride if hparams.head.local_pool_stride!=0 else hparams.head.local_pool_kernel_size,padding=(hparams.head.local_pool_kernel_size-1)//2)        
        else:#global pool
            if(hparams.head.concat_pool):
                self.pool = AdaptiveConcatPoolRNN(bidirectional=not(hparams.predictor.causal),cls_first= False)
            else:
                self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten()) #None

        #classifier
        output_dim =hparams.predictor.model_dim
                
        nf = 3*output_dim if (hparams.head.multi_prediction is False and hparams.head.concat_pool) else output_dim

        #concatenate static input
        self.static_input = hparams.head.static_input
        if(self.static_input):
            nf = nf + encoder_static_output_dim
                
        lin_ftrs = [nf, target_dim] if hparams.head.lin_ftrs is None else [nf] + hparams.head.lin_ftrs + [target_dim]
        ps_head = [hparams.head.dropout] if not isinstance(hparams.head.dropout, Iterable) else hparams.head.dropout
        if len(ps_head)==1:
            ps_head = [ps_head[0]/2] * (len(lin_ftrs)-2) + ps_head
        actns = [nn.ReLU(inplace=False)] * (len(lin_ftrs)-2) + [None]

        layers_head =[]
        for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps_head,actns):
            layers_head+=bn_drop_lin(ni,no,hparams.head.batch_norm,p,actn,layer_norm=False,permute=self.local_pool)
        self.head=nn.Sequential(*layers_head)

    def forward(self,seq,static=None):
        if(self.batch_first):#B,S,E
            seq = seq.transpose(1,2) 
        else:#S,B,E
            seq = seq.transpose(0,1).transpose(1,2)
        seq = self.pool(seq) if self.pool is not None else seq[:,:,-1] #local_pool: B, E, S global_pool: B, E
        if(self.local_pool):
            seq = seq.transpose(1,2)#B,S,E for local_pool
        if(static is not None and self.static_input):
            if(self.local_pool):
                seq = torch.cat([seq,static],dim=1)
            else:
                seq = torch.cat([seq,static.unsqueeze(1).repeat(1,seq.shape[1],1)],dim=1)
        seq = self.head(seq) #B,S,Nc for local_pool B,Nc for global_pool
        return seq
    
    def get_output_length(self,length):
        if(self.local_pool):
            return int(np.floor((length + 2*self.local_pool_padding- (self.local_pool_kernel_size-1)+1)/self.local_pool_stride))
        else:
            return 1


@dataclass
class RNNHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.cpc_rnn.RNNHead"
    batch_first: bool = True

    multi_prediction:bool = False
    local_pool_max:bool = False
    local_pool_kernel_size: int = 0
    local_pool_stride: int = 0 #kernel_size if 0
    #local_pool_padding= (kernel_size-1)//2

    concat_pool:bool = True
    dropout:float=0.5
    lin_ftrs:List[int]=field(default_factory=lambda: []) #help="Classification head hidden units (space-separated)")
    batch_norm:bool = True
    static_input:bool = True

#copied from RNN1d
class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self, bidirectional=False, cls_first=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.cls_first = cls_first

    def forward(self,x):
        #input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)

        if(self.bidirectional is False):
            if(self.cls_first):
                t3 = x[:,:,0]
            else:
                t3 = x[:,:,-1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:,:channels,-1],x[:,channels:,0]],1)
        out=torch.cat([t1.squeeze(-1),t2.squeeze(-1),t3],1) #output shape bs, 3*ch
        return out