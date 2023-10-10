__all__ = ['SpecConv2dEncoder','SpecConv2dEncoderConfig','SpecConv1dEncoder','SpecConv1dEncoderConfig']

from .cpc_template import EncoderBase, EncoderBaseConfig
from dataclasses import dataclass, field
from typing import List
import torch.nn as nn
import torch

class SpecConv2dEncoder(EncoderBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        layers = []
        for s,e in zip([hparams.base.input_channels]+hparams.encoder.features[:-1],hparams.encoder.features):
            layers.append(nn.Conv2d(s,e,kernel_size=3,padding=1))
            layers.append(nn.GELU())

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hparams.encoder.features[-1]

    def forward(self, inp, static=None):
        #input shape (bs, ch, freq, ts)
        out = self.encoder(inp)
        out = torch.mean(out,dim=2) # bs, ch, ts
        return out.transpose(1,2)# bs, ts, ch
    
    def get_output_length(self,length):
        return length

    def get_output_dim(self):
        return self.output_dim

@dataclass
class SpecConv2dEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.cpc_spec.SpecConv2dEncoder"
    features:List[int] = field(default_factory=lambda: [128,512])


class SpecConv1dEncoder(EncoderBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        layers = []
        for s,e in zip([hparams.base.input_channels*hparams.base.freq_bins]+hparams.encoder.features[:-1],hparams.encoder.features):
            layers.append(nn.Conv1d(s,e,kernel_size=3,padding=1,groups=hparams.base.input_channels if hparams.encoder.grouped else 1))
            layers.append(nn.GELU())

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hparams.encoder.features[-1]
        self.epoch_length = hparams.encoder.epoch_length
        self.epoch_stride = hparams.encoder.epoch_length if hparams.encoder.epoch_stride==0 else hparams.encoder.epoch_stride

    def forward(self, inp, static=None):
        #input shape (bs, ch, freq, ts)-> (bs, ch*freq, ts)
        assert inp.shape[3]>=self.epoch_length, "epoch_length should be less or equal to input_size should be same"#without copying
        epochs = 1+(inp.shape[3]-self.epoch_length)//self.epoch_stride

        assert self.epoch_length==self.epoch_stride, "epoch_length and epoch_stride should be same"#without copying

        x = inp[:,:,:,:self.epoch_length+(epochs-1)*self.epoch_stride].view(inp.shape[0],inp.shape[1],inp.shape[2],-1,self.epoch_length)#bs,channels,freq, epochs,epoch_length
        x = x.permute(0,3,1,2,4) # bs, epoch, ch, freq, epoch_lenth
        x = x.reshape(inp.shape[0]*epochs,-1,self.epoch_length) #bs*epochs,channels*freq, epoch_length

        out = self.encoder(x) # bs*epochs, features, epoch_lenth
        out = out.reshape(inp.shape[0],epochs, out.shape[1], out.shape[-1]).permute(0,1,3,2) #bs,features, epochs, epoch_lenth

        return out.reshape(out.shape[0], epochs*self.epoch_length, out.shape[-1]) # bs, ts, features
    
    def get_output_length(self,length):
        return length

    def get_output_dim(self):
        return self.output_dim

@dataclass
class SpecConv1dEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.cpc_spec.SpecConv1dEncoder"

    epoch_length:int = 3000
    epoch_stride:int = 0

    features:List[int] = field(default_factory=lambda: [128,512])
    grouped:bool = False # in this case all feature dimensions in the list from above have to be divisible by the input_channels
