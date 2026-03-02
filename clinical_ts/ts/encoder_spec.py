__all__ = ['SpecConv2dEncoder','SpecConv2dEncoderConfig','SpecConv1dEncoder','SpecConv1dEncoderConfig']

from ..template_modules import EncoderBase, EncoderBaseConfig
import dataclasses
from dataclasses import dataclass, field
from typing import List
import torch.nn as nn
import torch

class SpecConv2dEncoder(EncoderBase):
    def __init__(self, hparams_encoder, hparams_input_shape, static_stats_train):
        super().__init__(hparams_encoder, hparams_input_shape, static_stats_train)

        layers = []
        for s,e in zip([hparams_input_shape.channels]+hparams_encoder.features[:-1],hparams_encoder.features):
            layers.append(nn.Conv2d(s,e,kernel_size=3,padding=1))
            layers.append(nn.GELU())

        self.encoder = nn.Sequential(*layers)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = hparams_encoder.features[-1]
        self.output_shape.freq_bins = 0
        self.output_shape.sequence_last = False

    def forward(self, **kwargs):
        #input shape (bs, ch, freq, ts)
        seq = self.encoder(kwargs["seq"])
        seq = torch.mean(seq,dim=2) # bs, ch, ts
        return {"seq": seq.transpose(1,2)}# bs, ts, ch
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class SpecConv2dEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.ts.spec.SpecConv2dEncoder"
    features:List[int] = field(default_factory=lambda: [128,512])

class SpecConv1dEncoder(EncoderBase):
    def __init__(self, hparams_encoder, hparams_input_shape):
        super().__init__(hparams_encoder, hparams_input_shape)

        layers = []
        for s,e in zip([hparams_input_shape.channels*hparams_input_shape.freq_bins]+hparams_encoder.features[:-1],hparams_encoder.features):
            layers.append(nn.Conv1d(s,e,kernel_size=3,padding=1,groups=hparams_input_shape.channels if hparams_encoder.grouped else 1))
            layers.append(nn.GELU())

        self.encoder = nn.Sequential(*layers)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.channels = hparams_encoder.features[-1]
        self.output_shape.freq_bins = 0
        self.output_shape.sequence_last = False

    def forward(self, **kwargs):
        seq = kwargs["seq"]
        #input shape (bs, ch, freq, ts)-> (bs, ch*freq, ts)
        seq = self.encoder(seq.view(seq.shape[0],-1,seq.shape[-1]))
        return {"seq": seq.transpose(1,2)}# bs, ts, ch
    
    def get_output_shape(self):
        return self.output_shape

@dataclass
class SpecConv1dEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.ts.spec.SpecConv1dEncoder"
    features:List[int] = field(default_factory=lambda: [128,512])
    grouped:bool = False # in this case all feature dimensions in the list from above have to be divisible by the input_channels
