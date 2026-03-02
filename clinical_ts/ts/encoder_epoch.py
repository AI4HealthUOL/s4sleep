__all__ = ['EpochEncoderPre', 'EpochEncoderPreConfig', 'EpochEncoderPost', 'EpochEncoderPostConfig', 'EpochEncoderRNNConfig','EpochEncoderTransformerConfig','EpochEncoderS4Config','EpochEncoderNoneS4Config']

import torch

import dataclasses
from dataclasses import dataclass, field

from ..template_modules import PrePostBase, PrePostBaseConfig, TimeSeriesEncoderConfig

#for predefined epochencoder
from .encoder import RNNEncoderConfig, TransformerEncoderConfig, NoEncoderConfig
from .rnn import RNNPredictorConfig
from .transformer import TransformerPredictorConfig
from .s4 import S4PredictorConfig
from .head import PoolingHeadConfig, MLPHeadConfig

class EpochEncoderPre(PrePostBase):
    def __init__(self, hparams_pre, hparams_input_shape, hparams_input_shape_pre=None):
        super().__init__(hparams_pre, hparams_input_shape, hparams_input_shape_pre)
        assert(hparams_input_shape.sequence_last)
        self.epoch_length = hparams_pre.epoch_length
        self.epoch_stride = hparams_pre.epoch_length if hparams_pre.epoch_stride==0 else hparams_pre.epoch_stride
        self.epochs = 1+(hparams_input_shape.length-self.epoch_length)//self.epoch_stride
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.length = self.epoch_length
        if(self.output_shape.channels2>0):#spec input
            self.output_shape.channels = self.output_shape.channels*self.output_shape.channels2
            self.output_shape.channels2 = 0

    def forward(self, **kwargs):
        seq = kwargs["seq"]
        spec_freqs =  seq.size(2) if (len(seq.size())==4) else None
        if(spec_freqs is not None):#spectrogram input
            seq = seq.view(seq.size(0),-1,seq.size(-1))#flatten: output shape is bs, ch*freq, seq

        if(self.epoch_length==self.epoch_stride):#without copying
            seq = seq[:,:,:self.epoch_length+(self.epochs-1)*self.epoch_stride].view(seq.shape[0],seq.shape[1],-1,self.epoch_length)#bs,channels,epochs,epoch_length
            seq = seq.permute(0,2,1,3).reshape(-1,seq.shape[1],self.epoch_length) #bs*epochs,channels,epoch_length
        else:
            seq = torch.stack([seq[:,:,i*self.epoch_stride:i*self.epoch_stride+self.epoch_length] for i in range(self.epochs)],dim=1)
            seq = seq.view(-1,seq.shape[1],self.epoch_length) #bs*epochs,channels,epoch_length
        if(spec_freqs is not None):
            seq = seq.view(seq.size(0),-1,spec_freqs,seq.size(-1))#output has shape bs*epochs, ch, freq, epoch_length
        res = {"seq":seq}
        if("static" in kwargs.keys()):
            res["static"] = torch.cat([s.unsqueeze(0).repeat(self.epochs) for s in kwargs["static"]],dim=0) if kwargs["static"] is not None else None#bs*epochs
        return res
    
    def get_output_shape(self):
        return self.output_shape
    
    
class EpochEncoderPost(PrePostBase):
    def __init__(self, hparams_post, hparams_input_shape, hparams_input_shape_pre):
        super().__init__(hparams_post, hparams_input_shape)
        self.epoch_length = hparams_post.epoch_length
        self.epoch_stride = hparams_post.epoch_length if hparams_post.epoch_stride==0 else hparams_post.epoch_stride
        self.epochs = 1+(hparams_input_shape_pre.length-self.epoch_length)//self.epoch_stride # unfortunately this requires input_shape_pre

        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.length = self.epochs*hparams_input_shape.length 

    def forward(self, **kwargs):
        seq = kwargs["seq"]
        res = {"seq": seq.view(seq.shape[0]//self.epochs,-1,seq.shape[-1])} #bs*epochs,seq',feat->bs,epochs*seq',feat
        if("static" in kwargs.keys()):
            res["static"] = kwargs["static"][::self.epochs] if kwargs["static"] is not None else None
        return res
    
    def get_output_shape(self):
        return self.output_shape
    

@dataclass
class EpochEncoderPreConfig(PrePostBaseConfig):
    _target_:str = "clinical_ts.ts.encoder_epoch.EpochEncoderPre"
    epoch_length:int = 3000
    epoch_stride:int = 0 #0 means epoch_stride=epoch_length

@dataclass
class EpochEncoderPostConfig(PrePostBaseConfig):
    _target_:str = "clinical_ts.ts.encoder_epoch.EpochEncoderPost"
    epoch_length:int = 3000
    epoch_stride:int = 0 #0 means epoch_stride=epoch_length


########################################################################################
#predefined epoch encoder
########################################################################################
@dataclass
class EpochEncoderRNNConfig(TimeSeriesEncoderConfig):
    name: str = "tsencee"
    pre: EpochEncoderPreConfig = field(default_factory= EpochEncoderPreConfig)
    enc: RNNEncoderConfig = field(default_factory=RNNEncoderConfig)
    pred: RNNPredictorConfig = field(default_factory=RNNPredictorConfig)
    head: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)
    post: EpochEncoderPostConfig = field(default_factory= EpochEncoderPostConfig)

@dataclass
class EpochEncoderTransformerConfig(TimeSeriesEncoderConfig):
    name: str = "tsencee"
    pre: EpochEncoderPreConfig = field(default_factory= EpochEncoderPreConfig)
    enc: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)
    pred: TransformerPredictorConfig = field(default_factory=TransformerPredictorConfig)
    head: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)
    post: EpochEncoderPostConfig = field(default_factory= EpochEncoderPostConfig)

@dataclass
class EpochEncoderS4Config(TimeSeriesEncoderConfig):
    name: str = "tsencee"
    pre: EpochEncoderPreConfig = field(default_factory= EpochEncoderPreConfig)
    enc: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)
    pred: S4PredictorConfig = field(default_factory=S4PredictorConfig)
    head: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)
    post: EpochEncoderPostConfig = field(default_factory= EpochEncoderPostConfig)

@dataclass
class EpochEncoderNoneS4Config(TimeSeriesEncoderConfig):
    name: str = "tsencee"
    pre: EpochEncoderPreConfig = field(default_factory= EpochEncoderPreConfig)
    enc: NoEncoderConfig = field(default_factory=NoEncoderConfig)
    pred: S4PredictorConfig = field(default_factory=S4PredictorConfig)
    head: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)
    post: EpochEncoderPostConfig = field(default_factory= EpochEncoderPostConfig)
