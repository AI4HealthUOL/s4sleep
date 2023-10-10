__all__ = ['TransformerEncoder', 'TransformerPredictor', 'TransformerHeadGlobal', 'TransformerHeadMulti', 'TransformerEncoderConfig', 'TransformerPredictorConfig', 'TransformerHeadMultiConfig', 'TransformerHeadGlobalConfig']

from .transformer import *
from .cpc_template import EncoderBase, EncoderBaseConfig, PredictorBase, PredictorBaseConfig, HeadBase, HeadBaseConfig
from typing import List, Any
from dataclasses import dataclass, field

class TransformerEncoder(EncoderBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.encoder = TransformerConvStemTokenizer(features=hparams.encoder.features, input_channels=hparams.base.input_channels, kernel_sizes=hparams.encoder.kernel_sizes, strides=hparams.encoder.strides, paddings=hparams.encoder.paddings ,normalization=True, dim_data=1)#can mimic all other tokenizers
            
        self.output_dim = hparams.encoder.features[-1]

    def forward(self, input, static=None):
        return self.encoder(input)
    
    def get_output_length(self,length):
        return self.encoder.get_output_length(length)

    def get_output_dim(self):
        return self.output_dim

@dataclass
class TransformerEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.cpc_transformer.TransformerEncoder"
    features:List[int] = field(default_factory=lambda: [64,128,256,512])#wav2vec [64,128,256,512]
    kernel_sizes:List[int]=field(default_factory=lambda: [1,1,1,1])#wav2vec [5,5,5,5]
    paddings:List[int]=field(default_factory=lambda: [0,0,0,0])#wav2vec [3,3,3,3]
    strides:List[int]=field(default_factory=lambda: [1,1,1,1])#wav2vec [3,3,3,3]

#class TransformerPatchEncoder(EncoderBase):
#    def __init__(self, hparams, hparams_base):
#        super().__init__(hparams)
#        self.encoder = TransformerPatchTokenizer([hparams.encoder.output_dim], input_channels=hparams.base.input_channels,patch_size=hparams.encoder.timesteps_per_token, dim_data=1)
#        self.output_dim = hparams.encoder.output_dim
#
#    def forward(self, input, static=None):
#        return self.encoder(input)
#    
#    def get_output_length(self,length):
#        return self.encoder.get_output_length(length)
#
#    def get_output_dim(self):
#        return self.output_dim
#
#@dataclass
#class TransformerPatchEncoderConfig(EncoderBaseConfig):
#    _target_:str = "clinical_ts.cpc_transformer.TransformerPatchEncoder"
#    output_dim:int = 512

class TransformerPredictor(PredictorBase):
    def __init__(self, hparams, encoder_output_dim, encoder_output_length, encoder_static_output_dim):
        super().__init__(hparams, encoder_output_dim, encoder_output_length, encoder_static_output_dim)
        self.predictor = TransformerModule(dim_model=hparams.predictor.model_dim, mlp_ratio=hparams.predictor.mlp_ratio, dropout=hparams.predictor.dropout, num_layers=hparams.predictor.layers, num_heads=hparams.predictor.heads, masked=hparams.predictor.causal, max_length=encoder_output_length, pos_enc=hparams.predictor.pos_enc, activation=hparams.predictor.activation, norm_first=hparams.predictor.norm_first, cls_token=hparams.predictor.cls_token, input_size=encoder_output_dim if encoder_output_dim!=hparams.predictor.model_dim else None,output_size=None, native=(hparams.predictor.backbone=="native")) #note: only apply linear layer before if feature dimensions do not match

    def forward(self, seq, static=None):
        return self.predictor(seq)

    
@dataclass
class TransformerPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.cpc_transformer.TransformerPredictor"
    model_dim:int = 512 
    
    pos_enc:str="sine" #help="none/sine/learned")
    cls_token:bool=False
    
    mlp_ratio:float = 4.0
    heads:int = 8
    layers:int = 4
    dropout:float = 0.1
    attention:float = 0.1
    stochastic_depth_rate:float = 0.0
    activation:str ="gelu" #help="gelu/relu")
    norm_first:bool=True
    backbone:str = "native" #native/timm use native Pytorch transformer layers    
        
class TransformerHeadGlobal(HeadBase):
    #supervised transformer head
    def __init__(self, hparams, target_dim, encoder_output_length, encoder_static_output_dim):
        super().__init__(hparams, target_dim, encoder_output_length, encoder_static_output_dim)
        assert(hparams.predictor._target_!="clinical_ts.cpc_transformer.TransformerPredictor" or (hparams.predictor.cls_token is True or (hparams.predictor.cls_token is False and (hparams.head.head_pooling_type!="cls" and hparams.head.head_pooling_type!="meanmax-cls"))))

        self.head = TransformerHead(hparams.predictor.model_dim,target_dim,pooling_type=hparams.head.head_pooling_type,batch_first=False,n_heads_seq_pool=hparams.head.head_n_heads_seq_pool)

    def forward(self,x):   
        return self.head(x)


@dataclass
class TransformerHeadGlobalConfig(HeadBaseConfig):
    _target_ = "clinical_ts.cpc_transformer.TransformerHeadGlobal"
    multi_prediction:bool=False

    pooling_type:str="meanmax" #,help="cls/meanmax/meanmax-cls/seq/seq-meanmax/seq-meanmax-cls")
    head_n_heads_seq_pool:int=1

class TransformerHeadMulti(HeadBase):
    #pretraining head as used in modified CPC https://arxiv.org/abs/2002.02848 (in addition to layer norm)
    def __init__(self, hparams, target_dim, encoder_output_length, encoder_static_output_dim):
        super().__init__(hparams, target_dim, encoder_output_length, encoder_static_output_dim)

        self.tf = TransformerModule(dim_model=hparams.predictor.model_dim, num_layers=1, num_heads=hparams.head.num_heads, masked=hparams.head.causal, max_length=encoder_output_length, batch_first=True, input_size=hparams.predictor.model_dim, output_size=target_dim, norm_first=True, native=(hprams.head.backbone=="native"))

    def forward(self, seq):
        return self.tf(seq)

@dataclass
class TransformerHeadMultiConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.cpc_transformer.TransformerHeadMulti"
    multi_prediction: bool = True # sequence level prediction

    num_heads:int = 8
    causal:bool= True #causal layer (e.g. for CPC)
    backbone:str= "native" #native/timm use native pytorch transformer layer
