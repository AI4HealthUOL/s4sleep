__all__ = ['S4Predictor','S4PredictorConfig','S4Head','S4HeadConfig']

from .s4_model import *
from .cpc_template import PredictorBase, PredictorBaseConfig, HeadBase, HeadBaseConfig
from typing import Any
from dataclasses import dataclass

class S4Predictor(PredictorBase):
    def __init__(self, hparams, encoder_output_dim, encoder_output_length, encoder_static_output_dim):
        super().__init__(hparams, encoder_output_dim, encoder_output_length, encoder_static_output_dim)
        self.predictor = S4Model(
            d_input = encoder_output_dim if encoder_output_dim!=hparams.predictor.model_dim else None,#modified
            d_output = None,
            d_state = hparams.predictor.state_dim,
            d_model = hparams.predictor.model_dim,
            n_layers = hparams.predictor.layers,
            dropout = hparams.predictor.dropout,
            prenorm = hparams.predictor.prenorm,
            l_max = encoder_output_length,
            transposed_input = False,
            bidirectional=not(hparams.predictor.causal),
            layer_norm=not(hparams.predictor.batchnorm),
            pooling = False,
            backbone = hparams.predictor.backbone) #note: only apply linear layer before if feature dimensions do not match

    def forward(self, seq, static=None):   
        return self.predictor(seq) 

@dataclass
class S4PredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.cpc_s4.S4Predictor"
    model_dim:int = 512 
    causal: bool = True #use bidirectional predictor
    state_dim:int = 64 #help="S4: N")
    layers:int = 4
    dropout:float=0.2
    prenorm:bool=False
    batchnorm:bool=False
    backbone:str="s42" #help="s4original/s4new/s4d")  

class S4Head(HeadBase):

    def __init__(self, hparams, target_dim, encoder_output_length, encoder_static_output_dim):
        '''S4 analogue of the pretraining head used in modified CPC https://arxiv.org/abs/2002.02848 (in addition to layer norm) can also be used as global prediction head'''
        super().__init__(hparams, target_dim, encoder_output_length, encoder_static_output_dim)

        self.s4 = S4Model(
            d_input = None,#matches output dim of the encoder
            d_output = target_dim,
            d_state = hparams.head.state_dim,
            d_model = hparams.predictor.model_dim,
            n_layers = 1,
            #dropout = hparams.predictor.dropout, #use default
            #prenorm = hparams.predictor.prenorm, #
            l_max = encoder_output_length,
            transposed_input = False,
            bidirectional=not(hparams.head.causal),
            layer_norm=not(hparams.head.batchnorm),
            pooling = not(hparams.head.multi_prediction),
            backbone = hparams.head.backbone)
        #self.tf = TransformerModule(dim_model=hparams.head.model_dim, num_layers=1, num_heads=hparams.head.num_heads, masked=hparams.head.causal, max_length=encoder_output_length, batch_first=True, input_size=hparams.predictor.model_dim, output_size=target_dim)

    def forward(self, seq):
        return self.s4(seq)

@dataclass
class S4HeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.cpc_s4.S4Head"
    multi_prediction: bool = True # sequence level prediction or not

    state_dim:int = 64
    dropout:float=0.2
    prenorm:bool=False
    batchnorm:bool=False
    backbone:str="s42" #help="s4original/s4new/s4d")  

    causal:bool= True #causal layer (e.g. for CPC)
