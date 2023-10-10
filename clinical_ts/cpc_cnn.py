__all__ = ['ResnetPredictor','ResnetPredictorConfig']

from .xresnet1d import *
from .cpc_template import PredictorBase, PredictorBaseConfig
from typing import List,Any
from dataclasses import dataclass, field

class ResnetPredictor(PredictorBase):
    def __init__(self, hparams, encoder_output_dim, encoder_output_length, encoder_static_output_dim):
        super().__init__(hparams, encoder_output_dim, encoder_output_length, encoder_static_output_dim)
        self.predictor = _xresnet1d(
                expansion=hparams.predictor.expansion,
                layers=hparams.predictor.layers,
                input_channels=encoder_output_dim,
                stem_szs=hparams.predictor.stem_szs,
                input_size=encoder_output_length,
                heads=hparams.predictor.heads,
                mhsa=hparams.predictor.mhsa,
                kernel_size=hparams.predictor.kernel_size,
                kernel_size_stem=hparams.predictor.kernel_size_stem,
                widen=hparams.predictor.widen,
                model_dim=hparams.predictor.model_dim,
                num_classes=None)
        
    def forward(self, seq, static=None):   
        return self.predictor(seq.transpose(1,2)).transpose(1,2)

@dataclass
class ResnetPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.cpc_cnn.ResnetPredictor"
    expansion:int = 4
    layers:List[int] = field(default_factory=lambda: [3,4,6,3])
    stem_szs:List[int] = field(default_factory=lambda: [32,32,64])
    heads:int=4
    mhsa:bool=False 
    kernel_size:int=5
    kernel_size_stem:int=5
    widen:float = 1.0
    model_dim:int = 256

    