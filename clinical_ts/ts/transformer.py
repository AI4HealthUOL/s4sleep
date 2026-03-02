__all__ = ['TransformerPredictor', 'TransformerPredictorConfig']

from .transformer_modules.transformer import TransformerModule
from ..template_modules import PredictorBase, PredictorBaseConfig
from dataclasses import dataclass

class TransformerPredictor(PredictorBase):
    def __init__(self, hparams_predictor, hparams_input_shape):
        super().__init__(hparams_predictor, hparams_input_shape)
        self.predictor = TransformerModule(dim_model=hparams_predictor.model_dim, 
                                           mlp_ratio=hparams_predictor.mlp_ratio, 
                                           dropout=hparams_predictor.dropout, 
                                           num_layers=hparams_predictor.layers, 
                                           num_heads=hparams_predictor.heads, 
                                           masked=hparams_predictor.causal, 
                                           max_length=hparams_input_shape.length, 
                                           pos_enc=hparams_predictor.pos_enc, 
                                           activation=hparams_predictor.activation, 
                                           norm_first=hparams_predictor.norm_first, 
                                           cls_token=hparams_predictor.cls_token, 
                                           input_size=hparams_input_shape.channels if hparams_input_shape.channels!=hparams_predictor.model_dim else None,
                                           output_size=None, native=(hparams_predictor.backbone=="native")) #note: only apply linear layer before if feature dimensions do not match

    def forward(self, **kwargs):
        return {"seq": self.predictor(kwargs["seq"])}
    
@dataclass
class TransformerPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.ts.transformer.TransformerPredictor"
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