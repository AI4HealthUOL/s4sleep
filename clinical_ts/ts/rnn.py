__all__ = ['RNNPredictor', 'RNNPredictorConfig']

import torch.nn as nn

from ..template_modules import PredictorBase, PredictorBaseConfig

from dataclasses import dataclass

class RNNPredictor(PredictorBase):
    def __init__(self, hparams_predictor, hparams_input_shape):
        super().__init__(hparams_predictor, hparams_input_shape)
        rnn_arch = nn.LSTM if not(hparams_predictor.gru) else nn.GRU
        self.rnn = rnn_arch(hparams_input_shape.channels,hparams_predictor.model_dim//2 if not(hparams_predictor.causal) else hparams_predictor.model_dim,num_layers=hparams_predictor.n_layers,batch_first=True,bidirectional=not(hparams_predictor.causal))

        self.d_hidden = 2*hparams_predictor.n_layers if not(hparams_predictor.causal) else hparams_predictor.n_layers

        if(hparams_input_shape.static_dim>0 and  hparams_predictor.static_input):
            self.mlp1 = nn.Sequential(nn.Linear(hparams_input_shape.static_dim,hparams_predictor.model_dim*hparams_predictor.n_layers),nn.ReLU(inplace=True))
            self.mlp2 = nn.Sequential(nn.Linear(hparams_input_shape.static_dim,hparams_predictor.model_dim*hparams_predictor.n_layers),nn.ReLU(inplace=True))
        self.static_input = hparams_predictor.static_input

    def forward(self, **kwargs):
        seq = kwargs["seq"]
        static = kwargs["static"]
        if(static is not None and self.static_input):
            output_shape = (self.d_hidden,static.shape[0],-1)
            seq, _ = self.rnn(seq,(self.mlp1(static).view(output_shape),self.mlp2(static).view(output_shape)))
        else:
            seq, _ = self.rnn(seq)
        return {"seq":seq}

@dataclass
class RNNPredictorConfig(PredictorBaseConfig):
    _target_:str = "clinical_ts.ts.rnn.RNNPredictor"
    model_dim:int = 512 
    gru:bool=False # help="use GRU instead of LSTM")
    n_layers:int = 2 # help="number of RNN layers")
    static_input:bool=False #"do not use static information (if available) for initializing hidden states")- disabled by default
