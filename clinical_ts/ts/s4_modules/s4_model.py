__all__ = ['S4Model']
#adapted from https://github.com/HazyResearch/state-spaces/blob/main/example.py

import torch.nn as nn

from .s4original import S4 as S4original
from .s42 import S4 as S42
from .s4 import S4 as S4new
from .s4d import S4 as S4d
from .s4v4 import S4Block as S4v4

from .s4_utils import DropoutNd

class LowRankAdapter1d(nn.Module):
    """Residual low-rank adapter on (B, C, L) tensors."""

    def __init__(self, d_model, rank, dropout=0.0, scale=1.0):
        super().__init__()
        rank = max(1, int(rank))
        self.scale = float(scale)
        self.down = nn.Conv1d(d_model, rank, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Conv1d(rank, d_model, kernel_size=1, bias=False)

        # Start as near-identity residual block: adapter output is initially zero.
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.scale * self.up(self.drop(self.act(self.down(x))))


class S4Model(nn.Module):

    def __init__(
        self, 
        d_input, # None to disable encoder
        d_output, # None to disable decoder
        d_state=64, #MODIFIED: N
        d_model=512, #MODIFIED: H
        n_layers=4, 
        dropout=0.2,
        tie_dropout=False, #MODIFIED
        prenorm=False,
        l_max=1024,
        transposed_input=True, # behaves like 1d CNN if True else like a RNN with batch_first=True
        bidirectional=True, #MODIFIED
        layer_norm = True, # MODIFIED
        pooling = True, # MODIFIED
        backbone= "s42", # MODIFIED
        adapter_enabled=False,
        adapter_rank=16,
        adapter_dropout=0.0,
        adapter_last_k=0,
        adapter_scale=1.0,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.transposed_input = transposed_input
        
        # MODIFIED TO ALLOW FOR MODELS WITHOUT ENCODER
        if(d_input is None):
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Conv1d(d_input, d_model, 1) if transposed_input else nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.adapter_enabled = bool(adapter_enabled)

        if adapter_last_k is None:
            adapter_last_k = 0
        adapter_last_k = int(adapter_last_k)
        if adapter_last_k <= 0:
            adapter_start = 0
        else:
            adapter_start = max(0, n_layers - adapter_last_k)

        for _ in range(n_layers):
            if(backbone=="s4original"):
                self.s4_layers.append(
                S4original(
                    d_state=d_state,
                    l_max=l_max,
                    d_model=d_model, 
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout,
                    tie_dropout=tie_dropout, 
                    transposed=True,
                ))
            elif(backbone == "s4new"):
                self.s4_layers.append(
                S4new(
                    d_state=d_state,
                    l_max=l_max,
                    d_model=d_model, 
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout,
                    tie_dropout=tie_dropout, 
                    transposed=True,
                ))
            elif(backbone == "s42"):
                self.s4_layers.append(
                S42(
                    d_state=d_state,
                    l_max=l_max,
                    d_model=d_model, 
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout,
                    tie_dropout=tie_dropout, 
                    transposed=True,
                ))
            elif(backbone == "s4d"):
                self.s4_layers.append(
                S4d(
                    d_state=d_state,
                    l_max=l_max,
                    d_model=d_model, 
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout,
                    tie_dropout=tie_dropout, 
                    transposed=True,
                ))
            elif(backbone == "s4v4"):
                self.s4_layers.append(
                S4v4(
                    d_state=d_state,#passed to SSMKernel
                    l_max=l_max,
                    d_model=d_model, 
                    bidirectional=bidirectional,
                    final_act='glu',
                    dropout=dropout,
                    tie_dropout=tie_dropout, 
                    transposed=True,
                    mode=None,
                    kernel="s4"
                ))
            elif(backbone == "s4dv4"):
                self.s4_layers.append(
                S4v4(
                    d_state=d_state,#passed to SSMKernel
                    l_max=l_max,
                    d_model=d_model, 
                    bidirectional=bidirectional,
                    final_act='glu',
                    dropout=dropout,
                    tie_dropout=tie_dropout, 
                    transposed=True,
                    mode=None,
                    kernel="s4d"
                ))
            #MODIFIED TO ALLOW BATCH NORM MODELS
            self.layer_norm = layer_norm
            if(layer_norm):
                self.norms.append(nn.LayerNorm(d_model))
            else: #MODIFIED
                self.norms.append(nn.BatchNorm1d(d_model))

            if(tie_dropout):
                self.dropouts.append(DropoutNd(dropout,transposed=True))
            else:
                self.dropouts.append(nn.Dropout(dropout))

            if self.adapter_enabled and _ >= adapter_start:
                self.adapters.append(
                    LowRankAdapter1d(
                        d_model=d_model,
                        rank=int(adapter_rank),
                        dropout=float(adapter_dropout),
                        scale=float(adapter_scale),
                    )
                )
            else:
                self.adapters.append(nn.Identity())

        self.pooling = pooling
        # Linear decoder
        # MODIFIED TO ALLOW FOR MODELS WITHOUT DECODER
        if(d_output is None):
            self.decoder = None
        else:
            self.decoder = nn.Linear(d_model, d_output)

    #MODIFIED
    def forward(self, x, rate=1.0):
        """
        Input x is shape (B, d_input, L) if transposed_input else (B, L, d_input)
        """
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L) if transposed_input else (B, L, d_input) -> (B, L, d_model)

        if(self.transposed_input is False):
            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        
        for layer, norm, dropout, adapter in zip(self.s4_layers, self.norms, self.dropouts, self.adapters):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                # MODIFIED
                z = norm(z.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)
            
            # Apply S4 block: we ignore the state input and output
            # MODIFIED
            z, _ = layer(z, rate=rate)

            # Dropout on the output of the S4 block
            z = dropout(z)
            z = adapter(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                # MODIFIED
                x = norm(x.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)

        x = x.transpose(-1, -2) # (B, d_model, L) -> (B, L, d_model)

        # MODIFIED ALLOW TO DISABLE POOLING
        if(self.pooling):
            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)

        # Decode the outputs
        if(self.decoder is not None):
            x = self.decoder(x)  # (B, d_model) -> (B, d_output) if pooling else (B, L, d_model) -> (B, L, d_output)
            
        if(not self.pooling and self.transposed_input is True):
            x = x.transpose(-1, -2) # (B, L, d_output) -> (B, d_output, L)
        return x
