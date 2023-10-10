__all__ = [ 'LRMonitorCallback', '_string_to_class', 'LossConfig', 'SupervisedLossConfig', 'CrossEntropyFocalLoss', 'CEFLossConfig',  'CELossConfig', 'CrossEntropyLoss', 'CPCTemplate', 'EncoderBase', 'EncoderBaseConfig', 'EncoderStaticBase', 'EncoderStaticBaseConfig', 'PredictorBase', 'PredictorBaseConfig', 'HeadBase', 'HeadBaseConfig', 'QuantizerBase', 'GumbelQuantizer','GumbelQuantizerConfig','QuantizerBaseConfig', 'NoEncoder', 'NoEncoderConfig', 'BasicEncoderStatic', 'BasicEncoderStaticConfig','EpochEncoder','EpochEncoderConfig', 'NoPredictor', 'NoPredictorConfig', 'PoolingHead', 'PoolingHeadConfig' , 'MLPHead', 'MLPHeadConfig']

import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
import math
import numpy as np

from pathlib import Path
from pytorch_lightning.callbacks import Callback, BaseFinetuning
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

from dataclasses import dataclass, field
from typing import List

from itertools import chain
from collections.abc import Iterable
from .basic_conv1d import bn_drop_lin

from .schedulers import *
from .quantizer import *
from .transformer import *

from .timeseries_utils import *

torch.backends.cuda.reserved_memory = 4 * 1024 * 1024 * 1024  # Reserve 4GB GPU memory
torch.backends.cuda.max_split_size_mb = 1024  # Set the maximum chunk size to 1GB

class LRMonitorCallback(Callback):
    def __init__(self,interval="epoch",start=True,end=True):
        super().__init__()
        self.interval = interval
        self.start = start
        self.end = end
        
    def on_train_batch_start(self, trainer, *args, **kwargs):                
        if(self.interval == "step" and self.start):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)

    def on_train_epoch_start(self, trainer, *args, **kwargs):                
        if(self.interval == "epoch" and self.start):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)
    
    def on_train_batch_end(self, trainer, *args, **kwargs):                
        if(self.interval == "step" and self.end):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)

    def on_train_epoch_end(self, trainer, *args, **kwargs):                
        if(self.interval == "epoch" and self.end):
            current_lrs = [d['lr'] for d in trainer.optimizers[0].param_groups]
            print(f'Epoch: {trainer.current_epoch} Step: {trainer.global_step} LRs:',current_lrs)

############################################################################################################
def _freeze_bn_stats(model, freeze=True):
    for m in model.modules():
        if(isinstance(m,nn.BatchNorm1d)):
            if(freeze):
                m.eval()
            else:
                m.train()

############################################################################################################
def sanity_check(model, state_dict_pre):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading state dict for sanity check")
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'head.1.weight' in k or 'head.1.bias' in k:
            continue


        assert ((state_dict[k].cpu() == state_dict_pre[k].cpu()).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")

############################################################################################################
#from https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/vqvae.py
# -----------------------------------------------------------------------------
def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi/2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t


############################################################################################################
import importlib

def _string_to_class(_target_):
    if(len(_target_.split("."))==1):#assume global namespace
        cls_ = globals()[_target_]
    else:
        mod_ = importlib.import_module(".".join(_target_.split(".")[:-1]))
        cls_ = getattr(mod_, _target_.split(".")[-1])
    return cls_
        

@dataclass
class LossConfig:
    _target_:str = ""
    loss_type:str=""#supervised vs. cpc vs. masked_pred vs. masked_rec vs. lm

####################################################################################
# BASIC supervised losses
###################################################################################
@dataclass
class SupervisedLossConfig(LossConfig):
    _target_:str = "" #insert appropriate loss class
    loss_type:str ="supervised"
    supervised_type:str="classification_single"#"classification_multi","regression_quantile"

class CrossEntropyLoss(nn.Module):
    #standard CE loss that just passes the class_weights correctly
    def __init__(self, hparams):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(hparams.loss.weight,dtype=np.float32)) if len(hparams.loss.weight)>0 else None)
        
    def forward(self, preds, targs):
        return self.ce(preds,targs)

@dataclass
class CELossConfig(SupervisedLossConfig):
    _target_:str= "clinical_ts.cpc_template.CrossEntropyLoss"
    loss_type:str="supervised"
    supervised_type:str="classification_single"
    weight:List[float]=field(default_factory=lambda: [])#class weights e.g. inverse class prevalences

class CrossEntropyFocalLoss(nn.Module):
    """
    Focal CE loss for multiclass classification with integer labels
    Reference: https://github.com/artemmavrin/focal-loss/blob/7a1810a968051b6acfedf2052123eb76ba3128c4/src/focal_loss/_categorical_focal_loss.py#L162
    """
    def __init__(self, hparams):
        super().__init__()
        self.gamma = hparams.loss.gamma
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(hparams.loss.weight, dtype=np.float32)) if len(hparams.loss.weight)>0 else None, reduction='none')

    def forward(self, preds, targs):
        probs = F.softmax(preds, dim=-1).squeeze(-1)
        probs = torch.gather(probs, -1, targs.unsqueeze(-1)).squeeze(-1)
        focal_modulation = torch.pow((1 - probs), self.gamma if type(self.gamma)==float else self.gamma.index_select(dim=0, index=preds.argmax(dim=-1)))
        # mean aggregation
        return (focal_modulation*self.ce(input=preds, target=targs)).mean()
        
@dataclass
class CEFLossConfig(SupervisedLossConfig):
    _target_:str= "clinical_ts.cpc_template.CrossEntropyFocalLoss"
    loss_type:str="supervised"
    supervised_type:str="classification_single"
    weight:List[float]=field(default_factory=lambda: []) #ignored if empty list is passed
    gamma:float=2.


class EncoderBase(nn.Module):
    def __init__(self, hparams):
        '''
        input shape: (bs,channels,seq) + optional static (bs,feat)
        selected encoders e.g. NoEncoder also accept (bs,channels,freq,seq) for the first argument
        output shape: bs,seq,feat
        '''
        super().__init__()
    
    def get_output_length(self,length):
        raise NotImplementedError

    def get_output_dim(self):
        raise NotImplementedError

@dataclass
class EncoderBaseConfig:
    _target_:str = ""
    timesteps_per_token: int = 1 #timesteps per token a la vision transformer

    
class EncoderStaticBase(nn.Module):
    def __init__(self, hparams):
        '''
        input shape: bs, channels
        output shape: bs, feat
        '''
        super().__init__()
    
    def get_output_dim(self):
        raise NotImplementedError

@dataclass
class EncoderStaticBaseConfig:
    _target_:str = ""

class PredictorBase(nn.Module):
    def __init__(self, hparams, encoder_output_dim, encoder_output_length, encoder_static_output_dim):
        '''
        input shape: bs, seq, feat
        output shape: bs, seq, feat
        '''
        super().__init__()
        self.model_dim = hparams.predictor.model_dim
    
    def get_output_length(self,length):
        return length
    
    def get_output_dim(self):
        return self.model_dim

@dataclass
class PredictorBaseConfig:
    _target_:str = ""
    model_dim: int = 512 #model hidden units/internal dimension (typical 512 for RNNs, 768 for transformers)
    causal: bool = False #use unidirectional predictor

class HeadBase(nn.Module):
    def __init__(self, hparams, target_dim, encoder_output_length, encoder_static_output_dim):
        '''
        input shape: bs, seq, feat
        output shape: bs,seq,nc for multi_prediction else bs,nc for global_pool
        '''
        super().__init__()
        self.target_dim = target_dim
        self.multi_prediction = hparams.head.multi_prediction
    
    def get_output_length(self,length):
        return length if self.multi_prediction else 1
    
    def get_output_dim(self):
        return self.target_dim

@dataclass
class HeadBaseConfig:
    _target_:str = ""
    multi_prediction: bool = False #prediction for every token/set of pooled tokens
    
class QuantizerBase(nn.Module):
    def __init__(self, hparams, encoder_output_dim):
        super().__init__()
        self.output_dim = hparams.predictor.model_dim if hparams.base.pretraining_targets==0 else (hparams.quantizer.embedding_dim*hparams.codebooks if hparams.loss.pretraining_targets==1 else hparams.quantizer.vocab**hparams.quantizer.codebooks)

    def get_output_dim(self):
        return self.output_dim

@dataclass
class QuantizerBaseConfig:
    _target_:str = "clinical_ts.cpc_template.QuantizerBase"
    embedding_dim: int = 64 #model hidden units/internal dimension (typical 512 for RNNs, 768 for transformers)
    vocab: int = 32 #number of items in the vocabulary (in each codebook)
    codebooks: int = 8 #number of codebooks
    
    loss_weight:float=1e-4 #, help="prefactor of the quantizer loss")
    target_dim = 0 #will be set dynamically (through calling get_output_dim)

###########################################################################################################
#SPECIFIC IMPLEMENTATIONS

class NoEncoder(EncoderBase):
    def __init__(self, hparams):
        '''
        no encoder- flattens by default if multiple channels are passed
        '''
        super().__init__(hparams)
        self.timesteps_per_token = hparams.encoder.timesteps_per_token
        self.input_channels = hparams.base.input_channels if hparams.base.freq_bins==0 else hparams.base.freq_bins*hparams.base.input_channels
    def forward(self, input, static=None):
        if(len(input.size())==4):#spectrogram input
            input = input.view(input.size(0),-1,input.size(-1))#flatten    
        if(self.timesteps_per_token==1):
            return input.transpose(1,2)
        else:
            assert(input.size(-1)%self.timesteps_per_token==0)
            size = input.size()
            return input.view(input.shape[0],-1,input.shape[-1]).transpose(1,2).reshape(size[0],size[2]//self.timesteps_per_token,-1).transpose(1,2)
    
    def get_output_length(self,length):
        return length//self.timesteps_per_token
    
    def get_output_dim(self):
        return self.input_channels*self.timesteps_per_token


@dataclass
class NoEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.cpc_template.NoEncoder"

class EpochEncoder(EncoderBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.epoch_length = hparams.encoder.epoch_length
        self.epoch_stride = hparams.encoder.epoch_length if hparams.encoder.epoch_stride==0 else hparams.encoder.epoch_stride
        #create component modules
        hparams_copy = copy.deepcopy(hparams) #hparams.copy()
        hparams_copy.predictor = hparams.encoder.predictor
        hparams_copy.encoder = hparams.encoder.encoder
        hparams_copy.head = hparams.encoder.head

        self.encoder = _string_to_class(hparams_copy.encoder._target_)(hparams_copy)
        encoder_output_dim = self.encoder.get_output_dim()
        encoder_output_length = self.encoder.get_output_length(self.epoch_length)
        encoder_static_output_dim = 0
        self.predictor = _string_to_class(hparams_copy.predictor._target_)(hparams_copy, encoder_output_dim, encoder_output_length, encoder_static_output_dim)
        self.output_dim = hparams_copy.predictor.model_dim

        self.head = _string_to_class(hparams_copy.head._target_)(hparams_copy, self.output_dim, encoder_output_length, encoder_static_output_dim)

    def forward(self, input, static=None):
        #input shape: (bs,channels,seq) + optional static (bs,feat)
        #output shape: bs,seq,feat
        spectrogram_freqs =  input.size(2) if (len(input.size())==4) else None
        if(spectrogram_freqs is not None):#spectrogram input ######################################### wtz
            input = input.view(input.size(0),-1,input.size(-1))#flatten; input has shape bs, ch, freq, seq
        bs = input.shape[0]
        epochs = 1+(input.shape[2]-self.epoch_length)//self.epoch_stride
        if(self.epoch_length==self.epoch_stride):#without copying
            x = input[:,:,:self.epoch_length+(epochs-1)*self.epoch_stride].view(input.shape[0],input.shape[1],-1,self.epoch_length)#bs,channels,epochs,epoch_length
            x = x.permute(0,2,1,3).reshape(-1,x.shape[1],self.epoch_length) #bs*epochs,channels,epoch_length
        else:
            x = torch.stack([input[:,:,i*self.epoch_stride:i*self.epoch_stride+self.epoch_length] for i in range(epochs)],dim=1)
            x = x.view(-1,x.shape[1],self.epoch_length) #bs*epochs,channels,epoch_length
        if(static is not None):
            static = torch.cat([s.unsqueeze(0).repeat(epochs) for s in static],dim=0)#bs*epochs
        if(spectrogram_freqs is not None):
            x = x.view(x.size(0),-1,spectrogram_freqs,x.size(-1))#output has shape bs*epochs, ch, freq, epoch_length
        #print("before enc",x.shape)
        x = self.encoder(x, static)#bs*epochs, seq, feat
        #print("before pred",x.shape)
        x = self.predictor(x)
        #print("before head",x.shape)
        x = self.head(x) #bs*epochs, seq', feat or bs*epochs, feat for global pooling
        #print("before return",x.shape)
        return x.view(bs,-1,x.shape[-1])#bs,seq'',feat
    
    def get_output_length(self,length):
        epochs = 1+(length-self.epoch_length)//self.epoch_stride
        encoder_output_length = self.encoder.get_output_length(self.epoch_length)
        return epochs*self.head.get_output_length(encoder_output_length)
    
    def get_output_dim(self):
        return self.output_dim

@dataclass
class EpochEncoderConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.cpc_template.EpochEncoder"
    epoch_length:int = 3000
    epoch_stride:int = 0 #0 means epoch_stride=epoch_length

    encoder: EncoderBaseConfig = field(default_factory=EncoderBaseConfig)
    predictor: PredictorBaseConfig = field(default_factory=PredictorBaseConfig)
    head: HeadBaseConfig = field(default_factory=HeadBaseConfig)


class BasicEncoderStatic(EncoderStaticBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        assert(len(hparams.encoder_static.lib_ftrs)>0)
        self.input_channels_cat = hparams.base.input_channels_cat
        self.input_channels_cont = hparams.base.input_channels_cont
        assert(len(hparams.encoder_static.embedding_dims)==hparams.base.input_channels_cat and len(hparams.encoder_static.vocab_sizes)==hparams.base.input_channels_cat)
        self.embeddings = nn.ModuleList() if hparams.base.input_channels_cat is not None else None
        for v,e in zip(hparams.encoder_static.vocab_sizes,hparams.encoder_static.embedding_dims):
            self.embeddings.append(nn.Embedding(v,e))
        input_dim = int(np.sum(hparams.encoder_static.embedding_dims) + hparams.base.input_channels_cont)
        lin_ftrs = [input_dim] + hparams.encoder.lib_ftrs
        ps = [hparams.encoder_static.dropout] if not isinstance(hparams.encoder_static.dropout, Iterable) else hparams.encoder_static.dropout
        if len(ps)==1:
            ps= [ps[0]/2] * (len(lin_ftrs)-2) + ps
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
        layers = []
        for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
            layers+=bn_drop_lin(ni,no,hparams.encoder_static.batch_norm,p,actn,layer_norm=False)
        self.layers=nn.Sequential(*layers)

        self.input_channels = hparams.base.input_channels_cont + hparams.base.input_channels_cat
    
    def forward(self, x):
        assert(x.shape[1]==self.input_channels)
        res = []
        for i,e in enumerate(self.embeddings):
            res.append(e(x[:,i].long()))
        res = torch.cat([torch.cat(res,dim=1),x[:,len(self.embeddings):]],dim=1)
        return self.layers(res)
    
    def get_output_dim(self):
        return int(self.lib_ftrs[-1])


@dataclass
class BasicEncoderStaticConfig(EncoderStaticBaseConfig):
    _target_:str = "clinical_ts.cpc_template.BasicEncoderStatic"
    embedding_dims:List[int] = field(default_factory=lambda: []) #list with embedding dimensions
    vocab_sizes:List[int] = field(default_factory=lambda: []) #list with vocab sizes (space-separated)
    lin_ftrs:List[int] = field(default_factory=lambda: [512]) #list with MLP hidden layer sizes; last entry is the static encoder output dimension')
    dropout:float = 0.5
    batch_norm:bool = True


class NoPredictor(EncoderBase):
    def __init__(self, hparams):
        '''
        no predictor e.g. for pretraining purposes
        '''
        super().__init__(hparams)
        
    def forward(self, input, static=None):    
        return input

@dataclass
class NoPredictorConfig(EncoderBaseConfig):
    _target_:str = "clinical_ts.cpc_template.NoPredictor"


class PoolingHead(HeadBase):
    def __init__(self, hparams, target_dim, encoder_output_length, encoder_static_output_dim):
        super().__init__(hparams, target_dim, encoder_output_length, encoder_static_output_dim)
        self.local_pool = hparams.head.multi_prediction
        self.output_dim = hparams.predictor.model_dim if hparams.head.output_layer else target_dim
        
        if(self.local_pool):#local pool
            self.local_pool_padding = (hparams.head.local_pool_kernel_size-1)//2
            self.local_pool_kernel_size = hparams.head.local_pool_kernel_size
            self.local_pool_stride = hparams.head.local_pool_stride
            if(hparams.head.local_pool_max):
                self.pool = torch.nn.MaxPool1d(kernel_size=hparams.head.local_pool_kernel_size,stride=hparams.head.local_pool_stride if hparams.head.local_pool_stride!=0 else hparams.head.local_pool_kernel_size,padding=(hparams.head.local_pool_kernel_size-1)//2)
            else:
                self.pool = torch.nn.AvgPool1d(kernel_size=hparams.head.local_pool_kernel_size,stride=hparams.head.local_pool_stride if hparams.head.local_pool_stride!=0 else hparams.head.local_pool_kernel_size,padding=(hparams.head.local_pool_kernel_size-1)//2)        
        else:#global pool
            if(hparams.head.local_pool_max):
                self.pool = torch.nn.AdaptiveMaxPool1d(1)
            else:
                self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(hparams.predictor.model_dim, target_dim) if hparams.head.output_layer else nn.Identity()

    def forward(self, seq, static=None):
        #input has shape B,S,E
        seq = seq.transpose(1,2) 
        seq = self.pool(seq)
        return self.linear(seq.transpose(1,2))#return B,S,E
        
    def get_output_dim(self):
        return self.output_dim
    
    def get_output_length(self,length):
        if(self.local_pool):
            return int(np.floor((length + 2*self.local_pool_padding- (self.local_pool_kernel_size-1)+1)/self.local_pool_stride))
        else:
            return 1

@dataclass
class PoolingHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.cpc_template.PoolingHead"
    
    multi_prediction:bool = False
    local_pool_max:bool = False
    local_pool_kernel_size: int = 0
    local_pool_stride: int = 0 #kernel_size if 0
    #local_pool_padding=(kernel_size-1)//2
    output_layer: bool = False

class MLPHead(HeadBase):
    def __init__(self, hparams, target_dim, encoder_output_length, encoder_static_output_dim):
        super().__init__(hparams, target_dim, encoder_output_length, encoder_static_output_dim)

        self.multi_prediction = hparams.head.multi_prediction
        
        proj = []
        if(self.multi_prediction is False):
            proj += [torch.nn.AdaptiveAvgPool1d(1),nn.Flatten()]

        if(hparams.head.mlp):# additional hidden layer as in simclr
            proj += [nn.Linear(hparams.predictor.model_dim, hparams.predictor.model_dim),nn.ReLU(inplace=True),nn.Linear(hparams.predictor.model_dim, target_dim,bias=hparams.head.bias)]
        else:
            proj += [nn.Linear(hparams.predictor.model_dim, target_dim,bias=hparams.head.bias)]
        self.proj = nn.Sequential(*proj)

    def forward(self, seq):
        return self.proj(seq if self.multi_prediction else seq.transpose(1,2))

@dataclass
class MLPHeadConfig(HeadBaseConfig):
    _target_:str = "clinical_ts.cpc_template.MLPHead"
    multi_prediction: bool = True # sequence level prediction
    
    mlp: bool = False #mlp in prediction head
    bias: bool = True  #bias for final projection in prediction head 

class GumbelQuantizer(QuantizerBase):
    def __init__(self, hparams, encoder_output_dim):
        super().__init__(hparams)
        self.quantizers = nn.ModuleList([GumbelQuantize(encoder_output_dim, hparams.quantizer.vocab, hparams.quantizer.embedding_dim) for _ in range(hparams.quantizer.codebooks)]) if hparams.base.pretraining_targets>0 else None
        self.pretraining_targets = hparams.base.pretraining_targets
        
    def forward(self, seq):
        if(self.quantizers is None):
            return seq, 0
        else:
            output = []
            loss = 0
            for q in self.quantizers:#quantizers assume batch, features, timesteps
                input_quantized, loss_quantizer, soft_one_hot = q(seq.transpose(1,2))
                loss += loss_quantizer
                if(self.pretraining_targets==1):
                    output.append(input_quantized)
                elif(self.pretraining_targets==2):
                    output.append(soft_one_hot)
            output = torch.cat(output,dim=1).transpose(1,2)#output is assumed to be batch, timesteps, features
            return output, loss

@dataclass
class GumbelQuantizerConfig(QuantizerBaseConfig):
    _target_:str = "clinical_ts.cpc_template.GumbelQuantizer"
    

############################################################################################################
class CPCTemplate(pl.LightningModule):

    def __init__(self, hparams): #base_params, encoder_params, predictor_params, loss_params=CPCConfig(), encoder_static_params=EncoderStaticBaseConfig(), head_clas_params=HeadClasBaseConfig(), head_pretr_params=BasicHeadPretrConfig(), quantizer_params=GumbelQuantizerConfig()):
        super(CPCTemplate, self).__init__()
        assert(len(hparams.data0.input_channels_filter)==0 or hparams.base.input_channels==len(hparams.data0.input_channels_filter))#make sure input channels match in case filtering is applied
        assert(hparams.loss.loss_type=="supervised" or (hparams.trainer.precision==32 or hparams.loss.pretraining_targets==0))#gumbel softmax does not work in fp16
        assert(hparams.loss.loss_type=="supervised" or not(hparams.loss.loss_type=="masked" and hparams.loss.pretraining_targets==0))#masked requires quantizer
        
        # should not be enforced: assert(hparams.predictor.model_dim == hparams.encoder.get_output_dim())

        if(hparams.loss.loss_type=="supervised" and hparams.trainer.pretrained=="" and hparams.trainer.resume=="" and hparams.base.discriminative_lr_factor!=1):
            print("INFO: Setting discriminative-lr-factor=1 for training from scratch.")
            hparams.base.discriminative_lr_factor = 1

        self.lr = hparams.base.lr
        self.save_hyperparameters(hparams)

        self.encoder_seq = _string_to_class(hparams.encoder._target_)(hparams)
        encoder_output_dim = self.encoder_seq.get_output_dim()
        input_size = hparams.base.input_size if isinstance(hparams.base.input_size,int) else int(np.round(hparams.base.input_size*hparams.base.fs))
        encoder_output_length = self.encoder_seq.get_output_length(input_size)
        self.encoder_static = _string_to_class(hparams.encoder_static._target_)(hparams) if hparams.encoder_static._target_!="" else None
        encoder_static_output_dim = self.encoder_static.get_output_dim() if self.encoder_static is not None else 0
        self.predictor = _string_to_class(hparams.predictor._target_)(hparams, encoder_output_dim, encoder_output_length, encoder_static_output_dim)
        self.quantizer = _string_to_class(hparams.quantizer._target_)(hparams) if hparams.loss.loss_type!="supervised" and hparams.loss.pretraining_targets>0 else None
        
        self.masking_pretr = MaskingModule(hparams, encoder_output_dim) if (hparams.loss.loss_type.startswith("masked")) else None

        self.loss_type = hparams.loss.loss_type
        #self.head_clas = _string_to_class(hparams.head_clas._target_)(hparams, encoder_static_output_dim, encoder_output_length) if self.loss_type=="supervised" else None
        
        if(self.loss_type == "supervised"):

            def _prefetch_lbl_itos():#load lbl_itos of the first dataset
                _, lbl_itos, _, _ = self.preprocess_dataset(hparams.data0)
                return lbl_itos[hparams.data0.label_filter] if len(hparams.data0.label_filter)>0 else lbl_itos
    
            self.target_dim = len(hparams.loss.quantiles) if (self.loss_type=="supervised" and hparams.loss.supervised_type == "regression_quantile") else len(_prefetch_lbl_itos())
        elif(self.loss_type== "masked_pred"):
            self.target_dim = hparams.loss.num_outputs
        else: #pretraining
            self.target_dim = self.quantizer.get_output_dim() if self.quantizer is not None else encoder_output_dim
        self.head = _string_to_class(hparams.head._target_)(hparams, self.target_dim, encoder_output_length, encoder_static_output_dim)

        if(hparams.loss._target_=="torch.nn.functional.binary_cross_entropy_with_logits" or hparams.loss._target_=="torch.nn.functional.cross_entropy"):# do not pass hparams to standard losses
            self.criterion = _string_to_class(hparams.loss._target_)
        else:
            self.criterion = _string_to_class(hparams.loss._target_)(hparams)

    def forward(self, seq, stat=None):
        #encoder
        stat_enc = self.encoder_static(stat) if (self.encoder_static is not None and stat is not None) else None
        seq_enc = self.encoder_seq(seq, stat)#passing stat as optional argument to allow for encoders depending on static data
        #masking (optionally during pretraining)
        if(self.masking_pretr is not None):
            seq_enc, prediction_ids, distraction_ids = self.masking_pretr(seq_enc)
        #predictor
        seq_pred = self.predictor(seq_enc) if stat_enc is None else self.predictor(seq_enc, stat_enc)
        
        if(self.loss_type=="supervised"):#supervised
            #classification head
            return self.head(seq_pred, stat_enc) if stat_enc is not None else self.head(seq_pred)
        else:
            #target quantization (optional)
            if(self.quantizer is not None):
                seq_enc_target, loss_quantizer = self.quantizer(seq_enc)
            else:
                seq_enc_target = seq_enc
                loss_quantizer = 0

            #pretraining head
            seq_pred = self.head(seq_pred,stat_enc) if stat_enc is not None else self.head(seq_pred)
            #returning everything
            if(self.masking_pretr is not None):
                #* first-list will be passed to the loss function
                return [seq_pred, seq_enc_target, prediction_ids, distraction_ids], loss_quantizer
            else:
                return [seq_pred, seq_enc_target], loss_quantizer

    def _step(self,data_batch, batch_idx, train, test=False, freeze_bn=False):
        if(self.loss_type=="supervised"):
            if(len(data_batch)==3):
                preds_all = self.forward(data_batch.data, data_batch.static)
            else:
                preds_all = self.forward(data_batch.data)
            #reshape sequence level predictions for loss computation
            preds = preds_all.view(-1,preds_all.shape[-1]) if self.hparams.head.multi_prediction else preds_all #B*S, Nc
            if(self.hparams.loss.supervised_type=="classification_single"):
                targs = data_batch.label.long().view(-1)#casting to long in case labels have another integer type
                if(len(self.hparams.data0.label_filter)>0):#filter out undesired labels
                    preds=preds[targs>=0]
                    targs=targs[targs>=0]
            elif(self.hparams.loss.supervised_type=="classification_multi"):
                targs = data_batch.label.float().view(-1,data_batch.label.shape[-1])#casting to float in case labels have another type

            loss = self.criterion(preds,targs)
            self.log("train_loss" if train else ("val_loss" if not test else "test_loss"), loss)
            return {'loss':loss, "preds":preds_all.detach(), "targs": data_batch.label}
        else:
            outputs, loss_quantizer = self.forward(data_batch[0])
            if(train):
                self.log("loss_quantizer", loss_quantizer)
            if(self.loss_type == "cpc" or self.loss_type == "masked_rec"):
                loss_pretraining, acc = self.criterion(*outputs, eval_acc=True)
                self.log("acc_"+self.loss_type if train else "val_acc_"+self.loss_type, acc)
            elif(self.loss_type=="masked_pred"):
                loss_pretraining, acc = self.criterion(outputs[0],data_batch.label,*outputs[2:], eval_acc=True)
                self.log("acc_"+self.loss_type if train else "val_acc_"+self.loss_type, acc)
            elif(self.loss_type=="lm"):
                loss_pretraining = self.criterion(*outputs)

            self.log("loss_pretraining" if train else "val_loss_pretraining", loss_pretraining)
            #weighting factor only set if quantizer is enabled
            loss = loss_pretraining if self.quantizer is None else loss_pretraining + self.quantizer.loss_weight*loss_quantizer
            self.log("loss" if train else "val_loss", loss)
            
            return loss
      
    def training_step(self, train_batch, batch_idx):
        if(self.hparams.base.linear_eval):
            _freeze_bn_stats(self)
        return self._step(train_batch,batch_idx,train=True,test=False,freeze_bn=self.hparams.base.linear_eval)
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,train=False,test=False)

    def test_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,train=False,test=True)
    

    def validation_epoch_end(self, outputs_all):
        raise NotImplementedError

    def on_fit_start(self):
        if(self.hparams.trainer.pretrained!=""):
            print("Loading pretrained weights from",self.hparams.trainer.pretrained)
            self.load_weights_from_checkpoint(self.hparams.trainer.pretrained)
            if(self.hparams.base.auc_maximization and self.hparams.base.train_head_only):#randomize top layer weights
                print("Randomizing top-layer weights before AUC maximization")
                def init_weights(m):
                    if type(m)== nn.Linear:
                        torch.nn.init.xavier_uniform(m.weight)
                        m.bias.data.fill_(0.01)
                self.head[-1].apply(init_weights)

        if(self.hparams.base.linear_eval):
            print("copying state dict before training for sanity check after training")   
            self.state_dict_pre = copy.deepcopy(self.state_dict().copy())

    
    def on_fit_end(self):
        if(self.hparams.base.linear_eval):
            sanity_check(self,self.state_dict_pre)
    
    #to be implemented by derived classes
    def setup(self, stage):
        raise NotImplementedError

    #to be implemented by derived classes
    def train_dataloader(self):
        raise NotImplementedError

    #to be implemented by derived classes
    def val_dataloader(self):
        raise NotImplementedError

    #override in derived classes to modify
    def preprocess_dataset(self,dataset_kwargs):
        return load_dataset(Path(dataset_kwargs.path))

    #override in derived classes to modify
    def get_custom_transforms(self,dataset_kwargs,lst_default_transforms):
        return lst_default_transforms
    
    def get_params(self, modules=False):
        encoder_modules = []
        if(self.encoder_static is not None):
            encoder_modules.append(self.encoder_static)
        if(self.encoder_seq is not None):
            encoder_modules.append(self.encoder_seq)
        predictor_modules = [self.predictor]
        if(self.quantizer is not None):
            predictor_modules.append(self.quantizer)
        if(self.masking_pretr is not None):
            predictor_modules.append(self.masking_pretr)
        head_modules = [self.head]

        encoder_params = chain(*[e.parameters() for e in encoder_modules])
        predictor_params = chain(*[p.parameters() for p in predictor_modules])
        head_params = chain(*[h.parameters() for h in head_modules])

        if(self.loss_type=="supervised" and (self.hparams.base.linear_eval or self.hparams.base.train_head_only)):
            params = [{"params":head_modules if modules else self.head.parameters(), "lr":self.lr}]
        elif(self.loss_type=="supervised" and self.hparams.trainer.pretrained !="" and self.hparams.base.discriminative_lr_factor != 1.):#discrimative lrs
            params = [{"params":encoder_modules if modules else encoder_params, "lr":self.lr*self.hparams.base.discriminative_lr_factor*self.hparams.base.discriminative_lr_factor},{"params":predictor_modules if modules else predictor_params, "lr":self.lr*self.hparams.base.discriminative_lr_factor},{"params":head_modules if modules else head_params, "lr":self.lr}]
        else:
            #params = self.parameters()
            params = [{"params":head_modules if modules else head_params, "lr":self.lr},{"params":predictor_modules if modules else predictor_params, "lr":self.lr},{"params":encoder_modules if modules else encoder_params, "lr":self.lr}]
        return params


    def configure_optimizers(self):
        
        if(self.hparams.base.optimizer == "sgd"):
            opt = torch.optim.SGD
        elif(self.hparams.base.optimizer == "adam"):
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")
            
        params = self.get_params()
        optimizer = opt(params, self.lr, weight_decay=self.hparams.base.weight_decay)

        if(self.hparams.base.lr_schedule=="const"):
            scheduler = get_constant_schedule(optimizer)
        elif(self.hparams.base.lr_schedule=="const-plateau"):
            scheduler = ReduceLROnPlateau(optimizer)
        elif(self.hparams.base.lr_schedule=="warmup-const"):
            scheduler = get_constant_schedule_with_warmup(optimizer,self.hparams.base.lr_num_warmup_steps)
        elif(self.hparams.base.lr_schedule=="warmup-cos"):
            scheduler = get_cosine_schedule_with_warmup(optimizer,self.hparams.base.lr_num_warmup_steps,self.hparams.base.epochs*len(self.train_dataloader()),num_cycles=0.5)
        elif(self.hparams.base.lr_schedule=="warmup-cos-restart"):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,self.hparams.base.lr_num_warmup_steps,self.hparams.base.epochs*len(self.train_dataloader()),num_cycles=self.hparams.base.epochs-1)
        elif(self.hparams.base.lr_schedule=="warmup-poly"):
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,self.hparams.base.lr_num_warmup_steps,self.hparams.base.epochs*len(self.train_dataloader()),num_cycles=self.hparams.base.epochs-1)   
        elif(self.hparams.base.lr_schedule=="warmup-invsqrt"):
            scheduler = get_invsqrt_decay_schedule_with_warmup(optimizer,self.hparams.base.lr_num_warmup_steps)
        elif(self.hparams.base.lr_schedule=="linear"): #linear decay to be combined with warmup-invsqrt c.f. https://arxiv.org/abs/2106.04560
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, self.hparams.base.epochs*len(self.train_dataloader()))
        else:
            assert(False)

        return (
        [optimizer],
        [
            {
                'scheduler': scheduler,
                'interval': 'epoch' if self.hparams.base.lr_schedule == "const-plateau" else 'step',
                'frequency': 1,
                'monitor': 'val_loss/dataloader_idx_0' if len(self.val_dataloader())>1 else 'val_loss' #for plateau
            }
        ])
        
    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
            
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
