import hydra
from hydra.core.config_store import ConfigStore

from clinical_ts.cpc_template import *
from clinical_ts.cpc_rnn import *
from clinical_ts.cpc_cnn import *
from clinical_ts.cpc_s4 import *
from clinical_ts.cpc_transformer import *
from clinical_ts.cpc_spec import *


from dataclasses import dataclass, field
from typing import Any, List, Union

#utils to log mlflow
try:
    import mlflow
    from omegaconf import DictConfig, ListConfig

    def log_params_from_omegaconf_dict(params):
        for param_name, element in params.items():
            _explore_recursive(param_name, element)

    def _explore_recursive(parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    _explore_recursive(f'{parent_name}.{k}', v)
                else:
                    if(k!="_target_" and v is not None):
                        mlflow.log_param(f'{parent_name}.{k}'," " if v=="" else v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                mlflow.log_param(f'{parent_name}.{i}', v)
except ImportError:
    pass


@dataclass
class ExtraConfig:
    mainclass:str= "clinical_ts.cpc_main.CPCMain"

class ExtraConfigPSG(ExtraConfig):
    mainclass:str= "clinical_ts.cpc_psg.CPCPSG"


@dataclass
class BaseConfig:
    #optimizer
    optimizer:str ='adam'#, help='sgd/adam')#was sgd
    auc_maximization:bool=False #, help="direct auc maximization")
    lr:float = 1e-3# help='initial learning rate', dest="lr")
    weight_decay:float = 1e-3#, type=float, help='weight decay', dest="weight_decay")
    lr_schedule:str = "const" # help="const/const-plateau/warmup-const/warmup-cos/warmup-cos-restart/warmup-poly", default="const")
    lr_num_warmup_steps:int =1000 #help="number of linear lr warmup steps", default=1000)
    discriminative_lr_factor:float = 0.1 #", type=float, help="factor by which the lr decreases per layer group during finetuning", default=0.1)

    train_head_only:bool = False # help="freeze everything except classification head (note: --linear-eval defaults to no hidden layer in classification head)")
    linear_eval:bool = False #", action="store_true", help="linear evaluation instead of full finetuning")
    
    #for dataloader preparation
    batch_size:int = 64 # help='mini-batch size')
    #metrics
    metrics:List[str] = field(default_factory=lambda: [])#"macro_auc","accuracy","f1"
    aggregate_predictions:bool = True #report aggregate performance across chunks
    aggregate_strided_multi_predictions:bool = True #aggregate overlapping predictions (if stride_valtest<input_size for multi_predictions)

    fs:float=100.#sampling frequency
    input_size:int = 1000 #pass either time steps as into or seconds as float
    #input_size_max:Union[int,float] = 0 #maximum input size the model as able to process (e.g. to finetune transformer or s4 models on longer sequences)- default value 0 means setting this to input_size
    chunk_length_train:int = 0 #0: no chunkify, -1: input_size
    chunk_length_val:int = -1 #0: no chunkify, -1: input_size (for validation during training)
    chunk_length_valtest:int = -1 #(both val and test during inference)
    stride_train:int = -1 #-1: chunk_length_train
    stride_val:int = -1 #-1: chunk_length_valtest (for validation during training)
    stride_valtest:int = -1 #(both val and test during inference)
    stride_export:int = -1 #-1:input_size

    input_channels:int = 12 #NOTE: refers to the input channels passed to the model i.e. should coincide with len(input_channels_filter) if non-zero; should be freq_bins*channels if passed to a NoEncoder
    freq_bins:int = 0 # number of frequency bins in the case of spectrogram input
    input_channels_cat: int = 0 #nnumber of categorical input channels
    input_channels_cont: int = 0 #number of continuous input channels
    normalize:bool = True #normalize input signal
    
@dataclass
class BaseConfigData:
    _target_:str = "clinical_ts.cpc_config.BaseConfigData" #just used to filter out data configs from kwargs
    name:str = "" #dataset name (only for supervised training during preprocessing)
    path:str = "" # help='path to dataset')
    path_label:str= "" #separate path to annotations (by default will be inferred from path)
    fs:float = 100. #input sampling frequency

    col_train_fold:str = "strat_fold"#column in the dataset used to select the training set
    col_val_fold:str ="strat_fold"#column in the dataset used to select the validation set
    col_test_fold:str = "strat_fold"#column in the dataset used to select the test set
    train_fold_ids:List[int]=field(default_factory=lambda: [])#by default: 0...(n-3)- use negative numbers to select all except these numbers e.g. -3 all except fold 3
    val_fold_ids:List[int]=field(default_factory=lambda: [])#by default: n-2
    test_fold_ids:List[int]=field(default_factory=lambda: [])#by default: n-1

    input_channels_filter:List[int] = field(default_factory=lambda: []) #integer array to specify which channels to select []: use all

    label_filter:List[int] = field(default_factory=lambda: [])#supervised only: filter out certain labels from loss calculation and evaluation [] to include all
    annotation:bool = False # True for sequence annotation
    fs_annotation:float = 100.#sampling frequency for annotations; for PSG this should be 1./(30.*fs)
    
    #label aggregation i.e. aggregate some number of labels into a new one
    label_aggregation_epoch_length:int= -1 #how labels in the original sequence should be aggregated- 0: all -1: no label_aggregation
    label_aggregation_majority_vote:bool = False #decide on segment label via majority vote (i.e. single label per segment)
    label_aggregation_binary:bool = False #only count which segments are present (multi-label) rather than for what fraction of steps
    
    
@dataclass
class TrainerConfig:
    mainclass:str= "clinical_ts.cpc_main.CPCMain"#full qualifier i.e. file.classname
    executable:str = ""
    revision:str = ""
    username:str = ""

    export_features:bool = True
    export_predictions:bool = True
    
    epochs:int=50 # help='number of total epochs to run')
    frozen_epochs:int=0 # number of epochs of the above during which only the head is trained
    num_workers:int=4 #number of works in the dataloader

    resume:str=''# help='path to latest checkpoint (default: none)')
    pretrained:str='' # help='path to pretrained checkpoint (default: none)')
    eval_only:str='' # path to checkpoint for evaluation
       
    output_path:str='.'# help='output path')
    metadata:str='' # help='metadata for output') 
    
    gpus:int=1 # help="number of gpus")
    num_nodes:int=1 # help="number of compute nodes")
    precision:int=16 # help="16/32")
    distributed_backend:Any=None #, help="None/ddp")
    accumulate:int=1 # help="accumulate grad batches (total-bs=accumulate-batches*bs)")
        
    lr_find:bool=False #run lr finder before training run")
    auto_batch_size:bool=False # help="run batch size finder before training run")

    refresh_rate:int=0 # help="progress bar refresh rate (0 to disable)", default=0)

@dataclass
class EpochEncoderRNNConfig(EpochEncoderConfig):
    encoder: RNNEncoderConfig = field(default_factory=RNNEncoderConfig)
    predictor: RNNPredictorConfig = field(default_factory=RNNPredictorConfig)
    head: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)

@dataclass
class EpochEncoderTransformerConfig(EpochEncoderConfig):
    encoder: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)
    predictor: TransformerPredictorConfig = field(default_factory=TransformerPredictorConfig)
    head: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)

@dataclass
class EpochEncoderS4Config(EpochEncoderConfig):
    encoder: TransformerEncoderConfig = field(default_factory=TransformerEncoderConfig)
    predictor: S4PredictorConfig = field(default_factory=S4PredictorConfig)
    head: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)

@dataclass
class EpochEncoderNoneS4Config(EpochEncoderConfig):
    encoder: NoEncoderConfig = field(default_factory=NoEncoderConfig)
    predictor: S4PredictorConfig = field(default_factory=S4PredictorConfig)
    head: PoolingHeadConfig = field(default_factory=PoolingHeadConfig)

# https://hydra.cc/docs/tutorials/structured_config/config_groups/
@dataclass
class CPCConfig:

    base: BaseConfig
    data: BaseConfigData
    encoder: EncoderBaseConfig
    predictor: PredictorBaseConfig
    loss: LossConfig
    encoder_static: EncoderStaticBaseConfig
    head: HeadBaseConfig
    quantizer: QuantizerBaseConfig #NOTE: quantizer only active to produce targets during pretraining

    trainer: TrainerConfig
    extra: ExtraConfig

def create_default_config():
    cs = ConfigStore.instance()
    cs.store(name="config", node=CPCConfig)

    cs.store(group="base", name="base", node=BaseConfig)

    cs.store(group="data", name="base", node=BaseConfigData)
    
    cs.store(group="encoder", name="none", node=NoEncoderConfig)
    cs.store(group="encoder", name="rnn", node=RNNEncoderConfig)
    cs.store(group="encoder", name="tf", node=TransformerEncoderConfig)
    cs.store(group="encoder", name="spec2d", node=SpecConv2dEncoderConfig)#for spectrogram input
    cs.store(group="encoder", name="spec1d", node=SpecConv1dEncoderConfig)#for spectrogram input

    #epoch encoders

    cs.store(group="encoder", name="eernn", node=EpochEncoderRNNConfig)
    cs.store(group="encoder", name="eetf", node=EpochEncoderTransformerConfig)
    cs.store(group="encoder", name="ees4", node=EpochEncoderS4Config)
    cs.store(group="encoder", name="eens4", node=EpochEncoderNoneS4Config)
    
    cs.store(group="encoder_static", name="basic", node=BasicEncoderStaticConfig)
    cs.store(group="encoder_static", name="none", node=EncoderStaticBaseConfig)

    cs.store(group="predictor", name="none", node=NoPredictorConfig)
    cs.store(group="predictor", name="rnn", node=RNNPredictorConfig)
    cs.store(group="predictor", name="tf", node=TransformerPredictorConfig)
    cs.store(group="predictor", name="s4", node=S4PredictorConfig)

    
    #single prediction heads
    cs.store(group="head", name="rnn", node=RNNHeadConfig)
    cs.store(group="head", name="tfg", node=TransformerHeadGlobalConfig)
    #multi prediction heads
    cs.store(group="head", name="tfm", node=TransformerHeadMultiConfig)
    #universal heads
    cs.store(group="head", name="mlp", node=MLPHeadConfig)
    cs.store(group="head", name="s4", node=S4HeadConfig)
    cs.store(group="head", name="pool", node=PoolingHeadConfig)

    cs.store(group="quantizer", name="gumbel", node=GumbelQuantizerConfig)
    cs.store(group="quantizer", name="none", node=QuantizerBaseConfig)

    #supervised losses
    cs.store(group="loss", name="ce", node=CELossConfig)
    cs.store(group="loss", name="cef", node=CEFLossConfig)


    cs.store(group="trainer", name="trainer", node=TrainerConfig)

    return cs
