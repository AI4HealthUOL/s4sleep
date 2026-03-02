from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

#for base configs
from .template_modules import * 

#for specific configs
from .ts.encoder import *
from .ts.encoder_epoch import *
from .ts.head import *
from .ts.base import *
from .ts.rnn import *
from .ts.transformer import *
from .ts.encoder_spec import *

##################################################################
# try to import (optional) modules with special dependencies
S4_AVAILABLE = True
try:
    from .ts.s4 import *
except Exception as e:
    print(f"WARNING: Could not import s4 module: {e}")
    S4_AVAILABLE = False

from .loss.supervised import *
from .metric.base import *
from .task.psg import *


###########################################################################################################
# https://hydra.cc/docs/tutorials/structured_config/config_groups/
@dataclass
class FullConfig:

    base: BaseConfig
    data: BaseConfigData
    loss: LossConfig
    metric: MetricConfig
    trainer: TrainerConfig
    task: TaskConfig

    ts: TimeSeriesEncoderConfig
    static: EncoderStaticBaseConfig
    head: HeadBaseConfig
    

def create_default_config():
    cs = ConfigStore.instance()
    cs.store(name="config", node=FullConfig)

    ######################################################################
    # base
    ######################################################################
    cs.store(group="base", name="base", node=BaseConfig)

    ######################################################################
    # input data
    ######################################################################
    cs.store(group="data", name="base", node=BaseConfigData)
    
    ######################################################################
    # time series encoder
    ######################################################################
    cs.store(group="ts", name="tsenc",  node=TimeSeriesEncoderConfig)
    
    #ENCODER
    cs.store(group="ts/enc", name="none", node=NoEncoderConfig)
    cs.store(group="ts/enc", name="rnn", node=RNNEncoderConfig)
    cs.store(group="ts/enc", name="tf", node=TransformerEncoderConfig)
    cs.store(group="ts/enc", name="spec2d", node=SpecConv2dEncoderConfig)#for spectrogram input
    cs.store(group="ts/enc", name="spec1d", node=SpecConv1dEncoderConfig)#for spectrogram input

    cs.store(group="ts/enc", name="ts", node=TimeSeriesEncoderConfig)#generic time series encoder

    cs.store(group="ts/enc", name="eernn", node=EpochEncoderRNNConfig)#CNN+LSTM epoch encoder
    cs.store(group="ts/enc", name="eetf", node=EpochEncoderTransformerConfig)#CNN+TF epoch encoder
    if(S4_AVAILABLE):
        cs.store(group="ts/enc", name="ees4", node=EpochEncoderS4Config)#CNN+S4 epoch encoder
        cs.store(group="ts/enc", name="eens4", node=EpochEncoderNoneS4Config)#NONE+S4 epoch encoder (e.g. for use with spectrograms)
    
    #PREDICTOR
    cs.store(group="ts/pred", name="none", node=NoPredictorConfig)#no predictor
    cs.store(group="ts/pred", name="rnn", node=RNNPredictorConfig)#LSTM/GRU
    cs.store(group="ts/pred", name="tf", node=TransformerPredictorConfig)#transformer
    if(S4_AVAILABLE):
        cs.store(group="ts/pred", name="s4", node=S4PredictorConfig)#S4 model
    cs.store(group="ts/pred", name="cnn", node=CNNPredictorConfig)#basic cnn
    
    #HEADS
    cs.store(group="ts/head", name="none", node=HeadBaseConfig)
    #single output token prediction heads
    cs.store(group="ts/head", name="rnn", node=RNNHeadConfig)
    cs.store(group="ts/head", name="tfg", node=TransformerHeadGlobalConfig)
    cs.store(group="ts/head", name="attn", node=AttentionPoolingHeadConfig)
    cs.store(group="ts/head", name="lqattn", node=LearnableQueryAttentionPoolingHeadConfig)
    
    #multi-token output prediction heads
    cs.store(group="ts/head", name="tfm", node=TransformerHeadMultiConfig)
    #universal heads
    cs.store(group="ts/head", name="mlp", node=MLPHeadConfig)
    cs.store(group="ts/head", name="mlpr", node=MLPRegressionHeadConfig)
    if(S4_AVAILABLE):
        cs.store(group="ts/head", name="s4", node=S4HeadConfig)
    cs.store(group="ts/head", name="pool", node=PoolingHeadConfig)
    cs.store(group="ts/head", name="flat", node=FlattenHeadConfig)
    cs.store(group="ts/head", name="poolflat", node=PoolingFlattenHeadConfig)
    cs.store(group="ts/head", name="poolflatmlp", node=PoolingFlattenMLPHeadConfig)
    cs.store(group="ts/head", name="flatmlp", node=FlattenMLPHeadConfig)
    cs.store(group="ts/head", name="seq", node=SequentialHeadConfig)#generic sequential head

    #QUANTIZER
    cs.store(group="ts/quant", name="none", node=QuantizerBaseConfig)

    #MASK
    cs.store(group="ts/mask", name="none", node=MaskingBaseConfig)

    #LOSS (SSL)
    cs.store(group="ts/loss", name="none", node=SSLLossConfig)

    #PRE
    cs.store(group="ts/pre", name="none", node=PrePostBaseConfig)  
    cs.store(group="ts/pre", name="ee", node=EpochEncoderPreConfig)
    
    #POST
    cs.store(group="ts/post", name="none", node=PrePostBaseConfig)  
    cs.store(group="ts/post", name="ee", node=EpochEncoderPostConfig)

    #EMA
    cs.store(group="ts/ema", name="none", node=EMATimeSeriesEncoderBaseConfig)
        
    ##################################################################
    # static encoder
    ######################################################################
    for g in ["static", "ts/static"]:
        cs.store(group=g, name="none", node=EncoderStaticBaseConfig)

    ######################################################################
    # optional multimodal head
    ######################################################################
    cs.store(group="head", name="none", node=HeadBaseConfig)
    cs.store(group="head", name="rnn", node=RNNHeadConfig)

    ######################################################################
    # loss function
    ######################################################################
    #no global loss
    cs.store(group="loss", name="none", node=LossConfig)
    #supervised losses
    cs.store(group="loss", name="ce", node=CELossConfig)
    cs.store(group="loss", name="cef", node=CEFLossConfig)
    cs.store(group="loss", name="bce", node=BCELossConfig)
    cs.store(group="loss", name="bcef", node=BCEFLossConfig)
    cs.store(group="loss", name="mse", node=MSELossConfig)
    
    ######################################################################
    # metrics
    ######################################################################
    cs.store(group="metric", name="none", node=MetricConfig)
    cs.store(group="metric", name="auroc", node=MetricAUROCConfig)
    cs.store(group="metric", name="aurocagg", node=MetricAUROCAggConfig)
    cs.store(group="metric", name="aupr", node=MetricAUPRConfig)
    cs.store(group="metric", name="aupragg", node=MetricAUPRAggConfig)
    cs.store(group="metric", name="mae", node=MetricMAEConfig)
    cs.store(group="metric", name="maeagg", node=MetricMAEAggConfig)
    cs.store(group="metric", name="fbeta", node=MetricFbetaConfig)
    cs.store(group="metric", name="fbetaagg", node=MetricFbetaAggConfig)
    cs.store(group="metric", name="f1", node=MetricF1Config)
    cs.store(group="metric", name="f1agg", node=MetricF1AggConfig)
    cs.store(group="metric", name="acc", node=MetricAccuracyConfig)
    cs.store(group="metric", name="accagg", node=MetricAccuracyAggConfig)
    cs.store(group="metric", name="sensspec", node=MetricSensitivitySpecificityConfig)
    cs.store(group="metric", name="sensspecagg", node=MetricSensitivitySpecificityAggConfig)

    ######################################################################
    # trainer
    ######################################################################
    cs.store(group="trainer", name="trainer", node=TrainerConfig)
    
    ######################################################################
    # task
    ######################################################################
    cs.store(group="task", name="none", node=TaskConfig)
    cs.store(group="task", name="psg", node=TaskConfigPSG)
    
    return cs
