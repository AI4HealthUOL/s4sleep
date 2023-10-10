import os
import subprocess
import importlib
import shutil

from matplotlib import pyplot as plt
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from hydra.core.hydra_config import HydraConfig

#################
#specific
from clinical_ts.cpc_config import *
from omegaconf import OmegaConf
from clinical_ts.cpc_main import *

MLFLOW_AVAILABLE=True
try:
    import mlflow
    import mlflow.pytorch
except ImportError:
    MLFLOW_AVAILABLE=False

def get_git_revision_short_hash():
    return str(subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip())

def _string_to_class(_target_):
    if(len(_target_.split("."))==1):#assume global namespace
        cls_ = globals()[_target_]
    else:
        mod_ = importlib.import_module(".".join(_target_.split(".")[:-1]))
        cls_ = getattr(mod_, _target_.split(".")[-1])
    return cls_
        
###################################################################################################
#MAIN
###################################################################################################
cs = create_default_config()
    
@hydra.main(version_base=None, config_path="conf",  config_name="config_supervised")
def run(hparams: CPCConfig) -> None:
    hparams.trainer.executable = "main"
    hparams.trainer.revision = get_git_revision_short_hash()

    
    
    if not os.path.exists(hparams.trainer.output_path):
        os.makedirs(hparams.trainer.output_path)
    
    logger = TensorBoardLogger(
        save_dir=hparams.trainer.output_path,
        #version="",#hparams.trainer.metadata.split(":")[0],
        name="")
    classname = _string_to_class(hparams.trainer.mainclass)
    model = classname(hparams)
    #update output path
    hparams.trainer.output_path = Path(hparams.trainer.output_path)/logger.log_dir
    print("Output directory:",hparams.trainer.output_path)

    #get hydra configs
    hydra_cfg = HydraConfig.get()
    config_file = Path(hydra_cfg.runtime.config_sources[1]["path"])/hydra_cfg.job.config_name
    print("Main config:",config_file)
    print("Overrides:",OmegaConf.to_container(hydra_cfg.overrides.hydra))
    print("Runtime choices:",OmegaConf.to_container(hydra_cfg.runtime.choices))
    #print("Full config:",OmegaConf.to_yaml(hparams))
    #copy main config into output dir
    if not os.path.exists(hparams.trainer.output_path):
        os.makedirs(hparams.trainer.output_path)
    shutil.copyfile(config_file, Path(hparams.trainer.output_path)/(config_file.stem))

    #save full config
    #save_args_json(hparams,Path(logger.log_dir)/"config.json")  
    if(MLFLOW_AVAILABLE):
        #os.environ['MLFLOW_TRACKING_USERNAME'] = "ai4h"
        #os.environ['MLFLOW_TRACKING_PASSWORD'] = "mlf22!"
        #os.environ['MLFLOW_TRACKING_URI'] = "https://ai4hmlflow.nsupdate.info/"
        mlflow.set_experiment(hparams.trainer.executable+"("+hparams.trainer.mainclass.split(".")[-1]+")")
        mlflow.pytorch.autolog(log_models=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="best_model",
        save_top_k=1,
		save_last=True,
        verbose=True,
        monitor=(hparams.base.metrics[0]+'_agg0' if hparams.base.aggregate_predictions else hparams.base.metrics[0]+'_noagg0') if hparams.loss.loss_type=="supervised" else 'val_loss',#val_loss/dataloader_idx_0
        mode='max' if hparams.loss.loss_type=="supervised" else 'min')

    lr_monitor = LearningRateMonitor(logging_interval="step")
    lr_monitor2 = LRMonitorCallback(start=False,end=True)#interval="step")
    
    callbacks = [checkpoint_callback,lr_monitor,lr_monitor2]
    if(hparams.trainer.refresh_rate>0):
        callbacks.append(TQDMProgressBar(refresh_rate=hparams.trainer.refresh_rate))
    if(not hparams.loss.loss_type=="supervised" and hparams.loss.pretraining_targets>0):#quantizer
        callbacks.append(DecayTemperature(num_steps=3000))
        callbacks.append(RampBeta(num_steps=1000,betaend=hparams.quantizer_pretr.quantizer_loss_factor))
    if(hparams.loss.loss_type=="supervised" and hparams.trainer.frozen_epochs>0):
        callbacks.append(UnfreezingFinetuningCallback(unfreeze_epoch=hparams.trainer.frozen_epochs))

    trainer = pl.Trainer(
        #overfit_batches=0.01,
        auto_scale_batch_size = 'binsearch' if hparams.trainer.auto_batch_size else None,
        auto_lr_find = hparams.trainer.lr_find,
        accumulate_grad_batches=hparams.trainer.accumulate,
        max_epochs=hparams.trainer.epochs if hparams.trainer.eval_only=="" else 0,
    
        default_root_dir=hparams.trainer.output_path,
        
        #debugging flags for val and train
        num_sanity_val_steps=0,
        #overfit_batches=10,
        
        logger=logger,
        callbacks = callbacks,
        benchmark=True,
    
        gpus=hparams.trainer.gpus,
        num_nodes=hparams.trainer.num_nodes,
        precision=hparams.trainer.precision,
        #distributed_backend=hparams.distributed_backend,
        
        enable_progress_bar=hparams.trainer.refresh_rate>0,
        #weights_summary='top',
        )
        
    if(hparams.trainer.auto_batch_size):#batch size
        trainer.tune(model)

    if(hparams.trainer.lr_find):# lr find
        #torch.save(model.state_dict(), Path(hparams.trainer.output_path)/(logger.log_dir+"initial_weights.ckpt"))
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model)

        # Plot lr find plot
        fig = lr_finder.plot(suggest=True)
        fig.show()
        plt.savefig(Path(hparams.trainer.output_path)/(logger.log_dir+"/lrfind.png"))

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print("Suggested lr:",new_lr)
        # update hparams of the model
        model.hparams.base.lr = new_lr
        model.lr = new_lr

        # there is still some issue with the restored model- therefore just abort the run
        #model.load_state_dict(torch.load(Path(hparams.trainer.output_path)/(logger.log_dir+"initial_weights.ckpt")))
        return

    if(hparams.trainer.epochs>0 and hparams.trainer.eval_only==""):
        if(MLFLOW_AVAILABLE):
            with mlflow.start_run(run_name=hparams.trainer.metadata) as run:
                log_params_from_omegaconf_dict(hparams)
                trainer.fit(model,ckpt_path= None if hparams.trainer.resume=="" else hparams.trainer.resume)
                trainer.test(model,ckpt_path="best")
        else:
            trainer.fit(model,ckpt_path= None if hparams.trainer.resume=="" else hparams.trainer.resume)
            trainer.test(model,ckpt_path="best")

    elif(hparams.trainer.eval_only!=""):#eval only
    #else:
        if(MLFLOW_AVAILABLE):
            with mlflow.start_run(run_name=hparams.trainer.metadata) as run:
                log_params_from_omegaconf_dict(hparams)
                trainer.fit(model)#mock fit call as mlflow logging is only invoked for fit
                trainer.test(model,ckpt_path=hparams.trainer.eval_only)
        else:
            trainer.fit(model)#mock fit call as mlflow logging is only invoked for fit
            trainer.test(model,ckpt_path=hparams.trainer.eval_only)

    if(hparams.trainer.export_features):
        model.export_features(hparams.trainer.output_path/"features")

if __name__ == "__main__":
    run()
