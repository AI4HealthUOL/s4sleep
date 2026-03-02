import os
import shutil
import builtins
import importlib
import operator
import typing

from matplotlib import pyplot as plt
from pathlib import Path

import torch
import lightning.pytorch as lp
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, EarlyStopping
from clinical_ts.utils.callbacks import LRMonitorCallback, TriggerQuantizerHyperparameterUpdate, UnfreezingFinetuningCallback

import hydra
from hydra.core.hydra_config import HydraConfig

#################
#specific
from clinical_ts.config import *
from clinical_ts.utils.basic_io import get_git_revision_short_hash, get_slurm_job_id, get_work_dir, _string_to_class
from omegaconf import OmegaConf

def _resolve_global_symbol(symbol):
    if symbol == "_operator.getitem":
        return operator.getitem
    if symbol.startswith("typing."):
        return getattr(typing, symbol.split(".", 1)[1], None)
    if symbol.startswith("builtins."):
        return getattr(builtins, symbol.split(".", 1)[1], None)
    try:
        module_name, attr_name = symbol.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attr_name, None)
    except Exception:
        return None

def _allowlist_checkpoint_globals(checkpoint_path):
    if not hasattr(torch.serialization, "get_unsafe_globals_in_checkpoint"):
        return
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path == "" or not os.path.exists(checkpoint_path):
        return
    try:
        unsafe_symbols = torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path)
    except Exception as exc:
        print(f"[ckpt-safe-load] Failed to inspect checkpoint globals for {checkpoint_path}: {exc}")
        return
    if len(unsafe_symbols) == 0:
        return

    resolved = []
    unresolved = []
    for symbol in unsafe_symbols:
        symbol_obj = _resolve_global_symbol(symbol)
        if symbol_obj is None:
            unresolved.append(symbol)
        else:
            if hasattr(symbol_obj, "__module__") and hasattr(symbol_obj, "__qualname__"):
                resolved.append(symbol_obj)
            else:
                resolved.append((symbol_obj, symbol))

    if len(resolved) > 0 and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(resolved)
        print(f"[ckpt-safe-load] Added {len(resolved)} globals for safe checkpoint loading.")
    if len(unresolved) > 0:
        print(f"[ckpt-safe-load] Unresolved globals ({len(unresolved)}): {unresolved}")


#mlflow without autologging https://github.com/zjohn77/lightning-mlflow-hf/blob/74c30c784f719ea166941751bda24393946530b7/lightning_mlflow/train.py#L39
MLFLOW_AVAILABLE=True
try:
    import mlflow
    from lightning.pytorch.loggers import MLFlowLogger
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
                mlflow.log_param(f'{parent_name}.{i}', " " if v=="" else v)
    from mlflow.exceptions import MlflowException

    class RobustMLFlowLogger(MLFlowLogger):
        '''Robust MLFLowLogger that does not crash if connection is down'''
        def finalize(self, status: str) -> None:
            try:
                super().finalize(status)
            except (MlflowException, ConnectionError, TimeoutError) as e:
                print(f"MLflow finalization failed: {e}")
                print("Training completed successfully despite logging issues")
            except Exception as e:
                print(f"Unexpected error during MLflow finalization: {e}")
    
except ImportError:
    MLFLOW_AVAILABLE=False

###################################################################################################
#MAIN
###################################################################################################
cs = create_default_config()

@hydra.main(version_base=None, config_path="conf",  config_name="sedf_ts")
def run(hparams: FullConfig) -> None:
    hparams.trainer.executable = "main_all"
    hparams.trainer.revision = get_git_revision_short_hash()

    if hparams.trainer.seed is not None and int(hparams.trainer.seed) >= 0:
        lp.seed_everything(int(hparams.trainer.seed), workers=True)
        print(f"[seed] Using seed={int(hparams.trainer.seed)} (workers=True)")
    else:
        print("[seed] Disabled (trainer.seed < 0)")

    os.makedirs(hparams.trainer.output_path, exist_ok=True)
    
    #determine version/output path
    slurm_job_id = get_slurm_job_id()
    run_name = get_work_dir(hparams.trainer.output_path,pattern=slurm_job_id+"_")

    hparams.trainer.output_path = Path(hparams.trainer.output_path)/(run_name)
    os.makedirs(hparams.trainer.output_path, exist_ok=True)

    logger = [TensorBoardLogger(
        save_dir=Path(hparams.trainer.output_path).parent,
        version=run_name,#hparams.trainer.metadata.split(":")[0],
        name="")]
    
    print("FULL PARSED CONFIG:")
    print(OmegaConf.to_yaml(hparams))

    #get hydra configs
    hydra_cfg = HydraConfig.get()
    
    # Attempt to find the config file path
    config_sources = hydra_cfg.runtime.config_sources
    config_path = None
    for source in config_sources:
        if source.provider == "file":
             config_path = source.path
             break
             
    if config_path is None and len(config_sources) > 1:
        config_path = config_sources[1]["path"] # Fallback to index 1

    config_file = Path(config_path)/hydra_cfg.job.config_name
    
    # Check if we need to add extension
    if not config_file.exists():
        if not str(config_file).endswith('.yaml'):
             config_file_yaml = config_file.with_suffix('.yaml')
             if config_file_yaml.exists():
                 config_file = config_file_yaml
    
    print("Output directory:",hparams.trainer.output_path)
    print("Main config:",config_file)
    print("Overrides:",OmegaConf.to_container(hydra_cfg.overrides.hydra))
    print("Runtime choices:",OmegaConf.to_container(hydra_cfg.runtime.choices))
    #print("Full config:",OmegaConf.to_yaml(hparams))
    
    #copy main config into output dir
    if config_file.exists():
        shutil.copyfile(config_file, Path(hparams.trainer.output_path)/(config_file.name))
    else:
        print(f"WARNING: Could not find main config file at {config_file} to copy.")

    #create the model
    classname = _string_to_class(hparams.task.mainclass)
    model = classname(hparams)
    #model = torch.compile(model)

    print(f"Run name: {run_name}")

    if(MLFLOW_AVAILABLE):
        #os.environ['MLFLOW_TRACKING_USERNAME'] = "ai4h"
        #os.environ['MLFLOW_TRACKING_PASSWORD'] = "mlf22!"
        #os.environ['MLFLOW_TRACKING_URI'] = "https://ai4hmlflow.nsupdate.info/"
        mlflow.set_experiment(hparams.trainer.executable+"("+hparams.task.mainclass.split(".")[-1]+")")
        
        mlflow_run = mlflow.start_run(run_name=run_name)
        mlflow.set_tag("mlflow.runName", run_name)

        mlf_logger = RobustMLFlowLogger(
            experiment_name=mlflow.get_experiment(mlflow_run.info.experiment_id).name,
            tracking_uri=mlflow.get_tracking_uri(),
            log_model=False,
        )
        mlf_logger._run_id = mlflow_run.info.run_id
        mlf_logger.log_hyperparams = log_params_from_omegaconf_dict       
        logger.append(mlf_logger)

    key_summary_metric = model.metrics_train_val[0].key_summary_metric if len(model.metrics_train_val)>0 else 'val_loss_total0'#use the key_summary_metric of the first metric otherwise val_loss0
    mode_summary_metric = model.metrics_train_val[0].mode_summary_metric if len(model.metrics_train_val)>0 else 'min'

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger[0].log_dir,
        filename="best_model",
        save_top_k=1,
		save_last=True,
        verbose=True,
        monitor=key_summary_metric,
        mode=mode_summary_metric,
        every_n_train_steps = hparams.trainer.checkpoint_every_n_train_steps if hparams.trainer.checkpoint_every_n_train_steps>0 else None,
        every_n_epochs = hparams.trainer.checkpoint_every_n_epochs if hparams.trainer.checkpoint_every_n_epochs>0 else None,)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    lr_monitor2 = LRMonitorCallback(start=False,end=True)#interval="step")
    
    callbacks = [checkpoint_callback,lr_monitor,lr_monitor2]
    if hparams.trainer.early_stopping_patience > 0:
        es_monitor = hparams.trainer.early_stopping_monitor if hparams.trainer.early_stopping_monitor != "" else key_summary_metric
        es_mode = hparams.trainer.early_stopping_mode if hparams.trainer.early_stopping_mode != "" else mode_summary_metric
        callbacks.append(
            EarlyStopping(
                monitor=es_monitor,
                mode=es_mode,
                patience=hparams.trainer.early_stopping_patience,
                min_delta=hparams.trainer.early_stopping_min_delta,
                verbose=True,
            )
        )
        print(f"[early-stopping] enabled monitor={es_monitor} mode={es_mode} patience={hparams.trainer.early_stopping_patience} min_delta={hparams.trainer.early_stopping_min_delta}")
    if(hparams.trainer.refresh_rate>0):
        callbacks.append(TQDMProgressBar(refresh_rate=hparams.trainer.refresh_rate))
    quantizers = [m for m in model.modules() if isinstance(m,QuantizerBase)]
    if(len(quantizers)>0):
        print("Found",len(quantizers),"quantizer modules.")
        callbacks.append(TriggerQuantizerHyperparameterUpdate(quantizers))
    if(hparams.loss.loss_type=="supervised" and hparams.trainer.frozen_epochs>0):
        callbacks.append(UnfreezingFinetuningCallback(unfreeze_epoch=hparams.trainer.frozen_epochs))

    trainer = lp.Trainer(
        #overfit_batches=0.01,
        accumulate_grad_batches=hparams.trainer.accumulate,
        max_epochs=hparams.trainer.epochs if hparams.trainer.eval_only=="" else 0,
    
        default_root_dir=hparams.trainer.output_path,
        
        #debugging flags for val and train
        num_sanity_val_steps=0,
        #overfit_batches=10,
        
        logger=logger,
        callbacks = callbacks,
        benchmark=not hparams.trainer.deterministic,
    
        accelerator="gpu" if hparams.trainer.gpus>0 else "cpu",
        devices=hparams.trainer.gpus if hparams.trainer.gpus>0 else 1,
        num_nodes=hparams.trainer.num_nodes,
        precision=hparams.trainer.precision,
        strategy=hparams.trainer.strategy,
        deterministic=hparams.trainer.deterministic,
        
        enable_progress_bar=hparams.trainer.refresh_rate>0,
        #weights_summary='top',
        )
    
    if(hparams.trainer.fp32_matmul_precision!="highest"):
        torch.set_float32_matmul_precision(hparams.trainer.fp32_matmul_precision)
        
    if(hparams.trainer.auto_batch_size):#batch size
        tuner=Tuner(trainer)
        tuner.scale_batch_size(model, mode="binsearch")

    if(hparams.trainer.lr_find):# lr find
        tuner=Tuner(trainer)

        #torch.save(model.state_dict(), Path(hparams.trainer.output_path)/(logger.log_dir+"initial_weights.ckpt"))
        # Run learning rate finder
        lr_finder = tuner.lr_find(model)

        # Plot lr find plot
        fig = lr_finder.plot(suggest=True)
        fig.show()
        plt.savefig(Path(hparams.trainer.output_path)/("lrfind.png"))

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        print("Suggested lr:",new_lr)
        # update hparams of the model
        model.hparams.base.lr = [new_lr]
        model.lr = new_lr

        # there is still some issue with the restored model- therefore just abort the run
        #model.load_state_dict(torch.load(Path(hparams.trainer.output_path)/(logger.log_dir+"initial_weights.ckpt")))
        return

    if(MLFLOW_AVAILABLE):
        #version_number = hparams.trainer.output_path.split('/')[-1].replace('version_', '') # tw: extract version index
        mlflow.log_param("a_slurm_job_id", slurm_job_id)
        mlflow.log_param("a_run_name", run_name)
        mlflow.log_param("a_git_hash", get_git_revision_short_hash())
    
    if(hparams.trainer.epochs>0 and hparams.trainer.eval_only==""):
        if hparams.trainer.resume != "":
            _allowlist_checkpoint_globals(hparams.trainer.resume)
        trainer.fit(model,ckpt_path= None if hparams.trainer.resume=="" else hparams.trainer.resume)
    if hparams.trainer.skip_test_after_fit:
        print("Skipping trainer.test() because trainer.skip_test_after_fit=True")
    else:
        test_ckpt_path = hparams.trainer.eval_only if hparams.trainer.eval_only!="" else checkpoint_callback.best_model_path
        _allowlist_checkpoint_globals(test_ckpt_path)
        trainer.test(model,ckpt_path=test_ckpt_path if test_ckpt_path!="" else None)

    if(MLFLOW_AVAILABLE):
        mlflow.end_run()
    
    if(hparams.trainer.export_features!=""):
        model.export_features(Path(hparams.trainer.output_path)/"features",module=hparams.trainer.export_features)

if __name__ == "__main__":
    #specify config path: --config-dir=code/conf
    #specify config name: --config-name=config_supervised_ecg_s4
    #alternatively directly the config file: --config-file=code/conf/config_supervised_ecg_s4.yaml   
    run()
