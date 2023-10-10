###############
#generic
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

#################
#specific
from .timeseries_utils import *

from sklearn.metrics import accuracy_score, f1_score, classification_report

from pathlib import Path
import numpy as np
import pandas as pd
try:
    import pickle5 as pickle
except ImportError as e:
    import pickle
import os

from .eval_utils_cafa import eval_scores, eval_scores_bootstrap

from .cpc_template import *

##################
#utilities
def _to_int(val,fs,target_fs):
    return int(np.round(val*fs/target_fs)) if isinstance(val,int) else int(np.round(val*fs))

class ForwardHook:
    "Create a forward hook on module `m` "

    def __init__(self, m, store_output=True):
        self.store_output = store_output
        self.hook = m.register_forward_hook(self.hook_fn)
        self.stored, self.removed = None, False

    def hook_fn(self, module, input, output):
        "stores input/output"
        if self.store_output:
            self.stored = output
        else:
            self.stored = input

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()

############################################################################################################
class CPCMain(CPCTemplate):

    def __init__(self, hparams):
        #fix default chunk_length and stride hyperparameters
        if(hparams.base.chunk_length_train==-1):
            hparams.base.chunk_length_train = hparams.base.input_size
        if(hparams.base.stride_train==-1):
            hparams.base.stride_train = hparams.base.chunk_length_train
        if(hparams.base.chunk_length_val==-1):
            hparams.base.chunk_length_val = hparams.base.input_size
        if(hparams.base.stride_val==-1):
            hparams.base.stride_val = hparams.base.chunk_length_val
        if(hparams.base.chunk_length_valtest==-1):
            hparams.base.chunk_length_valtest = hparams.base.input_size
        if(hparams.base.stride_valtest==-1):
            hparams.base.stride_valtest = hparams.base.chunk_length_valtest
        if(hparams.base.stride_export==-1):
            hparams.base.stride_export = hparams.base.input_size
        super().__init__(hparams)
        
        self.score = None #keep track of best scores (model selection) across epochs
    
    def validation_epoch_end(self, outputs_all):
        return self._valtest_epoch_end(outputs_all, test=False)

    def test_epoch_end(self, outputs_all):
        return self._valtest_epoch_end(outputs_all, test=True)

    def _valtest_epoch_end(self, outputs_all, test):
        if(self.hparams.loss.loss_type=="supervised"):
            
            results_dict = {}
            for dataloader_idx,outputs in enumerate([outputs_all] if isinstance(outputs_all[0],dict) else outputs_all): #multiple val dataloaders or not
                aggregate_predictions_flag = self.hparams.base.aggregate_predictions and not self.trainer.sanity_checking #skip aggregation during sanity check
                id_map = self.val_idmap if test is False else self.test_idmaps[dataloader_idx]
                if(self.hparams.head.multi_prediction):
                    if(test and self.hparams.base.aggregate_strided_multi_predictions and not self.trainer.sanity_checking):#aggregate multi-predictions
                        
                        preds = torch.cat([x['preds'] for x in outputs]).cpu()# preds have shape bs,seq,classes
                        if(self.hparams.loss.supervised_type == "classification_single"):#apply softmax/sigmoid before aggregation
                            preds = F.softmax(preds.float(),dim=-1)
                        elif(self.hparams.loss.supervised_type == "classification_multi"):
                            preds = torch.sigmoid(preds.float())
                        targs = torch.cat([x['targs'] for x in outputs]).cpu()
                        targs_all = []
                        preds_all = []
                        stridetmp = int(self.hparams.base.stride_valtest/self.hparams.base.input_size*preds.shape[1]) if test is True else int(self.hparams.base.stride_val/self.hparams.base.input_size*preds.shape[1])

                        for x in np.unique(id_map):
                            idtmp = np.where(id_map==x)[0]
                            predstmp = torch.zeros((preds.shape[1]+(len(idtmp)-1)*stridetmp,preds.shape[2]),dtype=torch.float32)
                            predstmp_weight = torch.zeros(predstmp.shape[0],dtype=torch.int64)
                            targstmp = torch.zeros(predstmp.shape[0] if len(targs.shape)==2 else (predstmp.shape[0],targs.shape[-1]),dtype=torch.int64)
                            for i,(p,t) in enumerate(zip(preds[idtmp],targs[idtmp])):
                                start_idx = i*stridetmp
                                predstmp[start_idx:start_idx+preds.shape[1]]+=p
                                predstmp_weight[start_idx:start_idx+preds.shape[1]]+=1
                                targstmp[start_idx:start_idx+preds.shape[1]]=t
                            predstmp=predstmp/predstmp_weight.unsqueeze(-1)#take the weighted mean of all predictions
                            preds_all.append(predstmp)
                            targs_all.append(targstmp)
                        preds_all=torch.cat(preds_all,dim=0)
                        targs_all=torch.cat(targs_all,dim=0)

                    else:#naive approach: just concatenate everything
                        preds_all = torch.cat([x['preds'].view(-1,x['preds'].shape[-1]) for x in outputs])#flatten prediction
                        if(self.hparams.loss.supervised_type == "classification_single"):
                            preds_all = F.softmax(preds_all,dim=-1)
                        elif(self.hparams.loss.supervised_type == "classification_multi"):
                            preds_all = torch.sigmoid(preds_all)
                        targs_all = torch.cat([x['targs'] for x in outputs])
                else:#no multi-prediction
                    preds_all = torch.cat([x['preds'] for x in outputs])
                    if(self.hparams.loss.supervised_type == "classification_single"):
                        preds_all = F.softmax(preds_all,dim=-1)
                    elif(self.hparams.loss.supervised_type == "classification_multi"):
                        preds_all = torch.sigmoid(preds_all)
                    targs_all = torch.cat([x['targs'] for x in outputs])
                #export predictions
                if(test and self.hparams.trainer.export_predictions):
                    np.savez(Path(self.hparams.trainer.output_path)/("preds_val.npz" if dataloader_idx==0 else "preds_test.npz"),preds_all.cpu().numpy(),targs_all.cpu().numpy())

                if(self.hparams.loss.supervised_type == "classification_single"):
                    if(np.any([len(d.label_filter)>0 for d in [self.hparams[d] for d in self.dataset_keys]])): #filter out labels we don't care about
                        preds_all=preds_all[targs_all.view(-1)>=0]
                        targs_all=targs_all.view(-1)[targs_all.view(-1)>=0]
                    #preds_all = F.softmax(preds_all,dim=-1)
                    targs_all = torch.eye(len(self.lbl_itos))[targs_all.view(-1)].to(preds_all.device)#flatten targets 
                elif(self.hparams.loss.supervised_type == "classification_multi"):
                    #preds_all = torch.sigmoid(preds_all)
                    targs_all = targs_all.view(-1,targs_all.shape[-1])#flatten targets
                preds_all = preds_all.cpu().numpy()
                targs_all = targs_all.cpu().numpy()
                if(aggregate_predictions_flag):
                    preds_all_agg_mean,targs_all_agg_mean = aggregate_predictions(preds_all,targs_all,id_map,aggregate_fn=np.mean)
                    preds_all_agg_max,targs_all_agg_max = aggregate_predictions(preds_all,targs_all,id_map,aggregate_fn=np.max)
                    
                #instance level score
                if("macro_auc" in self.hparams.base.metrics):
                    res = eval_scores(targs_all.astype(np.int64),preds_all,classes=self.lbl_itos)
                    print("epoch",self.current_epoch,("test_" if test else "")+"macro_auc_noagg"+str(dataloader_idx)+":",res["label_AUC"]["macro"])
                    for k in res["label_AUC"].keys():
                        results_dict[("test_" if test else "")+k.replace(" ","_").replace("|","_").replace("(","_").replace(")","_")+"_auc_noagg"+str(dataloader_idx)] = res["label_AUC"][k]

                    if(aggregate_predictions_flag):
                        res_agg = eval_scores(targs_all_agg_mean,preds_all_agg_mean,classes=self.lbl_itos)
                        print("epoch",self.current_epoch,("test_" if test else "")+"macro_auc_agg"+str(dataloader_idx)+":",res_agg["label_AUC"]["macro"])
                        for k in res_agg["label_AUC"].keys():
                            results_dict[("test_" if test else "")+k.replace(" ","_").replace("|","_").replace("(","_").replace(")","_")+"_auc_agg"+str(dataloader_idx)] = res_agg["label_AUC"][k]
                        
                    #label aucs
                    #print("epoch",self.current_epoch,"label_auc_agg"+str(dataloader_idx)+":",res_agg["label_AUC"])

                thresholded_metrics_dict={"accuracy":accuracy_score, "f1":f1_score}
                thresholded_metrics = [m for m in self.hparams.base.metrics if m in thresholded_metrics_dict.keys()]

                if(len(thresholded_metrics)>0):
                    preds_all_bin= np.argmax(preds_all,axis=1)
                    targs_all_bin= np.argmax(targs_all,axis=1)
                    print("epoch",self.current_epoch,("test_" if test else "")+"dl",str(dataloader_idx), "noagg\n", classification_report(targs_all_bin, preds_all_bin, labels=range(len(self.lbl_itos)),target_names=self.lbl_itos))
                    if(aggregate_predictions_flag):
                        preds_all_agg_mean_bin = np.argmax(preds_all_agg_mean,axis=1)
                        targs_all_agg_mean_bin = np.argmax(targs_all_agg_mean,axis=1)
                        preds_all_agg_max_bin = np.argmax(preds_all_agg_max,axis=1)
                        targs_all_agg_max_bin = np.argmax(targs_all_agg_max,axis=1)
                    for metric in thresholded_metrics:
                        kwargs = [{"average":"macro"},{"average":None}] if metric=="f1" else [{}]
                        for kw in kwargs:
                            score_noagg = thresholded_metrics_dict[metric](targs_all_bin,preds_all_bin, **kw)
                            if(isinstance(score_noagg,np.ndarray)):
                                for s,l in zip(score_noagg,self.lbl_itos):
                                    results_dict[("test_" if test else "")+metric+"_"+l.replace(" ","_").replace("|","_").replace("(","_").replace(")","_")+"_noagg"+str(dataloader_idx)]=s
                            else:
                                results_dict[("test_" if test else "")+metric+"_noagg"+str(dataloader_idx)]=score_noagg
                        if(aggregate_predictions_flag):
                            for ix,kw in enumerate(kwargs):
                                score_mean = thresholded_metrics_dict[metric](targs_all_agg_mean_bin,preds_all_agg_mean_bin,**kwargs)
                                score_max = thresholded_metrics_dict[metric](targs_all_agg_max_bin,preds_all_agg_max_bin,**kwargs)
                                if(isinstance(score_mean,np.ndarray)):
                                    for smean,smax,l in zip(score_noagg,self.lbl_itos):
                                        results_dict[("test_" if test else "")+metric+"_"+l.replace(" ","_").replace("|","_").replace("(","_").replace(")","_")+"_agg_mean"+str(dataloader_idx)]=smean
                                        results_dict[("test_" if test else "")+metric+"_"+l.replace(" ","_").replace("|","_").replace("(","_").replace(")","_")+"_agg_max"+str(dataloader_idx)]=smax
                                else:
                                    results_dict[("test_" if test else "")+metric+"_agg_mean"+str(dataloader_idx)]=score_mean
                                    results_dict[("test_" if test else "")+metric+"_agg_max"+str(dataloader_idx)]=score_max
                                if(ix==0):
                                    print("epoch",self.current_epoch,metric+"_mean"+str(dataloader_idx)+":",score_mean,"score_max"+str(dataloader_idx)+":",score_max)
            if(not self.trainer.sanity_checking):
                self.log_dict(results_dict)
                if(test):
                    with open(Path(self.hparams.trainer.output_path)/("scores_val.pkl" if dataloader_idx==1 else "scores_test.pkl"), 'wb') as handle:
                        results_dict["epoch"]=self.current_epoch
                        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                #log results of best model
                #results_dict["epoch"]=self.current_epoch
                #model_selection_key = (self.hparams.base.metrics[0]+'_agg0' if aggregate_predictions_flag else self.hparams.base.metrics[0]+'_noagg0')
        
                #if(self.score is None or self.score[model_selection_key]<results_dict[model_selection_key]):
                #    self.score = results_dict
                #self.log_dict({"best_model_"+k:v for k,v in self.score.items()})
                #print("best score val/test/epoch:", self.score[model_selection_key], self.score[model_selection_key[:-1]+"1"], self.score["epoch"] )

    def modify_dataset_kwargs(self,dataset_kwargs,stage="train"):
        '''possibility modify kwargs passed to dataset in derived classes'''
        return dataset_kwargs

    def setup(self, stage):

        self.dataset_keys = [d for d in self.hparams if ("_target_" in self.hparams[d].keys() and self.hparams[d]["_target_"]=="clinical_ts.cpc_config.BaseConfigData")]
        assert(len(self.dataset_keys)>0)
        
        train_datasets = []
        val_datasets = []
        test_datasets_val = []
        test_datasets_test = []
        self.export_datasets = []

        self.lbl_itos = None

        ds_mean = None
        ds_std = None

        for d in [self.hparams[d] for d in self.dataset_keys]:
            
            df_mapped, lbl_itos, mean, std = self.preprocess_dataset(d)
            if(ds_mean is None):
                ds_mean, ds_std = mean, std

            #build up transforms
            assert(self.hparams.base.fs==d.fs)#resampling on the fly coming soon
            
            #aggregate sequence labels if desired
            if(self.hparams.loss.loss_type=="supervised" and d.label_aggregation_epoch_length>=0):
                tfms_train =[SequenceToSampleLabelTransform(num_classes=len(lbl_itos),majority_vote=d.label_aggregation_majority_vote,binary=d.label_aggregation_binary,epoch_length=d.label_aggregation_epoch_length)]
                tfms_valtest =[SequenceToSampleLabelTransform(num_classes=len(lbl_itos),majority_vote=d.label_aggregation_majority_vote,binary=True,epoch_length=d.label_aggregation_epoch_length)]#always use binary labels during test
            else:
                tfms_train = []
                tfms_valtest = []

            tfms=[]
            if(self.hparams.loss.loss_type=="supervised" and len(d.label_filter)>0):
                if(self.hparams.loss.supervised_type == "classification_single"):
                    #map labels to consecutive range (remaining labels to -1)
                    exttoint_dict = {e:i for i,e in enumerate(d.label_filter)}
                    exttoint = np.array([exttoint_dict[i] if i in exttoint_dict.keys() else -1 for i in range(len(lbl_itos))])
                    tfms+=[Transform(lambda x:exttoint[x])]
                elif(self.hparams.loss.supervised_type == "classification_multi"):
                    tfms+=[Transform(lambda x:x[:,d.label_filter] if len(x)==2 else x[d.label_filter])]#multi-hot encoded: just select appropriate rows
                lbl_itos=lbl_itos[d.label_filter]
                
            if(self.hparams.base.normalize):
                tfms+=[Normalize(ds_mean,ds_std)]
            if(len(d.input_channels_filter)>0):#spectrograms have ts,ch,freq other ts,ch
                tfms+=[ChannelFilter(channels=d.input_channels_filter,axis=1 if self.hparams.base.freq_bins>0 else -1)]
            
            #obligatory ToTensor
            tfms_train+=tfms+[ToTensor()]
            tfms_valtest+=tfms+[ToTensor()]

            assert(self.lbl_itos is None or np.all(self.lbl_itos == lbl_itos))#make sure all lbl_itos are identical
            self.lbl_itos = lbl_itos

            def get_folds_ids(fold_ids, all_ids, stage="train_supervised"):#stage: train_supervised, train_unsupervised, val_supervised, val_unsupervised, test_supervised
                if(len(fold_ids)==0):#use default assignments
                    max_fold_id=max(all_ids)
                    if(stage.startswith("train")):#train
                        res=[x for x in all_ids if (x<max_fold_id-1 if "train_supervised" else x<max_fold_id)]
                    elif(stage.startswith("val")):#val
                        res=[max_fold_id-1 if "val_supervised" else max_fold_id]
                    else:#test
                        res=[max_fold_id]
                else:
                    pos_ids = [x for x in fold_ids if x>=0] 
                    neg_ids = [-x for x in fold_ids if x<0]
                    assert(len(pos_ids)==0 or len(neg_ids)==0)#either only negative or only positive ids
                    if(len(neg_ids)>0):
                        res = [x for x in fold_ids if not x in neg_ids]
                    else:
                        res = fold_ids
                return res   

            #determine fold ids
            assert((len(d.train_fold_ids)>0 and len(d.val_fold_ids)>0) or d.col_train_fold==d.col_val_fold)
            train_ids = get_folds_ids(d.train_fold_ids,np.unique(df_mapped[d.col_train_fold]),"train_supervised" if self.hparams.loss.loss_type=="supervised" else "train_unsupervised")
            val_ids = get_folds_ids(d.val_fold_ids,np.unique(df_mapped[d.col_val_fold]),"val_supervised" if self.hparams.loss.loss_type=="supervised" else "val_unsupervised")
            df_train = df_mapped[df_mapped[d.col_train_fold].apply(lambda x: x in train_ids)]
            df_val = df_mapped[df_mapped[d.col_val_fold].apply(lambda x: x in val_ids)]

            if(self.hparams.loss.loss_type=="supervised"):
                test_ids = get_folds_ids(d.test_fold_ids,np.unique(df_mapped[d.col_test_fold]),"test")
                df_test = df_mapped[df_mapped[d.col_test_fold].apply(lambda x: x in test_ids)]

            #prepare default kwargs
            target_folder = Path(d.path)
            kwargs_train = {"df":df_train,
                "output_size":self.hparams.base.input_size, 
                "data_folder":target_folder,
                "chunk_length":self.hparams.base.chunk_length_train,
                "min_chunk_length":self.hparams.base.input_size,
                "stride":self.hparams.base.stride_train,
                "transforms":tfms_train,
                "annotation":d.annotation,
                "col_lbl":"label" if self.hparams.loss.loss_type=="supervised" else None,
                "memmap_filename":target_folder/("memmap.npy"),
                "memmap_label_filename":None if d.path_label=="" else Path(d.path_label)/("memmap_label.npy"),
                "fs_annotation_over_fs_data":d.fs_annotation/d.fs}
            kwargs_val = kwargs_train.copy()
            kwargs_val["df"]= df_val
            kwargs_val["chunk_length"]= self.hparams.base.chunk_length_val
            kwargs_val["stride"]= self.hparams.base.stride_val
            kwargs_val["transforms"]= tfms_valtest
            if(self.hparams.loss.loss_type=="supervised"):
                kwargs_valtest_val = kwargs_val.copy()
                kwargs_valtest_val["df"]= df_val
                kwargs_valtest_val["chunk_length"]= self.hparams.base.chunk_length_valtest
                kwargs_valtest_val["stride"]= self.hparams.base.stride_valtest
                kwargs_valtest_val = self.modify_dataset_kwargs(kwargs_valtest_val,stage="test")

                kwargs_valtest_test = kwargs_val.copy()
                kwargs_valtest_test["df"]= df_test
                kwargs_valtest_test["chunk_length"]= self.hparams.base.chunk_length_valtest
                kwargs_valtest_test["stride"]= self.hparams.base.stride_valtest
                kwargs_valtest_test = self.modify_dataset_kwargs(kwargs_valtest_test,stage="test")
            #possibly modify kwargs    
            kwargs_train = self.modify_dataset_kwargs(kwargs_train,stage="train")
            kwargs_val = self.modify_dataset_kwargs(kwargs_val,stage="val")

            train_datasets.append(TimeseriesDatasetCrops(**kwargs_train))
            val_datasets.append(TimeseriesDatasetCrops(**kwargs_val))
            if(self.hparams.loss.loss_type=="supervised"):
                test_datasets_val.append(TimeseriesDatasetCrops(**kwargs_valtest_val))
                test_datasets_test.append(TimeseriesDatasetCrops(**kwargs_valtest_test))
            if(self.hparams.trainer.export_features):
                kwargs_export = kwargs_val.copy()
                kwargs_export["df"] = df_mapped
                kwargs_export["chunk_length"]= self.hparams.base.input_size
                kwargs_export["stride"]= self.hparams.base.stride_export
                kwargs_export = self.modify_dataset_kwargs(kwargs_export,stage="export")
                self.export_datasets.append(TimeseriesDatasetCrops(**kwargs_export))
                print("export dataset:",len(self.export_datasets[-1]),"samples")
            
            print("\n",d.path)
            print("train dataset:",len(train_datasets[-1]),"samples")
            print("val dataset:",len(val_datasets[-1]),"samples")
            if(self.hparams.loss.loss_type=="supervised"):
                print("test dataset(val):",len(test_datasets_val[-1]),"samples")
                print("test dataset(test):",len(test_datasets_test[-1]),"samples")

        if(len(train_datasets)>1): #multiple data folders
            print("\nCombined:")
            self.train_dataset = ConcatDatasetTimeseriesDatasetCrops(train_datasets)
            self.val_dataset = ConcatDatasetTimeseriesDatasetCrops(val_datasets)
            print("train dataset:",len(self.train_dataset),"samples")
            print("val dataset:",len(self.val_dataset),"samples")
            if(self.hparams.loss.loss_type=="supervised"):
                self.test_dataset_val = ConcatDatasetTimeseriesDatasetCrops(test_datasets_val)
                self.test_dataset_test = ConcatDatasetTimeseriesDatasetCrops(test_datasets_test)
                print("test dataset(val):",len(self.test_dataset_val),"samples")
                print("test dataset(test):",len(self.test_dataset_test),"samples")
        else: #just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_dataset = val_datasets[0]
            if(self.hparams.loss.loss_type=="supervised"):
                self.test_dataset_val = test_datasets_val[0]
                self.test_dataset_test = test_datasets_test[0]
        # store idmaps for aggregation
        self.val_idmap = self.val_dataset.get_id_mapping()
        if(self.hparams.loss.loss_type=="supervised"): 
            self.test_idmaps = [self.test_dataset_val.get_id_mapping(), self.test_dataset_test.get_id_mapping()]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.base.batch_size, num_workers=4, shuffle=True, drop_last = True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.base.batch_size, num_workers=self.hparams.trainer.num_workers)
    
    def test_dataloader(self):
        if(self.hparams.loss.loss_type=="supervised"):#multiple val dataloaders
            return [DataLoader(self.test_dataset_val, batch_size=self.hparams.base.batch_size, num_workers=self.hparams.trainer.num_workers),DataLoader(self.test_dataset_test, batch_size=self.hparams.base.batch_size, num_workers=self.hparams.trainer.num_workers)]
        else:
            return DataLoader(self.val_dataset, batch_size=self.hparams.base.batch_size, num_workers=self.hparams.trainer.num_workers)

    def on_fit_start(self):
        if(self.hparams.trainer.pretrained!=""):
            print("Loading pretrained weights from",self.hparams.trainer.pretrained)
            self.load_weights_from_checkpoint(self.hparams.trainer.pretrained)
    
    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()

        pretrained_minus_model = [k for k in pretrained_dict.keys() if not k in model_dict.keys()]
        pretrained_minus_model.sort()
        model_minus_pretrained = [k for k in model_dict.keys() if not k in pretrained_dict.keys()]
        model_minus_pretrained.sort()

        if(len(pretrained_minus_model)>0):
            print("Warning: The following parameter were only present in the state_dict (not in the model):",pretrained_minus_model)
        if(len(model_minus_pretrained)>0):
            print("Warning: The following parameter were only present in the model (not in the state_dict):",model_minus_pretrained)
        
        #update only keys that are actually present in the model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def load_state_dict(self, state_dict):
        #S4-compatible load_state_dict
        for name, param in self.named_parameters():
            param.data = state_dict[name].data.to(param.device)
        for name, param in self.named_buffers():
            param.data = state_dict[name].data.to(param.device)
    
    def export_features(self, output_path, as_memmap=False, aggregate_strides=True):
        
        hook = ForwardHook(self.predictor, store_output=True)

        for di,d in enumerate([self.hparams[d] for d in self.dataset_keys]):

            output_path_ds = Path(output_path)/("features_"+d)
            if not os.path.exists(output_path_ds):
                os.makedirs(output_path_ds)
    
            print("Exporting features for dataset",d,"to",output_path_ds,"...")

            input_size = self.hparams.base.input_size
            stride = self.hparams.base.stride_export

            df_mapped, _, _, _ = self.preprocess_dataset(d)
            ds_export = self.export_datasets[di]
            dl_export = DataLoader(ds_export, batch_size=self.hparams.base.batch_size, num_workers=0)
            
            data_tmp = {}
            idx = 0
            id_map = np.array(ds_export.get_id_mapping())

            metadata = []

            self.eval()
            for i,data_batch in enumerate(iter(dl_export)):
                input_data = data_batch[0].to(self.device)
                self.forward(input_data)
                
                hidden_reps = hook.stored.detach().cpu().numpy()#bs,seq,feat
                ids = id_map[idx:idx+input_data.shape[0]]

                for x in np.unique(ids):
                    #prepare data
                    idtmp = np.where(ids==x)[0]
                    datatmp = hidden_reps[idtmp]#bs,seq,feat
                    
                    #store temporarily as bs,seq,feat (across multiple batches)
                    data_tmp[x]= np.concatenate((data_tmp[x],datatmp),axis=0) if x in data_tmp.keys() else datatmp
                    
                #write to file
                for x in list(data_tmp.keys()):
                    if(x != max(ids) or i==len(dl_export)-1):#sample is complete
                        filename_feat = "feat_"+str(df_mapped.iloc[x]["data_original"]).split("/")[-1]
                        if(aggregate_strides and stride!=input_size):
                            stride_pred = stride//input_size*data_tmp[x].shape[1]#stride in predictor units
                            datatmp=np.zeros((data_tmp[x].shape[1]+(data_tmp[x].shape[0]-1)*stride_pred,data_tmp[x].shape[1]),dtype=np.float32)
                            datatmp_weights = np.zeros(data_tmp[x].shape[1]+data_tmp[x].shape[0]*stride_pred,dtype=np.int64)
                            for j,y in enumerate(data_tmp[x]):
                                start_idx = j*stride_pred
                                datatmp[start_idx:start_idx+data_tmp[x].shape[1]]+=y
                                datatmp_weights[start_idx:start_idx+data_tmp[x].shape[1]]+=1
                            datatmp = datatmp/np.expand_dims(datatmp_weights,axis=-1)
                        else:
                            datatmp= np.concatenate([y for y in data_tmp[x]])
                        tmp_dict = {"id":x,"data_feat":filename_feat, "data_feat_length":len(datatmp)}
                        np.save(output_path_ds/filename_feat,datatmp)
                        del data_tmp[x]

                        metadata.append(tmp_dict)

                idx += input_data.shape[0]
            df_feat = pd.DataFrame(metadata).set_index("id")
            df_mapped["df_idx"]=range(len(df_mapped))
            df_mapped = df_mapped.join(df_feat)
            df_mapped.to_pickle(output_path_ds/("df_mapped.pkl"))
            
            if(as_memmap):
                print("Reformating as memmap...")
                reformat_as_memmap(df_mapped, output_path_ds/"memmap.npy", data_folder=output_path_ds, col_data="data_feat", delete_npys=True)
