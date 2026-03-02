
from ..template_model import SSLModel
from ..template_modules import TaskConfig
from ..data.time_series_dataset_utils import load_dataset
from pathlib import Path
from dataclasses import dataclass

class PSGModel(SSLModel):
    
    def preprocess_dataset(self,dataset_kwargs):
        df_filename = dataset_kwargs.df_filename if hasattr(dataset_kwargs, "df_filename") else dataset_kwargs.get("df_filename", "df_memmap.pkl")
        df_mapped, lbl_itos, mean, std = load_dataset(Path(dataset_kwargs.path), df_filename)
        
        #subsampling SEDF
        if(dataset_kwargs.name.startswith("sedf")):
            if(dataset_kwargs.name=="sedf78"):
                df_mapped = df_mapped[df_mapped[dataset_kwargs.col_lbl].apply(lambda x: x.name).str.startswith('SC')]

            elif(dataset_kwargs.name=="sedf30"):
                df_mapped = df_mapped[df_mapped[dataset_kwargs.col_lbl].apply(lambda x: x.name).str.startswith('ST')]

        return df_mapped, lbl_itos, mean, std
    
@dataclass
class TaskConfigPSG(TaskConfig):
    mainclass:str= "clinical_ts.task.psg.PSGModel"
