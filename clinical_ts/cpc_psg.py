
from .cpc_main import *

class CPCPSG(CPCMain):
    
    def preprocess_dataset(self,dataset_kwargs):
        df_mapped, lbl_itos, mean, std = load_dataset(Path(dataset_kwargs.path))
        
        #subsampling SEDF
        if(dataset_kwargs.name.startswith("sedf")):
            if(dataset_kwargs.name=="sedf78"):
                df_mapped = df_mapped[df_mapped['label'].apply(lambda x: x.name).str.startswith('SC')]

            elif(dataset_kwargs.name=="sedf30"):
                df_mapped = df_mapped[df_mapped['label'].apply(lambda x: x.name).str.startswith('ST')]

        return df_mapped, lbl_itos, mean, std
