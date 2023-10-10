__all__ = ['RNNEncoder', 'CPCModel']

import torch
import torch.nn.functional as F
import torch.nn as nn
from clinical_ts.basic_conv1d import _conv1d
import numpy as np

from collections.abc import Iterable
from .basic_conv1d import bn_drop_lin
from .transformer import *
from .quantizer import GumbelQuantize

class RNNEncoder(nn.Module):
    'CPC Encoder'
    def __init__(self, input_channels, strides=[5,4,2,2,2], kss=[10,8,4,4,4], features=[512,512,512,512],bn=False,layer_norm=False):
        super().__init__()
        assert(len(strides)==len(kss) and len(strides)==len(features))
        lst = []
        for i,(s,k,f) in enumerate(zip(strides,kss,features)):
            lst.append(_conv1d(input_channels if i==0 else features[i-1],f,kernel_size=k,stride=s,bn=bn,layer_norm=layer_norm))
        self.layers = nn.Sequential(*lst)
        self.downsampling_factor = np.prod(strides)
        self.output_dim = features[-1]
        # output: bs, output_dim, seq//downsampling_factor
    def forward(self, input):
        input_encoded = self.layers(input).transpose(1,2)
        return input_encoded

class RNNHead(nn.Module):
    def __init__(self,n_hidden,num_classes,concat_pooling=True, lin_ftrs_head=[], dropout_head=0.5, bn_head=True, bidirectional=True, batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        self.concat_pooling = concat_pooling

        layers_head =[]
        if(concat_pooling):
            layers_head.append(AdaptiveConcatPoolRNN(bidirectional=bidirectional,cls_first= False))

        #classifier
        output_dim = n_hidden*(2 if bidirectional else 1)
                
        nf = 3*output_dim if concat_pooling else output_dim
                
        lin_ftrs_head = [nf, num_classes] if lin_ftrs_head is None else [nf] + lin_ftrs_head + [num_classes]
        ps_head = [dropout_head] if not isinstance(dropout_head, Iterable) else dropout_head
        if len(ps_head)==1:
            ps_head = [ps_head[0]/2] * (len(lin_ftrs_head)-2) + ps_head
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs_head)-2) + [None]

        for ni,no,p,actn in zip(lin_ftrs_head[:-1],lin_ftrs_head[1:],ps_head,actns):
            layers_head+=bn_drop_lin(ni,no,bn_head,p,actn,layer_norm=False)
        self.head=nn.Sequential(*layers_head)

    def forward(self,x):   
        if(self.batch_first):#B,S,E
            x = x.transpose(1,2) 
        else:#S,B,E
            x = x.transpose(0,1).transpose(1,2)
        if(self.concat_pooling is False):
            x = x[-1,:,:]
        return self.head(x)

class CPCModel(nn.Module):
    "CPC model"
    def __init__(self, hparams):
        super().__init__()
        assert(hparams["skip_encoder"] is False or hparams["num_classes"] is not None)#pretraining only with encoder
        if(hparams["transformer"]):
            self.encoder = TransformerPatchTokenizer(hparams["dim_model_transformer"], input_channels=hparams["input_channels"],patch_size=hparams["timesteps_per_token"], dim_data=1)
        else:
            self.encoder = RNNEncoder(hparams["input_channels"]*hparams["timesteps_per_token"],strides=hparams["strides_encoder"],kss=hparams["kss_encoder"],features=hparams["features_encoder"],bn=hparams["bn_encoder"],layer_norm=hparams["transformer"]) if hparams["skip_encoder"] is False else None
            self.encoder_output_dim = self.encoder.output_dim if hparams["skip_encoder"] is False else None
            self.encoder_downsampling_factor = self.encoder.downsampling_factor if hparams["skip_encoder"] is False else None

        self.timesteps_per_token = hparams["timesteps_per_token"] 
        self.num_classes = hparams["num_classes"]
        
        self.transformer = hparams["transformer"]

        if(hparams["transformer"]):
            #self.transformer_model = TransformerModule(dim_model=hparams["dim_model_transformer"], mlp_ratio=hparams["mlp_ratio_transformer"], dropout=hparams["dropout_transformer"], num_layers=hparams["n_layers_transformer"], num_heads=hparams["n_heads_transformer"], masked=not(hparams["bidirectional"]), max_length=hparams["input_size"], batch_first=True, input_size=input_dim_predictor, pos_enc=hparams["pos_enc_transformer"], activation=hparams["activation_transformer"],cls_token=hparams["cls_token_transformer"])
            transformer_input_length = self.encoder.get_output_length(hparams["input_size"])
            self.transformer_model = TransformerModule(dim_model=hparams["dim_model_transformer"], mlp_ratio=hparams["mlp_ratio_transformer"], dropout=hparams["dropout_transformer"], num_layers=hparams["n_layers_transformer"], num_heads=hparams["n_heads_transformer"], masked=not(hparams["bidirectional"]), max_length=transformer_input_length, pos_enc=hparams["pos_enc_transformer"], activation=hparams["activation_transformer"],cls_token=hparams["cls_token_transformer"])
        else:
            input_dim_predictor = self.encoder_output_dim if hparams["skip_encoder"] is False else hparams["input_channels"]*hparams["timesteps_per_token"]
        
            rnn_arch = nn.LSTM if hparams["lstm"] else nn.GRU
            self.rnn = rnn_arch(input_dim_predictor,hparams["n_hidden_rnn"],num_layers=hparams["n_layers_rnn"],batch_first=True,bidirectional=hparams["bidirectional"])
        
        if(hparams["num_classes"] is None): #pretraining
            self.pretraining_mode = hparams["pretraining_mode"]
            self.pretraining_targets = hparams["pretraining_targets"]

            assert(hparams["bidirectional"] is False)#only unidirectional during CPC training
            n_hidden = hparams["dim_model_transformer"] if hparams["transformer"] else hparams["n_hidden_rnn"]

            if(self.pretraining_targets==0):
                n_output = self.encoder_output_dim
            elif(self.pretraining_targets==1):
                n_output = hparams["quantizer_embedding_dim"]
            else:
                n_output = hparams["quantizer_vocab"]

            if(hparams["mlp"]):# additional hidden layer as in simclr
                self.proj = nn.Sequential(nn.Linear(n_hidden, n_hidden),nn.ReLU(inplace=True),nn.Linear(n_hidden, n_output,bias=hparams["bias_proj"]))
            else:
                self.proj = nn.Linear(n_hidden, n_output,bias=hparams["bias_proj"])
            #hparams["pretraining_mode"]=="lm-cont-quant" or hparams["pretraining_mode"]=="lm-discrete"):
            self.quantizer = GumbelQuantize(self.encoder_output_dim, hparams["quantizer_vocab"], hparams["quantizer_embedding_dim"]) if(hparams["pretraining_targets"]>0) else None
            
        else: #classifier
            if(hparams["transformer"]):
                self.head = TransformerHead(hparams["dim_model_transformer"],hparams["num_classes"],pooling_type=hparams["pooling_type_transformer"],batch_first=False,n_heads_seq_pool=hparams["n_heads_seq_pool_transformer"])
            else:
                self.head = RNNHead(n_hidden=hparams["n_hidden_rnn"],num_classes=hparams["num_classes"],concat_pooling=hparams["concat_pooling"], lin_ftrs_head=hparams["lin_ftrs_head"], dropout_head=hparams["dropout_head"], bn_head=hparams["bn_head"], bidirectional=hparams["bidirectional"], batch_first=True)

    def forward(self, input):
        # input shape bs,ch,seq
        if(self.timesteps_per_token > 1 and self.transformer is False):#patches a la vision transformer
            assert(input.size(2)%self.timesteps_per_token==0)
            size = input.size()
            input = input.transpose(1,2).reshape(size[0],size[2]//self.timesteps_per_token,-1).transpose(1,2)

        input_encoded = self.encoder(input) if self.encoder is not None else input.transpose(1,2) #bs, seq, channels
        
        if(self.transformer):
            output_rnn = self.transformer_model(input_encoded)
        else:
            output_rnn, _ = self.rnn(input_encoded) #output_rnn: bs, seq, n_hidden
        
        if(self.num_classes is None):#pretraining
            if(self.pretraining_targets>0):#quantized
                input_quantized, loss_quantizer, soft_one_hot = self.quantizer(input_encoded.transpose(1,2))
                if(self.pretraining_targets==1):
                    target = input_quantized.transpose(1,2)
                else:
                    target = soft_one_hot.transpose(1,2)
            else:
                target = input_encoded
                loss_quantizer = torch.tensor(0,dtype=torch.float32).to(input.device)
            return target, self.proj(output_rnn), loss_quantizer    
        else:#classifier
            return self.head(output_rnn)
        
    def get_layer_groups(self):
        if(self.transformer):
            return (self.encoder,self.transformer_model,self.head)
        else:
            return (self.encoder,self.rnn,self.head)

    def get_output_layer(self):
        return self.head[-1]

    def set_output_layer(self,x):
        self.head[-1] = x

    def lm_loss(self,input, target=None, steps_predicted=1):
        assert(self.num_classes is None)
        input_processed, output, loss_quantizer = self.forward(input) #output: bs,seq,features; input_processed: bs,seq,features
        
        input_processed = input_processed[:,steps_predicted:] #bs,seq,features
        output = output[:,:-steps_predicted]
        if(self.pretraining_targets==2):
            #loss is kl divergence between the two
            return  F.kl_div(F.softmax(output,dim=2), input_processed, reduction="batchmean", log_target=False)+loss_quantizer
        else:
            return torch.mean(torch.sum(input_processed*output,dim=2)/(torch.norm(input_processed,p=2,dim=2)+1e-10)/(torch.norm(output,p=2,dim=2)+1e-10))+loss_quantizer
        
    def cpc_loss(self,input, target=None, steps_predicted=5, n_false_negatives=9, negatives_from_same_seq_only=False, eval_acc=False):
        assert(self.num_classes is None)
        assert(self.pretraining_targets!=2)

        input_encoded, output, loss_quantizer = self.forward(input) #input_encoded: bs, seq, features; output: bs,seq,features
        input_encoded_flat = input_encoded.reshape(-1,input_encoded.size(2)) #for negatives below: -1, features
        
        bs = input_encoded.size()[0]
        seq = input_encoded.size()[1]
        
        loss = torch.tensor(0,dtype=torch.float32).to(input.device)
        tp_cnt = torch.tensor(0,dtype=torch.int64).to(input.device)
        
        for i in range(input_encoded.size()[1]-steps_predicted):
            positives = input_encoded[:,i+steps_predicted].unsqueeze(1) #bs,1,encoder_output_dim
            if(negatives_from_same_seq_only):
                idxs = torch.randint(0,(seq-1),(bs*n_false_negatives,)).to(input.device)
            else:#negative from everywhere
                idxs = torch.randint(0,bs*(seq-1),(bs*n_false_negatives,)).to(input.device)
            idxs_seq = torch.remainder(idxs,seq-1) #bs*false_neg
            idxs_seq2 = idxs_seq * (idxs_seq<(i+steps_predicted)).long() +(idxs_seq+1)*(idxs_seq>=(i+steps_predicted)).long()#bs*false_neg
            if(negatives_from_same_seq_only):
                idxs_batch = torch.arange(0,bs).repeat_interleave(n_false_negatives).to(input.device)
            else:
                idxs_batch = idxs//(seq-1)
            idxs2_flat = idxs_batch*seq+idxs_seq2 #for negatives from everywhere: this skips step i+steps_predicted from the other sequences as well for simplicity
            
            negatives = input_encoded_flat[idxs2_flat].view(bs,n_false_negatives,-1) #bs*false_neg, encoder_output_dim
            candidates = torch.cat([positives,negatives],dim=1)#bs,false_neg+1,encoder_output_dim
            preds=torch.sum(output[:,i].unsqueeze(1)*candidates,dim=-1) #bs,(false_neg+1)
            targs = torch.zeros(bs, dtype=torch.int64).to(input.device)
            
            if(eval_acc):
                preds_argmax = torch.argmax(preds,dim=-1)
                tp_cnt += torch.sum(preds_argmax == targs)
               
            loss += F.cross_entropy(preds,targs) + loss_quantizer
        if(eval_acc):
            return loss, tp_cnt.float()/bs/(input_encoded.size()[1]-steps_predicted)
        else:
            return loss

#copied from RNN1d
class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self, bidirectional=False, cls_first=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.cls_first = cls_first

    def forward(self,x):
        #input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)

        if(self.bidirectional is False):
            if(self.cls_first):
                t3 = x[:,:,0]
            else:
                t3 = x[:,:,-1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:,:channels,-1],x[:,channels:,0]],1)
        out=torch.cat([t1.squeeze(-1),t2.squeeze(-1),t3],1) #output shape bs, 3*ch
        return out
