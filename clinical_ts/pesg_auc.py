__all__ = ['PESG_AUC','auc_loss']

import torch
from torch.optim import Optimizer
import torch.nn as nn
import numpy as np

def boring_ranges(A1, A2, A3, A4):
    print("A1 range: {}-{}".format(torch.min(A1), torch.max(A1)))
    print("A2 range: {}-{}".format(torch.min(A2), torch.max(A2)))
    print("A3 range: {}-{}".format(torch.min(A3), torch.max(A3)))
    print("A4 range: {}-{}".format(torch.min(A4), torch.max(A4)))

class PESG_AUC(Optimizer):
    """ optimizer from https://arxiv.org/abs/2012.03173
    params: NN params PLUS a and b as parameters of the loss
    params_alpha: alpha parameters in the loss
    lr aka eta

    """

    def __init__(self, params, lr=1e-3, ema=0.999, gamma=0*2e-3, weight_decay=1e-3):
        defaults = dict(lr=lr, ema=ema, gamma=gamma,  weight_decay=weight_decay, is_alpha=False)
        super(PESG_AUC, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(PESG_AUC, self).__setstate__(state)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        lr = None
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            gamma = group['gamma']
            ema = group['ema']
            lr = group['lr']
            is_alpha = group['is_alpha']

            for p in group['params']:
                if(not(is_alpha)):#all other params update
                    if p.grad is not None:
                        if(gamma !=0):
                            param_state = self.state[p]
                            if 'avg_param_buffer' not in param_state:
                                ref = param_state['avg_param_buffer'] = torch.clone(p).detach()
                            else:
                                ref = param_state['avg_param_buffer']
                                ref.mul_(1-ema).add_(p, alpha=ema)
                        d_p = p.grad
                        if weight_decay != 0:
                            d_p = d_p.add(p, alpha=weight_decay)
                        if gamma != 0:
                            d_p = d_p.add(gamma* (p-ref))
                
                    p.add_(d_p, alpha=-lr)
                else:#alpha update
                    if p.grad is not None:
                        p.add_(p.grad*lr)
                        p = p.clip(min=0)
        return loss

        
class auc_loss(nn.Module):
    '''loss from https://arxiv.org/abs/2012.03173
    note: different paramgroup for alpha, see e.g. https://stackoverflow.com/questions/52069377/how-can-i-only-update-some-specific-tensors-in-network-with-pytorch
    with is_alpha=True'''
    def __init__(self,ps=[0.5,0.5],m=0.5, sigmoid=False, reduction='mean'):
        super(auc_loss, self).__init__()
        
        self.a = nn.Parameter(0.1*torch.ones(len(ps)))# if len(ps)>2 else 1))
        self.b = nn.Parameter(0.1*torch.ones(len(ps)))# if len(ps)>2 else 1))
        self.alpha = nn.Parameter(torch.ones(len(ps)))# if len(ps)>2 else 1)) #ideal 1+b-a

        self.register_buffer("ps", torch.from_numpy(np.array(ps)))
        self.m = m
        self.sigmoid = sigmoid
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        #loss = torch.zeros(input.size(0)).to(input.device)
        if(len(self.ps)==2 and len(target.size())==1):#one-hot encode
            target = torch.stack([1-target,target],dim=1)
        if(self.sigmoid):
            input = torch.sigmoid(input)
        #for i in range(len(self.ps)):# if len(self.ps)>2 else [1]:
        #    p = self.ps[i]
        
        
        A1 = (1-self.ps)*(input-self.a).pow(2)*(target==1)
        A2 = self.ps*(input-self.b).pow(2)*(target==0)
        A3 = (1-self.ps)*self.ps*self.alpha*self.alpha
        A4 = 2*self.alpha*(self.ps*(1-self.ps)*self.m +self.ps*input*(target==0) - (1-self.ps)*input*(target==1))
        # boring_ranges(A1, A2, A3, A4)
        # pdb.set_trace()
        loss = torch.mean(A1 + A2 - A3 + A4,dim=1)#mean over classes
        #reduction over samples
        if(self.reduction=="mean"):
            return torch.mean(loss)
        elif(self.reduction=="sum"):
            return torch.sum(loss)
        elif(self.reduction=="none"):
            return loss
        assert(True)#invalid reduction
