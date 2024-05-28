import torch

class NoiseSchedule:
    def __init__(self, timesteps, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        self._alphas = self.all_alphas().to(device)
        self._betas = self.all_betas().to(device)
        
        self._cumul_alphas = torch.cumprod(self._alphas, dim=0)
        self._cumul_betas =  (1 - self._cumul_alphas)
        
 
    def all_alphas(self):
        return 1 - self.all_betas()
        
    def all_betas(self):
        raise NotImplementedError

    def alpha(self,t):
        return self._alphas[t]
    
    def beta(self,t):
        return self._betas[t]
    
    def cumul_alpha(self,t):
        return self._cumul_alphas[t]
    
    def cumul_beta(self,t):
        return self._cumul_betas[t]


class LinearSchedule(NoiseSchedule):
    def __init__(self, timesteps, device="cpu", init_t= 1e-4, end_t=0.01):
        self.init_t = init_t
        self.end_t = end_t
        super().__init__(timesteps,device)
        
    def all_betas(self):
        return torch.linspace(self.init_t, self.end_t,self.timesteps)



class CosineSchedule(NoiseSchedule):
    def __init__(self, timesteps, device="cpu", s=0.008):
        self.s = s
        super().__init__(timesteps,device)
        
    def all_betas(self):
        ts = torch.linspace(0,1,self.timesteps)
        ts = (ts + self.s)/ (1 + self.s)
        ts = ts* torch.pi /2
        f = torch.cos(ts)**2
        
        cum_alphas  = f/f[0]
        betas = 1 - cum_alphas[1:]/cum_alphas[:-1]
        betas = torch.cat([torch.tensor([0]),betas])
        betas = torch.clamp(betas,0,0.999)
        
        return betas