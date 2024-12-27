import torch
import torch.nn as nn

class O_Langevin(nn.Module):
    def __init__(self, logp, g, stepsize=1e-1, alpha = 1, beta=1, useHessian=True, M = 1000):
        super(O_Langevin,self).__init__()
        self.logp = logp
        self.g = g
        self.stepsize = stepsize
        self.alpha = alpha
        self.beta = beta
        self.useHessian = useHessian
        self.M = M
        self.dim = 2

    def step(self, x):
        xi = torch.randn_like(x)
        Dlogpx = self.compute_grad(x,self.logp)
        v = self.stepsize*Dlogpx + (2*self.stepsize)**.5*xi
        gx = self.g(x)
        Dgx = self.compute_grad(x,self.g)
        g_perp, g_para = self.project_g(v,  Dgx)

        phi = self.alpha*torch.sign(gx)*torch.abs(gx)**self.beta
        Dgx2 = torch.sum(Dgx**2,dim=1,keepdim=True).repeat(1,self.dim)
        if self.useHessian==True:
            DxD = torch.zeros_like(x)
            # calculating the Hessian term.
            for j in range(x.shape[0]):
                term1 = torch.sum(Dgx[j,:]**2)
                tDgx = Dgx[j,:]/term1
                Hgx = self.compute_hessian(x[[j],:],self.g)
                term3 = 2*torch.sum(tDgx @ Hgx * tDgx) * Dgx[j,:]
                DxD[j,:] = tDgx @ Hgx  +  tDgx * torch.trace(Hgx) - term3
            dx = g_perp - self.stepsize*phi.unsqueeze(1).repeat(1,self.dim)*Dgx/Dgx2 - self.stepsize*DxD
        else:
            dx =  g_perp - self.stepsize*phi.unsqueeze(1).repeat(1,self.dim)*Dgx/Dgx2

        x = x +  torch.clip(dx, -self.M, self.M)
        return x

    def compute_grad(self,x, model):
        x = x.requires_grad_()
        gx = torch.autograd.grad(model(x).sum(), x)[0]
        return gx.detach()

    def compute_hessian(self,x, model):
        x = x.requires_grad_()
        Hgx = torch.autograd.functional.hessian(model, x).squeeze()
        return Hgx.detach()


    def project_g(self,v, dg):
        proj = torch.sum(v*dg,dim=1)/torch.sum(dg**2,dim=1)
        g_para =proj.unsqueeze(1).repeat(1,self.dim)*dg
        g_perp = v - g_para
        return g_perp, g_para