class NormalizingFlowModel_cond(nn.Module):

    def __init__(self, prior, flows, device='cuda'):
        super().__init__()
        self.prior = prior
        self.device = device
        self.flows = nn.ModuleList(flows).to(self.device)

    def forward(self, x,obser):
        m, _ = x.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows:
            x, ld = flow.forward(x,obser)
            log_det += ld
        z, prior_logprob = x, self.prior.log_prob(x.float())
        return z, prior_logprob, log_det

    def inverse(self, z, obser):
        m, _ = z.shape
        log_det = torch.zeros(m).to(self.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z,obser)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples,obser):
        z = self.prior.sample((n_samples,)).to(self.device)
        x, _ = self.inverse(z,obser)
        return x


class RealNVP_cond(nn.Module):

    def __init__(self, dim, hidden_dim = 8, base_network=FCNN, obser_dim=None):
        super().__init__()
        self.dim = dim
        self.obser_dim=obser_dim
        self.t1 = base_network(dim // 2+self.obser_dim, dim // 2, hidden_dim)
        self.s1 = base_network(dim // 2+self.obser_dim, dim // 2, hidden_dim)
        self.t2 = base_network(dim // 2+self.obser_dim, dim // 2, hidden_dim)
        self.s2 = base_network(dim // 2+self.obser_dim, dim // 2, hidden_dim)
    def zero_initialization(self,var=0.1):
        for layer in self.t1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight,std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s1.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.t2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        for layer in self.s2.network:
            if layer.__class__.__name__=='Linear':
                nn.init.normal_(layer.weight, std=var)
                # layer.weight.data.fill_(0)
                layer.bias.data.fill_(0)
        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, x, obser):
        lower, upper = x[:,:self.dim // 2], x[:,self.dim // 2:]
        t1_transformed = self.t1(torch.cat([lower,obser],dim=-1))
        s1_transformed = self.s1(torch.cat([lower,obser],dim=-1))
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(torch.cat([upper,obser],dim=-1))
        s2_transformed = self.s2(torch.cat([upper,obser],dim=-1))
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(s1_transformed, dim=1) + \
                  torch.sum(s2_transformed, dim=1)
        return z, log_det

    def inverse(self, z, obser):
        lower, upper = z[:,:self.dim // 2], z[:,self.dim // 2:]
        t2_transformed = self.t2(torch.cat([upper,obser],dim=-1))
        s2_transformed = self.s2(torch.cat([upper,obser],dim=-1))
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(torch.cat([lower,obser],dim=-1))
        s1_transformed = self.s1(torch.cat([lower,obser],dim=-1))
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
                  torch.sum(-s2_transformed, dim=1)
        return x, log_det
