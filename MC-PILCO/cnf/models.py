import torch
from torch import nn
from nf.models import NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
from utils import et_distance
from nf.flows import *
from nf.cglow.CGlowModel import CondGlowModel
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import torch
import torch.nn as nn


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

def build_conditional_nf(n_sequence, hidden_size, state_dim, init_var=0.01, prior_mean=0.0, prior_std=1.0):
    flows = [RealNVP_cond(dim=state_dim, obser_dim=hidden_size) for _ in range(n_sequence)]

    for f in flows:
        f.zero_initialization(var=init_var)

    prior_init = MultivariateNormal(torch.zeros(state_dim).to(device) + prior_mean,
                                    torch.eye(state_dim).to(device) * prior_std**2)

    cond_model = NormalizingFlowModel_cond(prior_init, flows, device=device)

    return cond_model

def normalising_flow_propose(cond_model, particles_pred, obs, flow=RealNVP_cond, n_sequence=2, hidden_dimension=8, obser_dim=None):

    # theres are not trajectories --> how do we handle this
    # this is the samples we take --> we have to sample from y_train for each
    # these particles will be sampled
    B, N, dimension = particles_pred.shape

    #output of the gaussian process mean,var 
    pred_particles_mean, pred_particles_std = particles_pred.mean(dim=1, keepdim=True).detach().clone().repeat([1, N, 1]), \
                                            particles_pred.std(dim=1, keepdim=True).detach().clone().repeat([1, N, 1])
    
    
    #this is what we change to the mean_variance of the next_state from the GP output
    dyn_particles_mean_flatten, dyn_particles_std_flatten = pred_particles_mean.reshape(-1, dimension), pred_particles_std.reshape(-1, dimension)
    
    #context = mean_next, var_next, action
    action_list = torch.randint(low=0, high=10, size=(640, 1)) #replace with actions in actual environment
    context = torch.cat([dyn_particles_mean_flatten, dyn_particles_std_flatten,action_list], dim=-1)
    
    
    #particles_pred = (particles_pred - pred_particles_mean) / pred_particles_std
    particles_pred_flatten=particles_pred.reshape(-1,dimension)

    #observation will be the current state
    #we can just make the observations be the current state
    # predicted, current, state, action, pairs
    obs_reshape_og = obs.reshape(-1, dimension)
    print("obs_reshape before concat shape:", obs_reshape_og.shape)

    #obs_reshape_og = obs[:, None, :].repeat([1,N,1]).reshape(B*N,-1)

    #mean,variance of the next_states concatonated with current state,action
    #we also want to include the action as well
    obs_reshape = torch.cat([obs_reshape_og, context], dim=-1)

    print("particles_pred shape:", particles_pred.shape)
    print("observation shape:", obs.shape)
    print("action shape:", action_list.shape)

    print("pred_particles_mean shape:", pred_particles_mean.shape)
    print("pred_particles_std shape:", pred_particles_std.shape)
    print("dyn_particles_mean_flatten shape:", dyn_particles_mean_flatten.shape)
    print("dyn_particles_std_flatten shape:", dyn_particles_std_flatten.shape)
    print("context shape:", context.shape)
    print("particles_pred_flatten shape:", particles_pred_flatten.shape)
    print("obs_reshape shape:", obs_reshape.shape)


    #particles_pred are the samples from our priors, we do not call self.prior
    #inverse
    particles_update_nf, log_det=cond_model.inverse(particles_pred_flatten, obs_reshape)

    jac=-log_det
    jac=jac.reshape(particles_pred.shape[:2])

    particles_update_nf=particles_update_nf.reshape(particles_pred.shape)
    #particles_update_nf = particles_update_nf * pred_particles_std + pred_particles_mean

    return particles_update_nf, jac
