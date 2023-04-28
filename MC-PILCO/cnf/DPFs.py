import torch
import torch.nn as nn
import numpy as np
from utils import *
from nf.flows import *
from models import NormalizingFlowModel,NormalizingFlowModel_cond
from torch.distributions import MultivariateNormal
import os
from torch.utils.tensorboard import SummaryWriter
from plot import *
from model.models import *
import cv2
from resamplers.resamplers import resampler
from losses import *
from nf.cglow.CGlowModel import CondGlowModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#6,1994715,10,311,1006,54,23,6,24,98


class DPF(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.param = args
        self.NF = args.NF_dyn
        self.NFcond = args.NF_cond
        self.measurement = args.measurement
        self.hidden_size = args.hiddensize  # origin: 32
        self.state_dim = 2  # 4
        self.lr = args.lr
        self.alpha = args.alpha
        self.seq_len = args.sequence_length
        self.num_particle = args.num_particles
        self.batch_size = args.batchsize

        self.labeledRatio = args.labeledRatio

        self.spring_force = 0.1  # 0.1 #0.05  # 0.1 for one object; 0.05 for five objects
        self.drag_force = 0.0075  # 0.0075

        self.pos_noise = args.pos_noise  # 0.1 #0.1
        self.vel_noise = args.vel_noise  # 2.
        self.NF_lr = args.NF_lr
        self.n_sequence = 2

        self.build_model()

        self.eps = args.epsilon
        self.scaling = args.scaling
        self.threshold = args.threshold
        self.max_iter = args.max_iter
        self.resampler = resampler(self.param)

    def build_model(self):
        if self.measurement=='CGLOW':
            self.encoder = build_encoder_cglow(self.hidden_size)
            self.decoder = build_decoder_cglow(self.hidden_size)
            self.build_particle_encoder = build_particle_encoder_cglow
        else:
            self.encoder = build_encoder(self.hidden_size)
            self.decoder = build_decoder(self.hidden_size)
            self.build_particle_encoder = build_particle_encoder

        self.particle_encoder = self.build_particle_encoder(self.hidden_size, self.state_dim)
        self.transition_model = build_transition_model(self.state_dim)
        self.motion_update=motion_update

        # normalising flow dynamic initialisation
        self.nf_dyn = build_conditional_nf(self.n_sequence, 2 * self.state_dim, self.state_dim, init_var=0.01)
        self.cond_model = build_conditional_nf(self.n_sequence, 2 * self.state_dim + self.hidden_size, self.state_dim, init_var=0.01)

        if self.measurement=='CRNVP':
            self.cnf_measurement = build_conditional_nf(self.n_sequence, self.hidden_size, self.hidden_size,
                                                        init_var=0.01, prior_std=2.5)
            self.measurement_model = measurement_model_cnf(self.particle_encoder, self.cnf_measurement)
        

        self.prototype_density=compute_normal_density(pos_noise=self.pos_noise, vel_noise= self.vel_noise)

        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[30*(1+x) for x in range(10)], gamma=1.0)
