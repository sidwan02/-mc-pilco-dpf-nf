import torch
from torch import nn
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from torch.distributions.multivariate_normal import MultivariateNormal
# this file run indirectly with cwd = /MC-PILCO so must have relative import to that dir
from flows_learning.models import *
from flows_learning.loss import nll_loss
from flows_learning.dataset import Dataset
import numpy as np

class Flows_learning(torch.nn.Module):
    # builds conditional nf
    # sets params for training
    def __init__(self, n_sequence, hidden_size, state_dim, init_var=0.01, prior_mean=0.0, prior_std=1.0, device=torch.float64, dtype=torch.device('cpu')):
        super(Flows_learning, self).__init__()
        flows = [RealNVP_cond(dim=state_dim, obser_dim=hidden_size) for _ in range(n_sequence)]

        for f in flows:
            f.zero_initialization(var=init_var)

        prior_init = MultivariateNormal(torch.zeros(state_dim).to(device) + prior_mean,
                                        torch.eye(state_dim).to(device) * prior_std**2)

        self.cond_model = NormalizingFlowModel_cond(prior_init, flows, device=device)
        self.epochs = 100
        self.optimizer = torch.optim.Adam(self.cond_model.parameters(), lr=0.001)
        self.loss_function = nll_loss

    def normalizing_flow_propose(self, pred_particles, pred_particles_mean, pred_particles_var, pred_particles_inputs, n_sequence=2, hidden_dimension=8, obser_dim=None):

        # theres are not trajectories --> how do we handle this
        # this is the samples we take --> we have to sample from y_train for each
        # these particles will be sampled
        B, N, dimension = pred_particles.shape
        # print(B)
        print(pred_particles.shape)
        
        #this is what we change to the mean_variance of the next_state from the GP output
        dyn_particles_mean_flatten, dyn_particles_var_flatten = pred_particles_mean.reshape(-1, dimension), pred_particles_var.reshape(-1, dimension)
        
        print(dyn_particles_mean_flatten.shape)
        print(dyn_particles_var_flatten.shape)
        
        print(pred_particles_inputs.shape)
        
        #context = mean_next, var_next, action
        context = torch.cat([dyn_particles_mean_flatten, dyn_particles_var_flatten, pred_particles_inputs], dim=-1)
        
        #pred_particles = (pred_particles - pred_particles_mean) / pred_particles_var
        pred_particles_flatten=pred_particles.reshape(-1,dimension)

        print("pred_particles shape:", pred_particles.shape)

        print("pred_particles_mean shape:", pred_particles_mean.shape)
        print("pred_particles_var shape:", pred_particles_var.shape)
        print("dyn_particles_mean_flatten shape:", dyn_particles_mean_flatten.shape)
        print("dyn_particles_var_flatten shape:", dyn_particles_var_flatten.shape)
        print("context shape:", context.shape)
        print("pred_particles_flatten shape:", pred_particles_flatten.shape)
        print("obs_reshape shape:", obs_reshape.shape)


        #pred_particles are the samples from our priors, we do not call self.prior
        #inverse
        particles_update_nf, log_det=self.cond_model.inverse(pred_particles_flatten, obs_reshape)

        jac=-log_det
        jac=jac.reshape(pred_particles.shape[:2])

        particles_update_nf=particles_update_nf.reshape(pred_particles.shape)
        #particles_update_nf = particles_update_nf * pred_particles_var + pred_particles_mean

        return particles_update_nf, jac
    
    def reinforce_flows(self, particles_state_sequence, particles_state_mean_sequence, particles_state_var_sequence, particles_inputs_sequence):
        """
        Optimize cnf + pretrain cnf
        optimization_opt_list is a list collecting dictionaries with the optimization options
        """
        #CNF NN has already been initialized when init Flows_learning class
        
        # pretrain flows
        print(particles_state_sequence.shape)
        print(particles_state_mean_sequence.shape)
        print(particles_state_var_sequence.shape)
        print(particles_inputs_sequence.shape)
        
        with torch.no_grad():
            self.train_flows(particles_state_sequence, particles_state_mean_sequence, particles_state_var_sequence, particles_inputs_sequence)
    
    def get_next_state(self, gp_pred_next_state, gp_pred_next_state_mean, gp_pred_next_state_var, cur_input):
        # TODO: dimensions might need to be sorted out (propose takes batch data)
        # print(gp_pred_next_state)
        print(gp_pred_next_state.shape)
        # print(gp_pred_next_state_mean)
        print(gp_pred_next_state_mean.shape)
        print(gp_pred_next_state_var.shape)
        print(cur_input.shape)
        
        particles_update_nf, _ = self.normalizing_flow_propose(np.expand_dims(gp_pred_next_state, axis=0), np.expand_dims(gp_pred_next_state_mean, axis=0), np.expand_dims(gp_pred_next_state_var, axis=0), np.expand_dims(cur_input, axis=0))
        
        return particles_update_nf
    
    def train_flows(self, particles_state_sequence, particles_state_mean_sequence, particles_state_var_sequence, particles_inputs_sequence):
        
        print(particles_state_sequence.shape)
        print(particles_state_mean_sequence.shape)
        print(particles_state_var_sequence.shape)
        print(particles_inputs_sequence.shape)
        # hello
        
        training_set = Dataset(particles_state_sequence, particles_state_mean_sequence, particles_state_var_sequence, particles_inputs_sequence)
        
        params = {'batch_size': 64,
          'shuffle': True,
        #   'num_workers': 1
          }
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        
        loss_history = []
        for epoch in range(self.epochs):
            for batch_idx, (particles_state, particles_state_mean, particles_state_var, particles_inputs) in enumerate(training_generator):
                # Zero the gradients from the previous iteration
                self.optimizer.zero_grad()

                # Call the normalizing_flow_propose function with the current batch data
                particles_update_nf, jac = self.normalizing_flow_propose(particles_state, particles_state_mean, particles_state_var, particles_inputs)

                # Step 3: Calculate the loss and backpropagate the gradients
                # the prior is going to change !!!! it will be the prior from the gaussian!!
                #prior_distribution = torch.distributions.MultivariateNormal(torch.zeros(state_dim), torch.eye(state_dim))

                prior_distribution = MultivariateNormal(torch.zeros(state_dim), torch.eye(state_dim))
                loss = self.loss_function(particles_update_nf, jac, prior_distribution)  # Modify this line to calculate the loss using your loss function
                loss.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    loss_history.append(loss.cpu())

            print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}')
