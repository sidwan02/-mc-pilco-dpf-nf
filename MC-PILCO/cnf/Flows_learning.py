import torch
from torch import nn
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from torch.distributions.multivariate_normal import MultivariateNormal
from models import *
from loss import nll_loss
from dataset import Dataset

class Flows_learning(torch.nn.Module):
    # builds conditional nf
    # sets params for training
    def __init__(self, n_sequence, hidden_size, state_dim, init_var=0.01, prior_mean=0.0, prior_std=1.0):
        super(Flows_learning, self).__init__()
        flows = [RealNVP_cond(dim=state_dim, obser_dim=hidden_size) for _ in range(n_sequence)]

        for f in flows:
            f.zero_initialization(var=init_var)

        prior_init = MultivariateNormal(torch.zeros(state_dim).to(device) + prior_mean,
                                        torch.eye(state_dim).to(device) * prior_std**2)

        self.cond_model = NormalizingFlowModel_cond(prior_init, flows, device=device)
        self.epochs = 100
        self.optimizer = torch.optim.Adam(cnf.parameters(), lr=0.001)
        self.loss_function = nll_loss

    def normalizing_flow_propose(self, particles_pred, obs, n_sequence=2, hidden_dimension=8, obser_dim=None):

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
        particles_update_nf, log_det=self.cond_model.inverse(particles_pred_flatten, obs_reshape)

        jac=-log_det
        jac=jac.reshape(particles_pred.shape[:2])

        particles_update_nf=particles_update_nf.reshape(particles_pred.shape)
        #particles_update_nf = particles_update_nf * pred_particles_std + pred_particles_mean

        return particles_update_nf, jac
    
    def pretrain_flows(self):
        return None

    def train_flows(self, particles_states_sequence, particles_states_sequence_mean, particles_inputs_sequence):
        training_set = Dataset(particles_states_sequence, particles_states_sequence_mean, particles_inputs_sequence)
        params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        
        for epoch in range(self.epochs):
            for batch_idx, (particles_state, particles_state_mean, particles_state_var, particles_obs, particles_inputs) in enumerate(training_generator):
                # Zero the gradients from the previous iteration
                self.optimizer.zero_grad()

                # Call the normalising_flow_propose function with the current batch data
                particles_update_nf, jac = self.normalising_flow_propose(cnf, particles_pred, observations)

                # Step 3: Calculate the loss and backpropagate the gradients
                # the prior is going to change !!!! it will be the prior from the gaussian!!
                #prior_distribution = torch.distributions.MultivariateNormal(torch.zeros(state_dim), torch.eye(state_dim))

                
                prior_distribution = MultivariateNormal(torch.zeros(state_dim), torch.eye(state_dim))
                loss = self.loss_function(particles_update_nf, jac, prior_distribution)  # Modify this line to calculate the loss using your loss function
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}')
