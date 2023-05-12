import torch

# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  '''
    particles_states = particles per state in the sequence
    particles_states_mean = mean per state in the sequence
    particles_states_vars = var per state in the sequence
    particles_inputs = input per state in the sequence (observation from environment || action)
  '''
  def __init__(self, particles_states, particles_states_means, particles_states_vars, particles_inputs_observations, particles_inputs_actions):
        'Initialization'
        self.particles_states = particles_states
        self.particles_states_means = particles_states_means
        self.particles_states_vars = particles_states_vars
        self.particles_inputs_observations = particles_inputs_observations
        self.particles_inputs_actions = particles_inputs_actions
        
  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        particles_state = self.particles_states[index]
        particles_state_mean = self.particles_states_means[index]
        particles_state_var = self.particles_states_vars[index]
        particles_obs = self.particles_inputs_observations[index]
        particles_action = self.particles_inputs_actions[index]
        
        return particles_state, particles_state_mean, particles_state_var, particles_obs, particles_action
      
  def __len__(self):
    'Denotes the total number of samples'
    assert(len(self.particles_states) == len(self.particles_inputs_observations) == len(self.particles_states_means) == len(self.particles_states_vars) == len(self.particles_inputs_observations) == len(self.particles_inputs_actions))
    return len(self.particles_states)