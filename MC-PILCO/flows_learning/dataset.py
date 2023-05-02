import torch

# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  '''
    particles_states_sequence = particles per state in the sequence
    particles_states_sequence_mean = mean per state in the sequence
    particles_states_sequence_var = var per state in the sequence
    particles_inputs_sequence_mean = input per state in the sequence
  '''
  def __init__(self, particles_states_sequence, particles_states_sequence_mean, particles_inputs_sequence):
        'Initialization'
        self.particles_states_sequence = particles_states_sequence[:-1]
        self.particles_states_sequence_mean = particles_states_sequence_mean
        self.particles_inputs_sequence = particles_inputs_sequence
        self.particles_obs = self.particles_states_sequence[1:]
        
  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        particles_state = self.particles_states_sequence[index]
        particles_state_mean = self.particles_states_sequence_mean[index]
        particles_state_var = self.particles_states_sequence_var[index]
        particles_inputs = self.particles_inputs_sequence[index]
        particles_obs = self.particles_states_sequence[index]
        
        return particles_state, particles_state_mean, particles_state_var, particles_obs, particles_inputs