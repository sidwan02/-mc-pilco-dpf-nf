import torch

# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  '''
    particles_state_sequence = particles per state in the sequence
    particles_state_sequence_mean = mean per state in the sequence
    particles_state_var_sequence = var per state in the sequence
    particles_inputs_sequence_mean = input per state in the sequence (observation from environment || action)
  '''
  def __init__(self, particles_state_sequence, particles_state_mean_sequence, particles_state_var_sequence, particles_inputs_sequence):
        'Initialization'
        self.particles_state_sequence = particles_state_sequence
        self.particles_state_mean_sequence = particles_state_mean_sequence
        self.particles_state_var_sequence = particles_state_var_sequence
        self.particles_inputs_sequence = particles_inputs_sequence
        
  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        
        particles_state = self.particles_state_sequence[index]
        particles_state_mean = self.particles_state_mean_sequence[index]
        particles_state_var = self.particles_state_var_sequence[index]
        particles_inputs = self.particles_inputs_sequence[index]
        
        return particles_state, particles_state_mean, particles_state_var, particles_inputs