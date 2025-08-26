import numpy as np

import torch

from src.utils import sigmoid

from src import settings

import copy


__all__ = ["Gibbs_Update"]





class Gibbs_Update():

    r""" One Gibbs Move

    

      Args:

        temp (float64): Current temperature for gibbs update

        problem (problem class) : Contains methods to calculate energy and change in energy with spin flips

    """

  

    def __init__(self, temp = None, problem = None):

        self.temp = temp

        self.problem = problem

        self.device = torch.device(settings.properties["DEVICE"])
        self.lib = settings.properties["LIB"]
        self.log = np.log
        
        if(self.lib == 'pytorch'):
            self.device = torch.device(settings.properties["DEVICE"])
            self.log = torch.log


  

    def update_sample(self, spin_position, state, debug_mode = False):

        r""" Perform One Gibbs Update according to the spin position"""

        
        # print("Spin Position", spin_position)
        # print("State space", self.problem.state_space)
        # print("Spin State Space",self.problem.state_space[spin_position[0]])
        batch_size = state.shape[0]
        delta_energy = self.problem.calc_delta_energy(self.problem.state_space[spin_position[0]], state, mode = 'single_flip')

        if(self.lib == 'numpy'):
            rand = np.random.uniform(size = (batch_size))
        else:
            rand = torch.rand(size = (batch_size,), device = self.device)
            

        if(debug_mode == True):

            print("\nTemperature is : ", self.temp)

            print("Current state is : ", state)

            print("Checking at position ", spin_position)

            print("Change in energy to flip spin at position ", spin_position, " is : ", delta_energy)

            # print("Probability to flip spin at position ", spin_position,  " is : ", sigmoid(-delta_energy/self.temp))

            print("Random number generated is : ", rand)

            

            if(delta_energy < self.temp*np.log(1/(rand + 1e-12) - 1)):

              print("Accept the spin flip at position ", spin_position)

            else:

              print("Reject the spin flip at position ", spin_position)
        

        # print("delta_energy", delta_energy, "compare term ", (self.temp*self.log(1/(rand + 1e-12) - 1)), delta_energy < (self.temp*self.log(1/(rand + 1e-12) - 1)))
        spin_flip_index = delta_energy < (self.temp*self.log(1/(rand + 1e-12) - 1))     # Checking the condition for spin flip in the block 
        if(self.problem.spin_system):    # Ising spin system is 0/1
 
            state[spin_flip_index, spin_position] = 1 - state[spin_flip_index, spin_position]
          

        else:    # Ising spin system is +1/-1

            state[spin_flip_index, spin_position] = -state[spin_flip_index, spin_position]
        
        if(self.lib == 'numpy'):
            if delta_energy.size != 0:
                delta_energy[~spin_flip_index] = 0
            else:
                delta_energy=0
        else:
            if delta_energy.dim() != 0:
                delta_energy[~spin_flip_index] = 0
            else:
                delta_energy=0


        if(debug_mode == True):

            print("The new state is : ",state)

            print("\n\n")
        # print("Entered Update")
        return delta_energy

    

    

    

    def update_temp(self, temp):

        r""" Function to change the temperature according to schedule """

        self.temp = temp

      

    def set_problem(self, problem):

        self.problem = problem

    
