import numpy as np

import torch

import time

from src import settings

from tqdm import tqdm

import random

__all__ = ["Gibbs_Sampler", "Gibbs_Sampler_Parallel_Tempering"]



class Gibbs_Sampler():

    r"""Sampler Class which does Gibbs Sampling on the Ising Lattice

    

      Args:

          param (dict): 

          num_samples (int) : Number of samples

          type_order(str) : The order in which spins are to be updated

          scheduler (Scheduler class) : Gives the temperature according to a cooling schedule

          update (Updates Class) : Gives the next spin state  

          problem (Problem Class) : The Ising Problem
           
          state is the samples arranged according to the state_space tupple 
          It needs to be converted into dictionary while calculating the energy

          

      """  

      

    def __init__(self, time_steps, N, batch_size, scheduler, update, problem):

        self.time_steps = time_steps

        self.N = N

        self.batch_size = batch_size

        self.scheduler = scheduler

        self.update = update

        self.problem = problem

        self.stats = {"num_accepts" : 0}

        self.observables = {"ising_energy" : np.array([])}

        self.state = np.array([])

        self.state_space = []

        self.times = np.array([])

        self.lib = settings.properties["LIB"]
  

        # self.device = torch.device(settings.properties["DEVICE"])

        # self.observables = {"ising_energy" : np.empty((self.N*self.batch_size, self.time_steps))}
        if(self.lib == 'pytorch'):
            self.device = torch.device(settings.properties["DEVICE"])
            self.observables = {"ising_energy" : torch.empty((self.N*self.batch_size, self.time_steps), device = self.device)}
            self.allclose = torch.allclose
        else:
            self.observables = {"ising_energy" : np.empty((self.N*self.batch_size, self.time_steps))}
            self.allclose = np.allclose
        

        

    def init_spin(self):

          r""" Initializes the spin system

          """
          self.state_space =  self.problem.return_state_space()
        #   self.state = np.random.randint(0, 2, size = (self.batch_size, len(self.state_space)))
          if(self.lib == 'numpy'):
                self.state = np.random.randint(0, 2, size = (self.batch_size, len(self.state_space)))
          elif (self.problem.ising ==1):
                self.state = torch.randint(0, 2, size = (self.batch_size, len(self.state_space)), device = self.device, dtype = torch.float32)
          else:
                self.state = torch.randint(0, 2, size = (self.batch_size, len(self.state_space)), device = self.device, dtype = torch.uint8)
                
    def set_problem(self,problem):

          self.problem = problem

          self.update.set_problem(problem)

  

    def return_order(self):

          r""" Decides the order of spins for update for Gibbs Sampling

          """

          return np.random.permutation(len(self.problem.return_state_space()))

    

    def Run(self, return_states = False, progress_bar = True, convergence = True, debug_mode = False):

        r""" Starts the sampling 

        """

        
        self.state_space =  self.problem.return_state_space()
        if(return_states == True):
            # state_array = np.empty((self.N*self.batch_size, self.time_steps, len(self.state_space)))
            if(self.lib == 'numpy'):
                state_array = np.empty((self.N*self.batch_size, self.time_steps, len(self.state_space))) 
            elif (self.problem.ising ==1):
                state_array = torch.empty((self.N*self.batch_size, self.time_steps, len(self.state_space)), device = self.device, dtype = torch.float32)  
            else:
                state_array = torch.empty((self.N*self.batch_size, self.time_steps, len(self.state_space)), device = self.device, dtype = torch.uint8)       

        for nn in tqdm(range(self.N), disable = not progress_bar):
            self.init_spin()

            energy = self.problem.calc_ising_energy(self.state, mode = 'single_flip')

            num_reset = 0

            

            for ii in tqdm(range(self.time_steps), leave = False, disable = not progress_bar):

                ''' Extracting temperature from scheduler. Update the temperature for the updater. Determining the order of spin updates '''

                current_temp = self.scheduler.scheduled_temp(ii)

                self.update.update_temp(current_temp)

                order = self.return_order()

                # print("Spin Update Order", order)

                for jj in order:
                    energy += self.update.update_sample(np.array([jj]), self.state, debug_mode)
                

                if(return_states == True):    # Logging states
                    state_array[nn*self.batch_size:(nn + 1)*self.batch_size, ii, :] = self.state.reshape((self.batch_size, -1))
                  

                self.observables["ising_energy"][nn*self.batch_size:(nn + 1)*self.batch_size, ii] = energy

        if(self.lib == 'pytorch'):
            if(self.observables["ising_energy"].is_cuda):
                self.observables["ising_energy"] = self.observables["ising_energy"].cpu().detach().numpy()
                
                if(return_states == True):
                    state_array = state_array.cpu().detach().numpy()
                return state_array, self.observables["ising_energy"]
            else:
                self.observables["ising_energy"] = self.observables["ising_energy"].numpy()
                    
                if(return_states == True):
                    state_array = state_array.numpy()
                    return state_array, self.observables["ising_energy"]
                
        elif(return_states == True):
            return state_array, self.observables["ising_energy"]







class Gibbs_Sampler_Parallel_Tempering():

    r"""Sampler Class which does Gibbs Sampling on the Ising Lattice

    

      Args:

          param (dict): 

          num_samples (int) : Number of samples

          type_order(str) : The order in which spins are to be updated

          scheduler (Scheduler class) : Gives the temperature according to a cooling schedule

          update (Updates Class) : Gives the next spin state  

          problem (Problem Class) : The Ising Problem
           
          state is the samples arranged according to the state_space tupple 
          It needs to be converted into dictionary while calculating the energy

          

      """  

      

    def __init__(self, time_steps, pt_interval, N, batch_size, scheduler, update, problem):

        self.time_steps = time_steps

        self.pt_interval = pt_interval
        
        self.N = N

        self.batch_size = batch_size

        self.scheduler = scheduler

        self.update = update

        self.problem = problem

        self.stats = {"num_accepts" : 0}

        self.observables = {"ising_energy" : np.array([])}

        self.state = np.array([])

        self.state_space = []

        self.times = np.array([])

        self.lib = settings.properties["LIB"]
  

        # self.device = torch.device(settings.properties["DEVICE"])

        # self.observables = {"ising_energy" : np.empty((self.N*self.batch_size, self.time_steps))}
        if(self.lib == 'pytorch'):
            self.device = torch.device(settings.properties["DEVICE"])
            self.observables = {"ising_energy" : torch.empty((self.N*self.batch_size, self.time_steps), device = self.device)}
            self.allclose = torch.allclose
        else:
            self.observables = {"ising_energy" : np.empty((self.N*self.batch_size, self.time_steps))}
            self.allclose = np.allclose
        

        

    def init_spin(self):

          r""" Initializes the spin system

          """
          self.state_space =  self.problem.return_state_space()
        #   self.state = np.random.randint(0, 2, size = (self.batch_size, len(self.state_space)))
          if(self.lib == 'numpy'):
                self.state = np.random.randint(0, 2, size = (self.batch_size, len(self.state_space)))
          elif (self.problem.ising ==1):
                self.state = torch.randint(0, 2, size = (self.batch_size, len(self.state_space)), device = self.device, dtype = torch.float32)
          else:
                self.state = torch.randint(0, 2, size = (self.batch_size, len(self.state_space)), device = self.device, dtype = torch.uint8)
                
    def set_problem(self,problem):

          self.problem = problem

          self.update.set_problem(problem)

  

    def return_order(self):

          r""" Decides the order of spins for update for Gibbs Sampling

          """

          return np.random.permutation(len(self.problem.return_state_space()))

    

    def Run(self, return_states = False, progress_bar = True, convergence = True, debug_mode = False):

        r""" Starts the sampling 

        """

        
        self.state_space =  self.problem.return_state_space()
        # print("State Space",self.state_space )

        if(return_states == True):
            # state_array = np.empty((self.N*self.batch_size, self.time_steps, len(self.state_space)))
            if(self.lib == 'numpy'):
                state_array = np.empty((self.N*self.batch_size, self.time_steps, len(self.state_space))) 
            elif (self.problem.ising ==1):
                state_array = torch.empty((self.N*self.batch_size, self.time_steps, len(self.state_space)), device = self.device, dtype = torch.float32)  
            else:
                state_array = torch.empty((self.N*self.batch_size, self.time_steps, len(self.state_space)), device = self.device, dtype = torch.uint8)       

        for nn in tqdm(range(self.N), disable = not progress_bar):
            self.init_spin()

            energy = self.problem.calc_ising_energy(self.state, mode = 'single_flip')

            num_reset = 0

            

            for ii in tqdm(range(self.time_steps), leave = False, disable = not progress_bar):

                ''' Extracting temperature from scheduler. Update the temperature for the updater. Determining the order of spin updates '''

                if(self.lib == 'numpy'):
                    current_temp = self.scheduler.scheduled_temp()
                else:
                    current_temp = self.scheduler.scheduled_temp()
                    current_temp = torch.from_numpy(current_temp)
                    # Move the tensor to the desired device
                    current_temp = current_temp.to(self.device)

                self.update.update_temp(current_temp)  ##  parallel temperature schedules
                order = self.return_order()

                # print("Spin Update Order", order)

                for jj in order:

                    energy += self.update.update_sample(np.array([jj]), self.state, debug_mode)

                if ((ii+1)%self.pt_interval == 0):  
                    # print("Entered into tempering loop")
                    if ((ii+1)//self.pt_interval)%2 ==1 : ## even-leading-index swap {(0, 1), (2, 3)} 0 lower temp 1 higher
                        ### Pt calculations 
                        if (self.lib == "numpy"):
                            r = np.exp(energy[::2]- energy[1::2])/((1/self.update.temp[::2])-(1/self.update.temp[1::2]))
                            pswap = np.minimum(1, r)
                            valid_swaps = pswap > np.random.randint(0, 2, size = pswap.shape[0])
                            # Get the indices where mask is True
                            indices_to_swap = np.where(valid_swaps)[0]*2 ## valid swaps are on pair indices
                            # Create a copy of the array to avoid modifying the input array in-place
                            swapped_energy = energy.copy()
                            swapped_states = self.state.copy()
                            energy[indices_to_swap], energy[indices_to_swap + 1] = (
                                    swapped_energy[indices_to_swap + 1], swapped_energy[indices_to_swap]
                            )

                            self.state[indices_to_swap,:], self.state[indices_to_swap + 1,:] = (
                                    swapped_states[indices_to_swap + 1,:], swapped_states[indices_to_swap,:]
                            )
                        
                        else: ### For torch array
                            r = torch.exp(energy[::2]- energy[1::2])/((1/self.update.temp[::2])-(1/self.update.temp[1::2]))
                            pswap = torch.minimum(torch.tensor(1.0), r)
                            valid_swaps = pswap > torch.randint(0, 2, (pswap.shape[0],), device = self.device)
                            indices_to_swap = torch.where(valid_swaps)[0]*2 ## valid swaps are on pair indices
                            swapped_energy = energy.clone()
                            swapped_states = self.state.clone()
                            energy[indices_to_swap], energy[indices_to_swap + 1] = (
                                    swapped_energy[indices_to_swap + 1], swapped_energy[indices_to_swap]
                            )

                            self.state[indices_to_swap,:], self.state[indices_to_swap + 1,:] = (
                                    swapped_states[indices_to_swap + 1,:], swapped_states[indices_to_swap,:]
                            )

                    else:  ## odd-leading-index swap {(1, 2), (3, 4) ....}
                        if (self.lib == "numpy"):
                            ### Pt calculations 
                            r = np.exp(energy[1::2] - np.roll(energy[::2], shift=-1))/((1/self.update.temp[1::2])-(1/np.roll(self.update.temp[::2], shift=-1)))
                            pswap = np.minimum(1, r)
                            valid_swaps = pswap > np.random.randint(0, 2, size = pswap.shape[0])

                            # Get the indices where mask is True
                            indices_to_swap = np.where(valid_swaps)[0]*2+1 ## valid swaps are on pair indices
                            
                            # Create a copy of the array to avoid modifying the input array in-place
                            swapped_energy = energy.copy()
                            swapped_states = self.state.copy()
                        
                            # Perform the swap using advanced indexing 
                            ##idx % len(arr) for circular indexing
                            energy[indices_to_swap%len(energy)], energy[(indices_to_swap + 1)%len(energy)] = (
                                    swapped_energy[(indices_to_swap + 1)%len(energy)], swapped_energy[(indices_to_swap)%len(energy)]
                            )

                            self.state[(indices_to_swap)%len(energy),:], self.state[(indices_to_swap + 1)%len(energy),:] = (
                                    swapped_states[(indices_to_swap + 1)%len(energy),:], swapped_states[(indices_to_swap)%len(energy),:]
                            )
                        else:
                            ### Pt calculations 
                            r = torch.exp(energy[1::2] - torch.cat((energy[::2][-1:], energy[::2][:-1])))/((1/self.update.temp[1::2])-(1/torch.cat((self.update.temp[::2][-1:], self.update.temp[::2][:-1]))))
                            pswap = torch.minimum(torch.tensor(1.0), r)
                            valid_swaps = pswap > torch.randint(0, 2,(pswap.shape[0],), device = self.device)
                            indices_to_swap = torch.where(valid_swaps)[0]*2+1 ## valid swaps are on pair indices
                            swapped_energy = energy.clone()
                            swapped_states = self.state.clone()
                        
                            # Perform the swap using advanced indexing 
                            ##idx % len(arr) for circular indexing
                            energy[indices_to_swap%len(energy)], energy[(indices_to_swap + 1)%len(energy)] = (
                                    swapped_energy[(indices_to_swap + 1)%len(energy)], swapped_energy[(indices_to_swap)%len(energy)]
                            )

                            self.state[(indices_to_swap)%len(energy),:], self.state[(indices_to_swap + 1)%len(energy),:] = (
                                    swapped_states[(indices_to_swap + 1)%len(energy),:], swapped_states[(indices_to_swap)%len(energy),:]
                            )                                        

                 

                if(return_states == True):    # Logging states
                    state_array[nn*self.batch_size:(nn + 1)*self.batch_size, ii, :] = self.state.reshape((self.batch_size, -1))
                  

                self.observables["ising_energy"][nn*self.batch_size:(nn + 1)*self.batch_size, ii] = energy

        if(self.lib == 'pytorch'):
            if(self.observables["ising_energy"].is_cuda):
                self.observables["ising_energy"] = self.observables["ising_energy"].cpu().detach().numpy()
                
                if(return_states == True):
                    state_array = state_array.cpu().detach().numpy()
                return state_array, self.observables["ising_energy"]
            else:
                self.observables["ising_energy"] = self.observables["ising_energy"].numpy()
                    
                if(return_states == True):
                    state_array = state_array.numpy()
                    return state_array, self.observables["ising_energy"]
                
        elif(return_states == True):
            return state_array, self.observables["ising_energy"]

            