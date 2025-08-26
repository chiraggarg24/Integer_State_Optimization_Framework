
import numpy as np
import copy
import torch
from src import settings
from scipy.optimize import root
from src.graph_coloring_mapping_util import *
from src.tsp_util import *
import tsplib95
import os
import math

__all__ = ["Ising_Graph_Coloring", "Ising_Graph_Coloring_vectorized", "Ising_TSP_vectorized","Ising_TSP", "Ising_Graph_Coloring_benchmark",  "Ising_Graph_Coloring_hyperparam"]





class Ising_Graph_Coloring():

    """Problem class that handles ising graph coloring problems

      

      Args:

          nodes (1D dictionary): {0, 1, 2, 3}

          edges  (Tuple containing all the edges [(0, 1), (0, 2), (1, 3), (2, 3)]

          spin_system (0 or 1): 0 is for +1/-1 system and 1 is for 0/1 system
          Always 1 for the problems implemented 

          num_colors: Number of the colors that need to be used 
          

    """

      

    def __init__(self, file_prefix, nodes, edges, num_colors, spin_system = 1):
        self.file_prefix = file_prefix

        self.nodes = nodes

        self.edges = edges

        self.num_colors = num_colors   
        self.ising =1
        self.spin_system = spin_system    # 1: For 0/1    0: For +1/-1
        self.A=1 ## Multiplicative constant
        self.device = settings.properties["DEVICE"]
        self.hamiltonian = {}
        self.state_space = []

        if self.device == 'cpu':
            self.hamiltonian, self.state_space, self.Hamiltonian_matrix = create_ising_hamiltonian(self.nodes, self.edges, self.num_colors, self.device, self.A)
        else:
            self.hamiltonian, self.state_space, self.Hamiltonian_matrix = create_ising_hamiltonian(self.nodes, self.edges, self.num_colors,self.device, self.A )


    

    

    def return_hamiltonian(self): 

        """ Returns the Hamiltonian as dictionary of the ising problem """

        return self.hamiltonian

    def return_state_space(self): 

        """ Returns the state space tuple of the ising problem """


        return self.state_space

    def calc_ising_energy(self, state, mode = 'single_flip'):

        """ Returns the ising energy depending upon the state 

        

          Args:

              state: should be dictionary something like this {(0, 0): 1/0} 

        

        """
        state_config = {}
        energy = np.zeros(state.shape[0])
        if(mode == 'single_flip'):
            if self.device == 'cpu':
                energy = calculate_ising_hamiltonian_batch_parallel(self.nodes, self.hamiltonian, state, self.state_space)
            else:
                energy = calculate_ising_hamiltonian_matrix(self.Hamiltonian_matrix, state, self.nodes)
        return energy

      

      

    def calc_delta_energy(self, spin_flip, state, mode = 'single_flip'):

        """ Returns the change in ising energy according to spin flips 

        
          Args:

              spin_flips : node and color index of spin flip (node, color)

              state: should be dictionary something like this {(0, 0): 1/0} 
        

        """
        state_config = {}
        delta_E = np.zeros(state.shape[0])
        if(mode == 'single_flip'):
            if self.device == 'cpu':
                delta_E = delta_energy_batch_parallel(self.nodes,  self.hamiltonian, state, self.state_space, spin_flip[0], spin_flip[1])
            else:
                delta_E = delta_energy_matrix_torch(self.nodes, self.Hamiltonian_matrix, state, self.num_colors, spin_flip[0], spin_flip[1])
        else:

            raise NotImplementedError("Block flips for Ising 1D not yet implemented")

        return delta_E
    
class Ising_Graph_Coloring_hyperparam():

    """Problem class that handles ising graph coloring problems

      

      Args:

          nodes (1D dictionary): {0, 1, 2, 3}

          edges  (Tuple containing all the edges [(0, 1), (0, 2), (1, 3), (2, 3)]

          spin_system (0 or 1): 0 is for +1/-1 system and 1 is for 0/1 system
          Always 1 for the problems implemented 

          num_colors: Number of the colors that need to be used 
          

    """

      

    def __init__(self, file_prefix, nodes, edges, num_colors, A, B, spin_system = 1):

        self.file_prefix = file_prefix
        # print(self.file_prefix)
        self.nodes = nodes

        self.edges = edges

        self.num_colors = num_colors   
        self.ising =1
        self.spin_system = spin_system    # 1: For 0/1    0: For +1/-1
        self.A=1 ## Multiplicative constant
        self.device = settings.properties["DEVICE"]
        self.hamiltonian = {}
        self.state_space = []
        self.A=A ## Multiplicative constant
        self.B=B

        if self.device == 'cpu':
            self.hamiltonian, self.state_space, self.Hamiltonian_matrix = create_ising_hamiltonian_optm(self.nodes, self.edges, self.num_colors, self.device, self.A, self.B)
        else:
            self.hamiltonian, self.state_space, self.Hamiltonian_matrix = create_ising_hamiltonian_optm(self.nodes, self.edges, self.num_colors,self.device, self.A , self.B)


    

    

    def return_hamiltonian(self): 

        """ Returns the Hamiltonian as dictionary of the ising problem """

        return self.hamiltonian

    def return_state_space(self): 

        """ Returns the state space tuple of the ising problem """


        return self.state_space

    def calc_ising_energy(self, state, mode = 'single_flip'):

        """ Returns the ising energy depending upon the state 

        

          Args:

              state: should be dictionary something like this {(0, 0): 1/0} 

        

        """
        state_config = {}
        energy = np.zeros(state.shape[0])
        if(mode == 'single_flip'):
            if self.device == 'cpu':
                energy = calculate_ising_hamiltonian_batch_parallel(self.nodes, self.hamiltonian, state, self.state_space)
            else:
                energy = calculate_ising_hamiltonian_matrix_optm(self.Hamiltonian_matrix, state, self.nodes, self.B)
        return energy

      

      

    def calc_delta_energy(self, spin_flip, state, mode = 'single_flip'):

        """ Returns the change in ising energy according to spin flips 

        
          Args:

              spin_flips : node and color index of spin flip (node, color)

              state: should be dictionary something like this {(0, 0): 1/0} 
        

        """
        state_config = {}
        delta_E = np.zeros(state.shape[0])
        if(mode == 'single_flip'):
            if self.device == 'cpu':
                delta_E = delta_energy_batch_parallel(self.nodes,  self.hamiltonian, state, self.state_space, spin_flip[0], spin_flip[1])
            else:
                delta_E = delta_energy_matrix_torch(self.nodes, self.Hamiltonian_matrix, state, self.num_colors, spin_flip[0], spin_flip[1])
        else:

            raise NotImplementedError("Block flips for Ising 1D not yet implemented")

        return delta_E
    
class Ising_Graph_Coloring_benchmark():

    """Problem class that handles ising graph coloring problems

      

      Args:

          nodes (1D dictionary): {0, 1, 2, 3}

          edges  (Tuple containing all the edges [(0, 1), (0, 2), (1, 3), (2, 3)]

          spin_system (0 or 1): 0 is for +1/-1 system and 1 is for 0/1 system
          Always 1 for the problems implemented 

          num_colors: Number of the colors that need to be used 
          

    """

      

    def __init__(self, file_prefix, nodes, edges, num_colors, A, B,  spin_system = 1):
        self.file_prefix = file_prefix

        self.nodes = nodes

        self.edges = edges

        self.num_colors = num_colors   
        self.ising =1
        self.spin_system = spin_system    # 1: For 0/1    0: For +1/-1
        self.A=A ## Multiplicative constant
        self.B=B
        self.device = settings.properties["DEVICE"]
        self.hamiltonian = {}
        self.state_space = []

        if self.device == 'cpu':
            self.hamiltonian, self.state_space, self.Hamiltonian_matrix = create_ising_hamiltonian_optm(self.nodes, self.edges, self.num_colors, self.device, self.A, self.B)
        else:
            self.hamiltonian, self.state_space, self.Hamiltonian_matrix = create_ising_hamiltonian_optm(self.nodes, self.edges, self.num_colors,self.device, self.A , self.B)


    

    def return_hamiltonian_matrix(self): 

        """ Returns the Hamiltonian as dictionary of the ising problem """

        return self.Hamiltonian_matrix   

    def return_hamiltonian(self): 

        """ Returns the Hamiltonian as dictionary of the ising problem """

        return self.hamiltonian

    def return_state_space(self): 

        """ Returns the state space tuple of the ising problem """


        return self.state_space


class Ising_Graph_Coloring_vectorized():

    """Problem class that handles ising graph coloring problems

      

      Args:

          nodes (1D dictionary): {0, 1, 2, 3}

          edges  (Tuple containing all the edges [(0, 1), (0, 2), (1, 3), (2, 3)]

          spin_system (0 or 1): 0 is for +1/-1 system and 1 is for 0/1 system
          Always 1 for the problems implemented 

          num_colors: Number of the colors that need to be used 
          

    """

      

    def __init__(self,file_prefix, nodes, edges, num_colors, spin_system = 1):
        self.file_prefix = file_prefix
        self.nodes = nodes

        self.edges = edges
        self.ising =0
        self.num_colors = num_colors   
        self.mux_impl = 1  

        self.spin_system = spin_system    # 1: For 0/1    0: For +1/-1

        self.A=1 ## Multiplicative constant

        self.device = settings.properties["DEVICE"]
        self.lib = settings.properties["LIB"]
        self.hamiltonian = {}
        self.state_space = []
        self.hamiltonian_expression = ""
        self.gate_expression = ""
        self.or_gate_count = 0  ## equivalent of number of terms in logic expression
        self.hamiltonian_mat = []
        self.vecmul_mat = []
        self.num_colors_bits = int(math.floor(math.log2(self.num_colors-1) +1))
        self.problem_pubmed = (self.file_prefix == 'pubmed_graph.col')

        if self.mux_impl != 1:
            if self.device == 'cpu':
                self.hamiltonian, self.hamiltonian_expression, self.gate_expression, self.or_gate_count = generate_vectorized_hamiltonian(self.nodes, self.edges, self.num_colors, self.A)
                # self.state_space = generate_vectorized_unique_spin_tuple(self.hamiltonian)
                self.state_space = generate_vectorized_unique_spin_tuple_in_order(self.nodes, self.num_colors)
                self.padded_indices, self.mask, self.coefficients, self.col_width = generate_padded_indices_mask(self.hamiltonian, self.state_space)
            else:
                # raise NotImplementedError("Torch Implementation work in progess")
                self.hamiltonian, self.hamiltonian_expression, self.gate_expression, self.or_gate_count = generate_vectorized_hamiltonian(self.nodes, self.edges, self.num_colors, self.A)
                # self.state_space = generate_vectorized_unique_spin_tuple(self.hamiltonian)
                self.state_space = generate_vectorized_unique_spin_tuple_in_order(self.nodes, self.num_colors)
                #### Only these arrays are conveted to torch arrays as these will help improve vector matrix multiplications to calculate the energy
                self.padded_indices, self.mask, self.coefficients, self.col_width = generate_padded_indices_mask(self.hamiltonian, self.state_space)
                self.padded_indices = torch.from_numpy(self.padded_indices ).to(self.device)
                self.mask = torch.from_numpy(self.mask).to(self.device)
                self.coefficients = torch.from_numpy(self.coefficients).to(device=self.device) #
        else: 
            self.hamiltonian_mat = generate_gc_hamiltonian(self.nodes, self.edges, self.lib, device=self.device)
            self.vecmul_mat = write_vecmul_gc(self.nodes, self.num_colors,self.lib, device=self.device)
            self.state_space = generate_vectorized_unique_spin_tuple_in_order(self.nodes, self.num_colors)

    def return_hamiltonian(self): 

        """ Returns the weight matrix of the ising problem """

        return self.hamiltonian

    def return_state_space(self): 

        """ Returns the state space tuple of the ising problem """
        
        return self.state_space

    def calc_ising_energy(self, state, mode = 'single_flip'):

        """ Returns the ising energy depending upon the state 

        

          Args:

              state: should be dictionary something like this {(0, 0): 1/0} 

        

        """
        state_config = {}
        energy = np.zeros(state.shape[0])
        if(mode == 'single_flip'):
            if self.mux_impl != 1:

                if self.device == 'cpu':
                    energy = calculate_ising_vectorized_parallel(self.hamiltonian, state, self.state_space)
                else:
                    energy = calculate_ising_vectorized_matrix_torch(self.padded_indices, self.mask, state, self.coefficients)
            else:
                energy = calculate_ising_vectorized_gc(self.lib, self.hamiltonian_mat, self.vecmul_mat, state, self.num_colors_bits, self.problem_pubmed)

        return energy
      

    def calc_delta_energy(self, spin_flip, state, mode = 'single_flip'):

        """ Returns the change in ising energy according to spin flips 

          Args:

              spin_flips : node and color index of spin flip (node, color)

              state: should be dictionary something like this {(0, 0): 1/0} 
        

        """
        state_config = {}
        if(mode == 'single_flip'):
            if self.mux_impl != 1:
                if self.device == 'cpu':
                    delta_E = delta_energy_vectorized_batch_parallel(self.hamiltonian, state, self.state_space, spin_flip[0], spin_flip[1])
                else:
                    # raise NotImplementedError("Torch Implementation work in progess")
                    delta_E = delta_energy_vectorized_matrix_torch(self.col_width, self.padded_indices, self.mask, state, self.coefficients, spin_flip[0], spin_flip[1])
            else:
                delta_E = calculate_delta_ising_vectorized_gc_optimized(self.lib, self.hamiltonian_mat, self.vecmul_mat, state, self.num_colors_bits, spin_flip[0], spin_flip[1])
        else:

            raise NotImplementedError("Block flips for Ising 1D not yet implemented")

        return delta_E
 
class Ising_TSP():

    """Problem class that handles ising graph coloring problems

      

      Args:

          nodes : list given by the tsplib

          edges  (Tuple containing all the edges [(0, 1), (0, 2), (1, 3), (2, 3)]

          spin_system (0 or 1): 0 is for +1/-1 system and 1 is for 0/1 system
          Always 1 for the problems implemented 
          

    """

      

    def __init__(self, lib, A, B, spin_system = 1):
        self.file_prefix = lib.name
        self.lib = lib
        Graph = lib.get_graph()
        self.nodes = Graph.nodes

        self.edges = Graph.edges

        self.num_time_len = int(len(self.nodes))
        self.spin_system = spin_system    # 1: For 0/1    0: For +1/-1

        self.A=A ## Multiplicative constant for weights for invalid time states  (same time for 2 nodes)
        self.B=B
        self.ising =1
        self.device = settings.properties["DEVICE"]
        self.weight_mat = []
        self.state_space =[]
        self.vec_mul_mat = []
        self.weight_const = []
        self.linear_sum_wt = 1 ### For the weights with -1 replace them with sum across the weight row instead of max

        if self.device == 'cpu':
            raise NotImplementedError("Numpy Not Implemented ")
        else:
            self.weight_mat = ising_hamiltonian_tsp_torch(lib,self.device, A=self.A, B=self.B)
            self.weight_mat = self.weight_mat
            
    def return_weights_matrix(self): 

        """ Returns the weight matrix of the ising problem """

        return self.weight_mat
    def return_state_space(self): 

        """ Returns the state space tuple of the ising problem """
        state_space = []
        for i in range(len(self.nodes)):
            for j in range(self.num_time_len):
                state_space.append((i,j))
        self.state_space = state_space
        return self.state_space

    def calc_ising_energy(self, state, mode = 'single_flip'):

        """ Returns the ising energy depending upon the state 

        

          Args:

              state: should be dictionary something like this {(0, 0): 1/0} 

        """
        state_config = {}
        energy = np.zeros(state.shape[0])
        if(mode == 'single_flip'):
            if self.device == 'cpu':
                raise NotImplementedError("Numpy Not Implemented ")

            else:
                energy = calculate_ising_hamiltonian_matrix_tsp(self.weight_mat, state, self.nodes, self.A)
        return energy
                      


    def calc_delta_energy(self, spin_flip, state, mode = 'single_flip'):

        """ Returns the change in ising energy according to spin flips 

          Args:

              spin_flips : node and color index of spin flip (node, color)

              state: should be dictionary something like this {(0, 0): 1/0} 
        

        """
        state_config = {}
        if(mode == 'single_flip'):
            if self.device == 'cpu':
                raise NotImplementedError("Numpy Not Implemented ")

            else:
                
                delta_E=delta_energy_matrix_torch_tsp(self.nodes, self.weight_mat, state, len(self.nodes), spin_flip[0], spin_flip[1], self.A)
        else:

            raise NotImplementedError("Block flips for Ising 1D not yet implemented")

        return delta_E

class Ising_TSP_vectorized():

    """Problem class that handles ising graph coloring problems

      

      Args:

          nodes : list given by the tsplib

          edges  (Tuple containing all the edges [(0, 1), (0, 2), (1, 3), (2, 3)]

          spin_system (0 or 1): 0 is for +1/-1 system and 1 is for 0/1 system
          Always 1 for the problems implemented 
          

    """

      

    def __init__(self, lib, spin_system = 1):
        self.file_prefix = lib.name
        self.lib = lib
        Graph = lib.get_graph()
        self.nodes = Graph.nodes

        self.edges = Graph.edges

        self.num_time_len = int(math.floor(math.log2(len(self.nodes)-1) +1))
        self.spin_system = spin_system    # 1: For 0/1    0: For +1/-1

        self.A=2 ## Multiplicative constant for weights for invalid time states  (same time for 2 nodes)
        self.ising =1
        self.device = settings.properties["DEVICE"]
        self.weight_mat = []
        self.state_space =[]
        self.vec_mul_mat = []
        self.weight_const = []
        self.linear_sum_wt = 1 ### For the weights with -1 replace them with sum across the weight row instead of max

        if self.device == 'cpu':
            self.weight_mat = generate_tsp_hamiltonian(lib)
            self.weight_mat = self.weight_mat/np.max(self.weight_mat)
            self.vec_mul_mat = write_vecmul(lib)
            col_sums = np.sum(self.weight_mat, axis=1)  
            col_sums_col_vector = col_sums[:, np.newaxis]  
            self.weight_const = np.repeat(col_sums_col_vector, self.weight_mat.shape[1], axis=1)

        else:
            self.weight_mat = generate_tsp_hamiltonian(lib)
            self.weight_mat = torch.from_numpy(self.weight_mat/np.max(self.weight_mat)).to(self.device)
            self.vec_mul_mat = write_vecmul(lib)
            self.vec_mul_mat = torch.from_numpy(self.vec_mul_mat).to(self.device)
            col_sums = torch.sum(self.weight_mat, dim=1)
            col_sums_col_vector = col_sums.unsqueeze(1)
            self.weight_const = col_sums_col_vector.repeat(1, self.weight_mat.shape[1])

        # print("Weight_matrix",  self.weight_mat)
    def return_weights_matrix(self): 

        """ Returns the weight matrix of the ising problem """

        return self.weight_mat
    def return_state_space(self): 

        """ Returns the state space tuple of the ising problem """
        state_space = []
        for i in range(len(self.nodes)):
            for j in range(self.num_time_len):
                state_space.append((i,j))
        self.state_space = state_space
        
        return self.state_space

    def calc_ising_energy(self, state, mode = 'single_flip'):

        """ Returns the ising energy depending upon the state 

        

          Args:

              state: should be dictionary something like this {(0, 0): 1/0} 

        

        """
        state_config = {}
        energy = np.zeros(state.shape[0])
        if(mode == 'single_flip'):
            if self.device == 'cpu':
                # energy = calculate_ising_vectorized_matrix(self.padded_indices, self.mask, state, self.coefficients)
                energy = calculate_ising_vectorized_tsp_cpu(self.weight_mat, self.weight_const, self.vec_mul_mat, state, self.num_time_len, self.linear_sum_wt, large_wt_mult=self.A)
            else:
                energy = calculate_ising_vectorized_tsp_torch(self.weight_mat, self.weight_const, self.vec_mul_mat, state, self.num_time_len, self.linear_sum_wt, large_wt_mult=self.A)
        return energy
      

    def calc_delta_energy(self, spin_flip, state, mode = 'single_flip'):

        """ Returns the change in ising energy according to spin flips 

          Args:

              spin_flips : node and color index of spin flip (node, color)

              state: should be dictionary something like this {(0, 0): 1/0} 
        

        """
        state_config = {}
        if(mode == 'single_flip'):
            if self.device == 'cpu':
                delta_E = calculate_delta_ising_vectorized_tsp(self.weight_mat, self.weight_const, self.vec_mul_mat, state, self.num_time_len,  spin_flip[0], spin_flip[1], self.linear_sum_wt ,large_wt_mult=self.A)

            else:
                delta_E = calculate_delta_ising_vectorized_tsp_torch(self.weight_mat, self.weight_const, self.vec_mul_mat, state, self.num_time_len,  spin_flip[0], spin_flip[1], self.linear_sum_wt ,large_wt_mult=self.A)
        else:

            raise NotImplementedError("Block flips for Ising 1D not yet implemented")

        return delta_E
    
