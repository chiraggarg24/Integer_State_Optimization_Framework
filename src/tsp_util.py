import tsplib95
import numpy as np
import math
from sklearn.manifold import MDS
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch 
# Load the problem
__all__ = [
           "generate_tsp_hamiltonian",
           "write_vecmul",
           "calculate_ising_vectorized_tsp_cpu",
           "calculate_delta_ising_vectorized_tsp",
           "solution_quality",
           "tsp_tour_cost",
           "calculate_ising_vectorized_swap_tsp_cpu",
           "calculate_delta_ising_vectorized_swap_tsp",
           "solution_quality_swap",
           "plot_route",
           "plot_ising_energy", 
           "calculate_ising_hamiltonian_matrix_tsp",
           "ising_hamiltonian_tsp_torch",
           "delta_energy_matrix_torch_tsp",
           "calculate_delta_ising_vectorized_tsp_torch",
           "calculate_ising_vectorized_tsp_torch",
           "solution_quality_ising", 
           "tsp_ising_param",
           "tsp_vec_param"]
def generate_tsp_hamiltonian(problem): 
    G = problem.get_graph()
    nodes = G.nodes
    weight_mat = np.zeros((len(nodes), len(nodes)), dtype = int)
    for i in nodes: 
        for j in nodes:
            weight_mat[i-1,j-1] = problem.get_weight(i, j)
    np.fill_diagonal(weight_mat, 1*np.max(weight_mat)) ### Diagonal weight is not considered in the overall energy calculation
    return weight_mat


def ising_hamiltonian_tsp_torch(problem, device, A=1.0, B=1.0):
    """
    Constructs the Ising Hamiltonian matrix for TSP using PyTorch.

    Parameters:
    - distance_matrix: NxN PyTorch tensor, where (i, j) is the distance between city i and city j.
    - A: penalty weight for constraint satisfaction.
    - B: penalty weight for path minimization.

    Returns:
    - H: (N^2 x N^2) Ising Hamiltonian matrix.
    """
    G = problem.get_graph()
    nodes = G.nodes
    distance_matrix = np.zeros((len(nodes), len(nodes)), dtype = int)
    for i in nodes: 
        for j in nodes:
            distance_matrix[i-1,j-1] = problem.get_weight(i, j)

    N = distance_matrix.shape[0]
    H = torch.zeros((N**2, N**2), dtype=torch.float32, device = device)  # Ising matrix of size (N^2, N^2)

    # Constraint: Each city must appear exactly once in the sequence
    for i in range(N):
        for j in range(N):
            H[i * N + j, i * N + j] -= A  # Penalize non-unique city entries

            for k in range(j + 1, N):
                H[i * N + j, i * N + k] += 2 * A  # Penalize multiple occurrences in a row
                H[j * N + i, k * N + i] += 2 * A  # Penalize multiple occurrences in columns

    # Distance penalty: Encourage shorter paths
    for i in range(N):
        for j in range(N):
            if i != j:
                for k in range(N - 1):
                    H[i * N + k, j * N + (k + 1)] += B * distance_matrix[i, j] /np.max(distance_matrix)
    # print("Hamiltonian", H)
    return H

def calculate_ising_hamiltonian_matrix_tsp(Hamiltonian_matrix, state, nodes, A):
    energy = 2*A*len(nodes) + torch.sum(state*(Hamiltonian_matrix@state.T).T, axis = 1, keepdims = True)
    return energy.squeeze()   

tsp_ising_param ={'burma4': [0.125, 5], 'burma6': [0.125, 5], 'burma8': [0.25, 10], 'burma10': [0.25, 15], 'burma12': [0.25, 15], 'burma14': [0.25, 5]}
tsp_vec_param ={'burma4': 0.4, 'burma6': 0.2, 'burma8': 0.2, 'burma10': 0.1, 'burma12': 0.1, 'burma14': 0.1}

def delta_energy_matrix_torch_tsp(nodes, Hamiltonian_matrix, state, num_city, city, time, A):

    # delta_E = torch.zeros(state.shape[0])
    flip_index = int((time)+(city)*num_city) 
        # Calculate current energy

    current_energy = calculate_ising_hamiltonian_matrix_tsp(Hamiltonian_matrix, state, nodes, A)
    # Flip the spin

    state[:,flip_index] =  1 - state[:,flip_index]
    
    # Calculate new energy
    new_energy = calculate_ising_hamiltonian_matrix_tsp(Hamiltonian_matrix, state, nodes, A)
    # Flip the spin back
    state[:,flip_index] =  1 - state[:,flip_index]
    # Calculate delta energy
    delta_E = new_energy - current_energy

    return delta_E

def write_vecmul(problem):
    G = problem.get_graph()
    nodes = G.nodes
    num_nodes = len(nodes)
    num_time = int(math.floor(math.log2(num_nodes-1) +1))
    vecmul_array = np.zeros((num_nodes,int(math.pow(2, 2*num_time))))

    ### 0 if time stamp are seprated >1 index (So overall it should be reaching there but this won't be possible)
    #### 1 the states in time stamp are seprated by 1 will be minimized 
    #### -1 is deadzone with high -ve weight which should be avoided
    for j in range(num_nodes):
        if j == 0:
            vecmul_array[:, int(j+math.pow(2, num_time)*(j+1))] = 1
            vecmul_array[:, int(j+1+math.pow(2, num_time)*(j))] = 1
            vecmul_array[:, int(j+math.pow(2, num_time)*( num_nodes-1))] = 1
            vecmul_array[:, int(num_nodes-1+math.pow(2, num_time)*(j))] = 1

        elif (j == num_nodes-1):
            vecmul_array[:, int(j+math.pow(2, num_time)*(j-1))] = 1
            vecmul_array[:, int(j-1+math.pow(2, num_time)*(j))] = 1
        else :
            vecmul_array[:, int(j+math.pow(2, num_time)*(j-1))] = 1
            vecmul_array[:, int(j+math.pow(2, num_time)*(j+1))] = 1          
            vecmul_array[:, int(j-1+math.pow(2, num_time)*(j))] = 1
            vecmul_array[:, int(j+1+math.pow(2, num_time)*(j))] = 1   

    for j in range(num_nodes):
            vecmul_array[:, int(j+math.pow(2, num_time)*(j))] = -1
    ### thier weights require some tuning

    num_arr = np.arange(0,int(math.pow(2, num_time)))
    for k in range(num_nodes, int(math.pow(2, num_time))):
        vecmul_array[:, (math.pow(2, num_time)*num_arr+k).astype(int)] = -2
        vecmul_array[:, (k*math.pow(2, num_time)+num_arr).astype(int)] = -2
    ### put these -1 as really high numbers
    return vecmul_array

def calculate_ising_vectorized_tsp_cpu(weight_mat, weight_const, vecmul_array, state, num_time, linear_sum_wt, large_wt_mult=2):
    state1 = state.reshape(state.shape[0], int(state.shape[1]/num_time), int(num_time))

    # print("State1", state1)
    # state_int = np.zeros((state1.shape[0], state1.shape[1]))
    # for i in range(state1.shape[0]):
    #     state_int[i,:] = np.array([int(''.join(map(str, row)), 2) for row in state1[i,:,:]])

    # Flatten the last two dimensions
    # state_flattened = state1.reshape(state1.shape[0], -1, state1.shape[-1])
    # print("State Flattened", state_flattened)
  
    # Convert each row to binary strings and then to integers
    # state_int = np.apply_along_axis(lambda row: int(''.join(row.astype(str)), 2), axis=2, arr=state_flattened)

    powers_of_two = 2 ** np.arange(state1.shape[-1])[::-1]
    state_int = np.dot(state1, powers_of_two)

    vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_time))*state_int[:, np.newaxis, :]

    # mask = np.zeros_like(vecmul_inp_array)
    # for k in range(vecmul_inp_array.shape[0]):
    #     for i in range(vecmul_inp_array.shape[1]):
    #         mask[k,i,:] = [row[int(idx)] for row, idx in zip(vecmul_array, vecmul_inp_array[k,i,:])]

    mask1 = np.zeros_like(vecmul_inp_array)
    # Expand dimensions to match shapes correctly
    vecmul_array_expanded = np.expand_dims(vecmul_array, axis=0)  # Shape (1, 14, 256)
    vecmul_array_expanded = np.expand_dims(vecmul_array_expanded, axis=0)  # Shape (1, 14, 256)

    vecmul_inp_expanded = np.expand_dims(vecmul_inp_array, axis=-1)  # Shape (40, 14, 14, 1)

    # Gather elements using np.take_along_axis
    mask1 = np.take_along_axis(vecmul_array_expanded, vecmul_inp_expanded, axis=3)
    mask1 = np.squeeze(mask1, axis=-1)


    diag_indices = np.arange(mask1.shape[1])
    mask1[:, diag_indices, diag_indices] = 0

    # print("Mask compare", np.sum(mask-mask1))
    energy_array = np.where(mask1 == 1, weight_mat, mask1)
    if linear_sum_wt ==0:
        energy_array[energy_array == -1] = np.max(weight_mat)*large_wt_mult
    else: 
        weight_const = np.repeat(weight_const[np.newaxis, :, :], energy_array.shape[0], axis=0)
        energy_array[energy_array == -1] = weight_const[energy_array == -1]*large_wt_mult

    energy_array[energy_array == -2] = np.max(weight_mat)*20
    return np.sum(energy_array, axis=(1,2))

def calculate_ising_vectorized_tsp_torch(weight_mat, weight_const, vecmul_array, state, num_time, linear_sum_wt, large_wt_mult=2):
    state1 = state.reshape(state.shape[0], int(state.shape[1]/num_time), int(num_time))
    state1 = state1.to(torch.float32)
    vecmul_array = vecmul_array.to(torch.float32)

    powers_of_two =  2 ** torch.arange(state1.shape[-1] - 1, -1, -1, dtype=torch.float32, device = state1.device)

    state_int = torch.matmul(state1, powers_of_two)

    vecmul = vecmul_array.unsqueeze(0).unsqueeze(1).repeat(state.shape[0], int(state.shape[1]/num_time), 1, 1)

    vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_time))*state_int[:, np.newaxis, :]

    vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)



    # Gather elements using np.take_along_axis
    mask1 = vecmul.gather(3, vecmul_inp_array_expanded).squeeze(-1) 


    diag_indices = torch.arange(mask1.shape[1])
    mask1[:, diag_indices, diag_indices] = 0

    energy_array = torch.where(mask1 == 1, weight_mat, mask1)
    if linear_sum_wt ==0:
        energy_array[energy_array == -1] = torch.max(weight_mat)*large_wt_mult
    else: 
        weight_const = weight_const.unsqueeze(0).expand(energy_array.shape[0], -1, -1)
        energy_array[energy_array == -1] = weight_const[energy_array == -1]*large_wt_mult

    energy_array[energy_array == -2] = torch.max(weight_mat)*20
    return torch.sum(energy_array, dim=(1,2))

def calculate_delta_ising_vectorized_tsp_torch(weight_mat, weight_const, vecmul_array, state, num_time,  node_tsp, indx, linear_sum_wt, large_wt_mult=2):
    flip_index = int(num_time)*node_tsp + indx

    state1 = state.reshape(state.shape[0], int(state.shape[1]/num_time), int(num_time))
    state1 = state1.to(torch.float32)
    vecmul_array = vecmul_array.to(torch.float32)
    powers_of_two =  2 ** torch.arange(state1.shape[-1] - 1, -1, -1, dtype=torch.float32, device = state1.device)

    state_int = torch.matmul(state1, powers_of_two)
    vecmul = vecmul_array.unsqueeze(0).unsqueeze(1).repeat(state.shape[0], int(state.shape[1]/num_time), 1, 1)

    vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_time))*state_int[:, np.newaxis, :]

    vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)

    mask =  vecmul.gather(3, vecmul_inp_array_expanded).squeeze(-1) 

    diag_indices = torch.arange(mask.shape[1])
    mask[:, diag_indices, diag_indices] = 0

    energy_array_in = torch.where(mask == 1, weight_mat, mask)
    if linear_sum_wt ==0:
        energy_array_in[energy_array_in == -1] = torch.max(weight_mat)*large_wt_mult
    else: 
        weight_const =  weight_const.unsqueeze(0).expand(energy_array_in.shape[0], -1, -1)
        energy_array_in[energy_array_in == -1] = weight_const[energy_array_in == -1]*large_wt_mult

    energy_array_in[energy_array_in == -2] = torch.max(weight_mat)*20


    state1[:,node_tsp, indx] =  1 -  state1[:,node_tsp, indx]

    powers_of_two = 2 ** torch.arange(state1.shape[-1] - 1, -1, -1, dtype=torch.float32, device = state1.device)
    state_int = torch.matmul(state1, powers_of_two)

    vecmul = vecmul_array.unsqueeze(0).unsqueeze(1).repeat(state.shape[0], int(state.shape[1]/num_time), 1, 1)

    vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_time))*state_int[:, np.newaxis, :]

    vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)

    mask =  vecmul.gather(3, vecmul_inp_array_expanded).squeeze(-1) 

    diag_indices = torch.arange(mask.shape[1])
    mask[:, diag_indices, diag_indices] = 0

    energy_array_f = torch.where(mask == 1, weight_mat, mask)
    if linear_sum_wt ==0:
        energy_array_f[energy_array_f == -1] = torch.max(weight_mat)*large_wt_mult
    else: 
        energy_array_f[energy_array_f == -1] = weight_const[energy_array_f == -1]*large_wt_mult

    energy_array_f[energy_array_f == -2] = torch.max(weight_mat)*20

    state1[:,node_tsp, indx] =  1 -  state1[:,node_tsp, indx]

    return (torch.sum(energy_array_f, dim=(1,2)) - torch.sum(energy_array_in, dim=(1,2)))



def calculate_delta_ising_vectorized_tsp(weight_mat, weight_const, vecmul_array, state, num_time,  node_tsp, indx, linear_sum_wt, large_wt_mult=2):
    flip_index = int(num_time)*node_tsp + indx

    state1 = state.reshape(state.shape[0], int(state.shape[1]/num_time), int(num_time))

    # state_int = np.zeros((state1.shape[0], state1.shape[1]))
    # for i in range(state1.shape[0]):
    #     state_int[i,:] = np.array([int(''.join(map(str, row)), 2) for row in state1[i,:,:]])
    # state_flattened = state1.reshape(state1.shape[0], -1, state1.shape[-1])
    # state_int = np.apply_along_axis(lambda row: int(''.join(row.astype(str)), 2), axis=2, arr=state_flattened)


    powers_of_two = 2 ** np.arange(state1.shape[-1])[::-1]
    state_int = np.dot(state1, powers_of_two)
    vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_time))*state_int[:, np.newaxis, :]

    # mask = np.zeros_like(vecmul_inp_array)
    # for k in range(vecmul_inp_array.shape[0]):
    #     for i in range(vecmul_inp_array.shape[1]):
    #         mask[k,i,:] = [row[int(idx)] for row, idx in zip(vecmul_array, vecmul_inp_array[k,i,:])]

    vecmul_array_expanded = np.expand_dims(vecmul_array, axis=0)  # Shape (1, 14, 256)
    vecmul_array_expanded = np.expand_dims(vecmul_array_expanded, axis=0)  # Shape (1, 14, 256)
    vecmul_inp_expanded = np.expand_dims(vecmul_inp_array, axis=-1)  # Shape (40, 14, 14, 1)
    mask = np.take_along_axis(vecmul_array_expanded, vecmul_inp_expanded, axis=-1)
    mask = np.squeeze(mask, axis=-1)

    diag_indices = np.arange(mask.shape[1])
    mask[:, diag_indices, diag_indices] = 0

    energy_array_in = np.where(mask == 1, weight_mat, mask)
    if linear_sum_wt ==0:
        energy_array_in[energy_array_in == -1] = np.max(weight_mat)*large_wt_mult
    else: 
        weight_const = np.repeat(weight_const[np.newaxis, :, :], energy_array_in.shape[0], axis=0)
        energy_array_in[energy_array_in == -1] = weight_const[energy_array_in == -1]*large_wt_mult

    energy_array_in[energy_array_in == -2] = np.max(weight_mat)*20


    state1[:,node_tsp, indx] =  1 -  state1[:,node_tsp, indx]
    # state[:,flip_index] =  1 -  state[:,flip_index]
    # state1 = state.reshape(state.shape[0], int(state.shape[1]/num_time), int(num_time))


    # state_int = np.zeros((state1.shape[0], state1.shape[1]))
    # for i in range(state1.shape[0]):
    #     state_int[i,:] = np.array([int(''.join(map(str, row)), 2) for row in state1[i,:,:]])
    # state_flattened = state1.reshape(state1.shape[0], -1, state1.shape[-1])
    # state_int = np.apply_along_axis(lambda row: int(''.join(row.astype(str)), 2), axis=2, arr=state_flattened)


    powers_of_two = 2 ** np.arange(state1.shape[-1])[::-1]
    state_int = np.dot(state1, powers_of_two)

    vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_time))*state_int[:, np.newaxis, :]

    # mask = np.zeros_like(vecmul_inp_array)
    # for k in range(vecmul_inp_array.shape[0]):
    #     for i in range(vecmul_inp_array.shape[1]):
    #         mask[k,i,:] = [row[int(idx)] for row, idx in zip(vecmul_array, vecmul_inp_array[k,i,:])]
    vecmul_array_expanded = np.expand_dims(vecmul_array, axis=0)  
    vecmul_array_expanded = np.expand_dims(vecmul_array_expanded, axis=0)  
    vecmul_inp_expanded = np.expand_dims(vecmul_inp_array, axis=-1) 
    mask = np.take_along_axis(vecmul_array_expanded, vecmul_inp_expanded, axis=-1)
    mask = np.squeeze(mask, axis=-1)

    diag_indices = np.arange(mask.shape[1])
    mask[:, diag_indices, diag_indices] = 0

    energy_array_f = np.where(mask == 1, weight_mat, mask)
    if linear_sum_wt ==0:
        energy_array_f[energy_array_f == -1] = np.max(weight_mat)*large_wt_mult
    else: 
        energy_array_f[energy_array_f == -1] = weight_const[energy_array_f == -1]*large_wt_mult

    energy_array_f[energy_array_f == -2] = np.max(weight_mat)*20

    state1[:,node_tsp, indx] =  1 -  state1[:,node_tsp, indx]
    # state[:,flip_index] =  1 -  state[:,flip_index]

    return (np.sum(energy_array_f, axis=(1,2)) - np.sum(energy_array_in, axis=(1,2)))

def solution_quality_ising(lib, states, num_states_test, ising_energy, num_tsp_nodes, num_time_len):
    true_state_count = 0
    total_state_tested = states.shape[0]*num_states_test
    tour_cost =[]
    states_val = []
    for i in range(states.shape[0]):
        for j in range(num_states_test):
            mod_state = np.array(states[i,-j,:].reshape( int(num_tsp_nodes), num_time_len), dtype = int)
            state_int_array = np.argmax(mod_state, axis=0).tolist()
            if (np.all(np.sum(mod_state, axis=1) == 1)) and np.all(np.sum(mod_state, axis=0) == 1):
                true_state_count +=1
                sample_tour_cost = 0
                state_int_array = np.argmax(mod_state, axis=1).tolist()
                ### test state in array and tour cost calculation
                for k in range(len(state_int_array)):
                    sample_tour_cost += lib.get_weight(state_int_array[k-1]+1, state_int_array[k]+1)
                tour_cost.append(sample_tour_cost)
                states_val.append(state_int_array)
            else:
                tour_cost.append(float('inf'))
                states_val.append(state_int_array)

    min_ising_energy = np.min(ising_energy)
    # print(tour_cost)
    return true_state_count/total_state_tested, min_ising_energy, tour_cost, states_val


def solution_quality(lib, states, num_states_test, ising_energy, num_tsp_nodes, num_time_len):
    true_state_count = 0
    total_state_tested = states.shape[0]*num_states_test
    tour_cost =[]
    states_val = []
    for i in range(states.shape[0]):
        for j in range(num_states_test):
            mod_state = np.array(states[i,-j,:].reshape( int(num_tsp_nodes), num_time_len), dtype = int)
            state_int_array = np.array([int(''.join(map(str, row)), 2) for row in mod_state])
            if (len(np.unique(state_int_array)) == int(num_tsp_nodes)) and (np.max(state_int_array) < int(num_tsp_nodes)):
                true_state_count +=1
                sample_tour_cost = 0
                for k in range(len(state_int_array)):
                    sample_tour_cost += lib.get_weight(state_int_array[k-1]+1, state_int_array[k]+1)
                tour_cost.append(sample_tour_cost)
                states_val.append(state_int_array)
                # print("Tour Cost", sample_tour_cost)
            else:
                tour_cost.append(float('inf'))
                states_val.append(state_int_array)

    min_ising_energy = np.min(ising_energy)
    # print(tour_cost)
    return true_state_count/total_state_tested, min_ising_energy, tour_cost, states_val

def tsp_tour_cost(lib, states, ising_energy, num_states_test, num_tsp_nodes, num_time_len):
    ### Get optimal tour from the dataset 
    # optimal_tour_load = tsplib95.load(f"../Dataset/tsp_datset/tour/{lib.name}.opt.tour")
    # optimal_tour = list(optimal_tour_load.tours[0])
    # optimal_tour_cost = 0
    # for i in range(len(optimal_tour)):
    #     optimal_tour_cost += lib.get_weight(optimal_tour[i-1], optimal_tour[i])
    optimal_tour = None 
    optimal_tour_cost = None 

    min_index_flat = np.argmin(ising_energy)
    min_index = np.unravel_index(min_index_flat, ising_energy.shape)
    
    mod_state = np.array(states[min_index[0],min_index[1],:].reshape( int(num_tsp_nodes), num_time_len), dtype = int)
    state_int_array = np.array([int(''.join(map(str, row)), 2) for row in mod_state])
    sample_correctness = (len(np.unique(state_int_array)) == int(num_tsp_nodes))
    sample_tour_cost = 0
    for i in range(len(state_int_array)):
        sample_tour_cost += lib.get_weight(state_int_array[i-1]+1, state_int_array[i]+1)
    
    return optimal_tour, optimal_tour_cost, state_int_array, sample_correctness, sample_tour_cost




def calculate_ising_vectorized_swap_tsp_cpu(weight_mat, vecmul_array, state, num_time):
    vecmul_inp_array = state[:, :, np.newaxis] + int(math.pow(2, num_time))*state[:, np.newaxis, :]
    
    # mask = np.zeros_like(vecmul_inp_array)
    # for k in range(vecmul_inp_array.shape[0]):
    #     for i in range(vecmul_inp_array.shape[1]):
    #         mask[k,i,:] = [row[int(idx)] for row, idx in zip(vecmul_array, vecmul_inp_array[k,i,:])]
    vecmul_array_expanded = np.expand_dims(vecmul_array, axis=0)  
    vecmul_array_expanded = np.expand_dims(vecmul_array_expanded, axis=0)  
    vecmul_inp_expanded = np.expand_dims(vecmul_inp_array, axis=-1) 
    mask = np.take_along_axis(vecmul_array_expanded, vecmul_inp_expanded, axis=-1)
    mask = np.squeeze(mask, axis=-1)

    energy_array = np.where(mask == 1, weight_mat, mask)
    energy_array[energy_array == -1] = 0 

    ### These cases won't happen
    # energy_array[energy_array == -2] = np.max(weight_mat)*20

    return np.sum(energy_array, axis=(1,2))



def calculate_delta_ising_vectorized_swap_tsp(weight_mat, vecmul_array, state, num_time,  swap_pair):

    vecmul_inp_array = state[:, :, np.newaxis] + int(math.pow(2, num_time))*state[:, np.newaxis, :]

    # mask = np.zeros_like(vecmul_inp_array)
    # for k in range(vecmul_inp_array.shape[0]):
    #     for i in range(vecmul_inp_array.shape[1]):
    #         mask[k,i,:] = [row[int(idx)] for row, idx in zip(vecmul_array, vecmul_inp_array[k,i,:])]
    vecmul_array_expanded = np.expand_dims(vecmul_array, axis=0)  
    vecmul_array_expanded = np.expand_dims(vecmul_array_expanded, axis=0)  
    vecmul_inp_expanded = np.expand_dims(vecmul_inp_array, axis=-1) 
    mask = np.take_along_axis(vecmul_array_expanded, vecmul_inp_expanded, axis=-1)
    mask = np.squeeze(mask, axis=-1)

    energy_array_in = np.where(mask == 1, weight_mat, mask)
    energy_array_in[energy_array_in == -1] = 0
    # print("Energy In ", energy_array_in)
    state1 =  np.copy(state)
    state1[:, swap_pair[0]] = state[:, swap_pair[1]]
    state1[:, swap_pair[1]] = state[:, swap_pair[0]]
    vecmul_inp_array = state1[:, :, np.newaxis] + int(math.pow(2, num_time))*state1[:, np.newaxis, :]

    # mask = np.zeros_like(vecmul_inp_array)
    # for k in range(vecmul_inp_array.shape[0]):
    #     for i in range(vecmul_inp_array.shape[1]):
    #         mask[k,i,:] = [row[int(idx)] for row, idx in zip(vecmul_array, vecmul_inp_array[k,i,:])]
    vecmul_array_expanded = np.expand_dims(vecmul_array, axis=0)  
    vecmul_array_expanded = np.expand_dims(vecmul_array_expanded, axis=0)  
    vecmul_inp_expanded = np.expand_dims(vecmul_inp_array, axis=-1) 
    mask = np.take_along_axis(vecmul_array_expanded, vecmul_inp_expanded, axis=-1)
    mask = np.squeeze(mask, axis=-1)
    
    energy_array_f = np.where(mask == 1, weight_mat, mask)
    energy_array_f[energy_array_f == -1] = 0

    state1 =  state
    return (np.sum(energy_array_f, axis=(1,2)) - np.sum(energy_array_in, axis=(1,2)))



def solution_quality_swap(lib, states, ising_energy):
    tour_cost =  np.zeros((states.shape[0], states.shape[1]))
    mod_state1 = np.zeros((states.shape[0], states.shape[1], states.shape[2]))
    for i in range(states.shape[0]):
        for j in range(states.shape[1]):
            mod_state = states[i,j,:]
            sample_tour_cost = 0
            for k in range(len(mod_state)):
                sample_tour_cost += lib.get_weight(int(mod_state[k-1]+1), int(mod_state[k]+1))
            tour_cost[i, j] =  sample_tour_cost

            mod_state1[i, j,:] = mod_state

                
    min_ising_energy = np.min(ising_energy)
    return min_ising_energy, tour_cost, mod_state1


def distance_matrix_to_coordinates(distance_matrix, n_components=2):
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=0)
    coordinates = mds.fit_transform(distance_matrix)
    return coordinates

def plot_route(weight_matrix,  route_sequence):
    coords = distance_matrix_to_coordinates(weight_matrix)
    plt.figure(figsize=(8, 8))
    x = coords[:, 0]
    y = coords[:, 1]

    # Plot cities
    # plt.scatter(x, y, color='blue')
    # for i, (xi, yi) in enumerate(zip(x, y)):
    #     plt.text(xi, yi, f'City {i+1}', fontsize=12, ha='right')
    # Plot cities with circle ('o' symbol) and city numbers inside
    plt.scatter(x, y, color='blue', s=200, edgecolor='black', zorder=3)
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, f'{i+1}', fontsize=12, ha='center', va='center', color='white', zorder=4)

    # Plot route
    route_coords = coords[np.array(route_sequence) - 1]
    plt.plot(route_coords[:, 0], route_coords[:, 1], 'r-', marker='o')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('TSP Route')
    plt.grid(True)
    plt.show()


def plot_ising_energy(ising_energy):
    num_subplots = ising_energy.shape[0]

    # Create subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 2 * num_subplots), sharex=True)

    # Plot each energy[i, :] in its own subplot
    for i in range(num_subplots):
        axs[i].plot(np.arange(1, len(ising_energy[i, :]) + 1), ising_energy[i, :])
        axs[i].set_title(f'Energy[{i}, :]')
        axs[i].set_ylabel('Energy')
        axs[i].grid(True)

    # Label x-axis on the bottom subplot
    axs[-1].set_xlabel('Index')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()