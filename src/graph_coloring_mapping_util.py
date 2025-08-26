import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import itertools
import numpy as np
import math 
from sympy import *
import re
import multiprocessing
from scipy.sparse import coo_matrix
import numexpr as ne
from sympy.logic import SOPform
from sympy import symbols
import torch
__all__ = [
           "ising_opt",
           "parse_dimacs",
           "create_ising_hamiltonian",
           "visualize_graph",
           "chromatic_numbers",
           "calculate_ising_hamiltonian",
           "delta_energy",
           "generate_all_possible_s_vectors",
           "calculate_quadratic_form",
           "format_spin_config",
           "dict_to_vector",
           "validate_solution",
           "create_constraint_terms_ising_hamiltonian",
           "create_constraint_terms_ising_hamiltonian_optm",
           "logic_conversion_truth_table_graph_coloring",
           "bool_to_ising_airth",
           "parse_vectorized_hamiltonian_expression",
           "expand_expression",
           "generate_vectorized_hamiltonian",
           "generate_vectorized_unique_spin_tuple",
           "calculate_ising_vectorized",
           "delta_energy_vectorized",
           "calculate_ising_hamiltonian_batch",
           "delta_energy_batch",
           "delta_energy_vectorized_batch",
           "calculate_ising_vectorized_batch",
           "calculate_ising_vectorized_parallel",
           "delta_energy_vectorized_batch_parallel",
           "calculate_ising_hamiltonian_batch_parallel",
           "calculate_ising_hamiltonian_batch_parallel_optm",
           "delta_energy_batch_parallel",
           "graph_plot_ising_coloring",
           "graph_plot_vectorized_coloring",
           "valid_invalid_state_space_ising",
           "valid_invalid_state_space_ising_optm",
           "valid_invalid_state_space_vectorized",
           "generate_vectorized_unique_spin_tuple_in_order",
           "calculate_ising_vectorized_parallel_expression_eval",
           "delta_energy_vectorized_batch_parallel_expression_eval",
           "failed_edges_count",
           "failed_edges_count_vectorized",
           "calculate_ising_vectorized_parallel_njit",
           "generate_padded_indices_mask",
           "calculate_ising_vectorized_matrix",
           "delta_energy_vectorized_matrix",
           "calculate_ising_vectorized_matrix_torch",
           "delta_energy_vectorized_matrix_torch",
           "calculate_ising_hamiltonian_matrix",
           "calculate_ising_hamiltonian_matrix_optm",
           "delta_energy_matrix_torch",
           "generate_gc_hamiltonian",
           "write_vecmul_gc",
           "pt_hyperparameter",
           "calculate_ising_vectorized_gc",
           "calculate_delta_ising_vectorized_gc",
           "plot_ising_energy",
           "generate_gc_hamiltonian_rbm",
           "write_vecmul_rbm", 
           "sample_nodes",
           "calculate_ising_vectorized_gc_rbm",
           "calculate_delta_ising_vectorized_gc_optimized",
           "check_double_count_edge",
           "create_ising_hamiltonian_optm",
           "failed_edges_count_sbm",
           "failed_edges_count_sbm_v1",
           "hamiltonian_cim",
           "failed_edges_count_cim"
           ]

ising_opt= {'anna': [0.5, 6], 
            'david': [0.25, 6], 
            'huck': [0.25, 2], 
            'myciel3': [2, 6], 
            'myciel4': [2, 6], 
            'myciel5': [0.5, 2], 
            'myciel6': [0.25, 2], 
            'myciel7': [0.25, 4], 
            'queen11_11': [0.5, 8], 
            'queen13_13': [0.5, 8], 
            'queen5_5': [0.5, 6], 
            'queen6_6': [0.5, 4], 
            'queen7_7': [0.25, 4], 
            'queen8_12': [0.25, 4], 
            'queen8_8': [0.5, 8], 
            'queen9_9': [0.25, 4],
            'cora':[0.25, 32],
            'citeseer':[0.25, 32], 
            'pubmed_graph':[0.25, 32]
}
def parse_dimacs(file_path):
    nodes = set()
    edges = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # try:
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         lines = file.readlines()
    # except UnicodeDecodeError:
    #     # Handle the case where the file is not UTF-8 encoded
    #     with open(file_path, 'r', encoding='latin-1') as file:
    #         lines = file.readlines()
    #         print("Error Lines",lines)
    # Process the lines as needed
            
    for line in lines:
        if line.startswith('c') or line.startswith('p'):
            continue
        if line.startswith('e'):
            parts = line.split()
            u = int(parts[1])
            v = int(parts[2])
            edges.append((u, v))
            nodes.add(u)
            nodes.add(v)
    
    return nodes, edges


def create_ising_hamiltonian(nodes, edges, num_colors,device, A=1 ):
    n = max(nodes)  # Number of nodes
    num_nodes = len(nodes)
    
    # Initialize the coefficients for the quadratic terms
    Q = {}
    unique_spins = []
    ### Ordered list of nodes
    sorted_nodes = np.sort(list(nodes))
    for node in sorted_nodes:
        for j in range(1, num_colors+1):
            unique_spins.append((node, j))

    Hamiltonian_matrix = torch.zeros((len(unique_spins),len(unique_spins)), device = device)
    # Add the term to ensure each node gets exactly one color
    for v in nodes:
        for i in range(1, num_colors+1):
            Q[(v, i, v, i)] = Q.get((v, i, v, i), 0) - 2*A
            for j in range(1, num_colors+1):
                if i != j:
                    Q[(v, i, v, j)] = Q.get((v, i, v, j), 0) + 2 * A
    
    # Add the term to ensure adjacent nodes do not get the same color
    for (u, v) in edges:
        for i in range(1, num_colors+1):
            Q[(u, i, v, i)] = Q.get((u, i, v, i), 0) + A
            Q[(v, i, u, i)] = Q.get((v, i, u, i), 0) + A
            # if (u, i) not in unique_spins:
            #     unique_spins.append((u, i))
            # if (v, i) not in unique_spins:
            #     unique_spins.append((v, i))
    for  (v1, i1, v2, i2) in Q:
        Hamiltonian_matrix[int((i1-1)+(v1-1)*num_colors), int((i2-1)+(v2-1)*num_colors)] = Q[(v1, i1, v2, i2)]
    return Q, unique_spins, Hamiltonian_matrix

def create_constraint_terms_ising_hamiltonian(nodes, edges, num_colors, A=1):
    n = max(nodes)  # Number of nodes
    num_nodes = len(nodes)
    
    # Initialize the coefficients for the quadratic terms
    Q = {}
    
    # Add the term to ensure each node gets exactly one color
    for v in nodes:
        for i in range(1, num_colors+1):
            Q[(v, i, v, i)] = Q.get((v, i, v, i), 0) - 2*A
            for j in range(1, num_colors+1):
                if i != j:
                    Q[(v, i, v, j)] = Q.get((v, i, v, j), 0) + 2 * A
    return Q

def create_constraint_terms_ising_hamiltonian_optm(nodes, edges, num_colors, B):
    n = max(nodes)  # Number of nodes
    num_nodes = len(nodes)
    
    # Initialize the coefficients for the quadratic terms
    Q = {}
    
    # Add the term to ensure each node gets exactly one color
    for v in nodes:
        for i in range(1, num_colors+1):
            Q[(v, i, v, i)] = Q.get((v, i, v, i), 0) - B
            for j in range(1, num_colors+1):
                if i != j:
                    Q[(v, i, v, j)] = Q.get((v, i, v, j), 0) + B
    return Q
def visualize_graph(dataset, nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)  # Layout for visualization
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(f"Graph Coloring Problem ({dataset})")
    plt.show()

def generate_all_possible_s_vectors(length):
    # Generate all possible binary vectors of given length
    return list(itertools.product([0, 1], repeat=length))

def calculate_quadratic_form(s, A):
    s = np.array(s)
    return np.dot(np.dot(s, A), s.T)

def format_spin_config(s, num_colors):
    # Format the binary vector in the specified (node, color) format
    spin_config = {}
    num_nodes = len(s) // num_colors
    for node in range(num_nodes):
        for color in range(1, num_colors + 1):
            spin_config[(node, color)] = s[node * num_colors + (color - 1)]
    return spin_config


chromatic_numbers = {
    # COLOR graphs
    'jean.col': 10,
    'anna.col': 11,
    'huck.col': 11,
    'david.col': 11,
    'homer.col': 13,
    'myciel5.col': 6,
    'myciel6.col': 7,
    'queen5_5.col': 5,
    'queen6_6.col': 7,
    'queen7_7.col': 7,
    'queen8_8.col': 9,
    'queen9_9.col': 10,
    'queen8_12.col': 12,
    'queen11_11.col': 11,
    'queen13_13.col': 13,
    # Citations graphs
    'cora.cites': 5,
    'citeseer.cites': 6,
    'pubmed.cites': 8,
    'test_example.col':2,
    'flat1000_50_0.col.b' : 50, 
    'flat1000_60_0.col.b' : 60, 
    'flat1000_76_0.col.b' : 76, 
    'flat300_20_0.col.b' : 20, 
    'flat300_26_0.col.b' : 26, 
    'flat300_28_0.col.b' : 28, 
    'fpsol2.i.1.col': 65,
    'fpsol2.i.2.col': 30,
    'fpsol2.i.3.col': 30, 
    'inithx.i.1.col': 54,
    'inithx.i.2.col': 31,
    'inithx.i.3.col': 31, 
    'le450_15a.col' : 15,
    'le450_15b.col' : 15,
    'le450_15c.col' : 15, 
    'le450_15d.col' : 15, 
    'le450_25a.col' : 25, 
    'le450_25b.col' : 25, 
    'le450_25c.col' : 25, 
    'le450_25d.col' : 25, 
    'le450_5a.col'  :  5, 
    'le450_5b.col'  : 5, 
    'le450_5c.col' : 5, 
    'le450_5d.col' : 5, 
    'mulsol.i.1.col' : 49, 
    'mulsol.i.2.col' : 31, 
    'mulsol.i.3.col': 31, 
    'mulsol.i.4.col' : 31, 
    'mulsol.i.5.col' : 31, 
    'zeroin.i.1.col' : 49, 
    'zeroin.i.2.col' : 30, 
    'zeroin.i.3.col' : 30, 
    'anna.col' : 11, 
    'david.col' : 11, 
    'homer.col' : 13, 
    'huck.col' : 11, 
    'jean.col' : 10, 
    'games120.col' : 9, 
    'miles1000.col' : 42, 
    'miles1500.col' : 73, 
    'miles250.col' : 8, 
    'miles500.col' : 20, 
    'miles750.col' : 31, 
    'queen11_11.col' : 11, 
    'queen13_13.col' : 13, 
    'queen5_5.col' : 5, 
    'queen6_6.col' : 7, 
    'queen7_7.col' : 7, 
    'queen8_12.col' : 12, 
    'queen8_8.col' : 9, 
    'queen9_9.col' : 10, 
    'myciel3.col' : 4, 
    'myciel4.col' : 5, 
    'myciel5.col' : 6, 
    'myciel6.col' : 7, 
    'myciel7.col' : 8,
    'cora.col':5,
    'citeseer.col':6,
    'pubmed_graph.col':8
}
def calculate_ising_hamiltonian(nodes, Q, spin_config):
    H = 0 + 2*len(nodes)

    # Linear terms
    for (v, i) in spin_config:
        if (v, i) in Q:
            H += Q[(v, i)] * spin_config[(v, i)]
    # Quadratic terms
    for (v1, i1, v2, i2) in Q:
        if (v1, i1) in spin_config and (v2, i2) in spin_config:
            H += Q[(v1, i1, v2, i2)] * spin_config[(v1, i1)] * spin_config[(v2, i2)]

    return H 
def pt_hyperparameter():
    ## batch size ## pt interval
    pt_dict = {
        'anna.col':[100, 15], 
        'david.col':[100, 15],
        'games120.col':[100, 15],
        'huck.col':[100, 15],
        'myciel3.col':[100, 15],
        'myciel4.col':[100, 15],
        'myciel5.col':[100, 15],
        'myciel6.col':[100, 15],
        'myciel7.col':[100, 15],
        'queen11_11.col':[100, 15],
        'queen13_13.col':[100,15],
        'queen5_5.col':[100, 15],
        'queen6_6.col':[100, 15],
        'queen7_7.col':[100, 15],
        'queen8_12.col':[100, 15],
        'queen8_8.col':[100, 15], 
        'queen9_9.col':[100, 15],
        'cora.col':[100, 15],
        'citeseer.col':[100, 15]

    }
    return pt_dict
def calculate_ising_hamiltonian_batch(nodes, Q, state, state_space):
    energy = np.zeros(state.shape[0])
    for batch_indx in range (state.shape[0]) :
        spin_config = {}
        spin_config = {(v[0], v[1]): state[batch_indx,indx_state] for  indx_state, v in enumerate(state_space)} 
        H = 0 + 2*len(nodes)
        # Linear terms
        for (v, i) in spin_config:
            if (v, i) in Q:
                H += Q[(v, i)] * spin_config[(v, i)]
        # Quadratic terms
        for (v1, i1, v2, i2) in Q:
            if (v1, i1) in spin_config and (v2, i2) in spin_config:
                H += Q[(v1, i1, v2, i2)] * spin_config[(v1, i1)] * spin_config[(v2, i2)]
        energy[batch_indx] = H

    return energy 


def calculate_ising_hamiltonian_batch_parallel(nodes, Q,state, state_space):
    numpy_state=np.array(state_space)
    num_colors = np.max(numpy_state[:, 1])
    num_nodes = np.max(numpy_state[:, 0])
    state_config = np.zeros((state.shape[0],(num_colors+1)*(num_nodes+1)))
    for  indx_state, v in enumerate(state_space):
        state_config[:, int(v[1]+v[0]*(num_colors+1))] = state[:,indx_state]

    hamiltoninan = []
    spin_mult = []
    for  (v1, i1, v2, i2) in Q:
        # Calculate the product of spins for this interaction term
        hamiltoninan.append(Q[(v1, i1, v2, i2)])
        interaction_product = np.ones(state.shape[0])
        interaction_product *= state_config[:, int(i1+v1*(num_colors+1))]
        interaction_product *= state_config[:, int(i2+v2*(num_colors+1))]

        spin_mult.append(interaction_product)
    hamil_arr = np.array(hamiltoninan)
    spin_mult_arr = np.array(spin_mult)

    energy = np.sum(hamil_arr[:, np.newaxis] * spin_mult_arr, axis=0)+ 2*len(nodes)

    return energy              

def calculate_ising_hamiltonian_batch_parallel_optm(nodes, Q,state, state_space, B):
    numpy_state=np.array(state_space)
    num_colors = np.max(numpy_state[:, 1])
    num_nodes = np.max(numpy_state[:, 0])
    state_config = np.zeros((state.shape[0],(num_colors+1)*(num_nodes+1)))
    for  indx_state, v in enumerate(state_space):
        state_config[:, int(v[1]+v[0]*(num_colors+1))] = state[:,indx_state]

    hamiltoninan = []
    spin_mult = []
    for  (v1, i1, v2, i2) in Q:
        # Calculate the product of spins for this interaction term
        hamiltoninan.append(Q[(v1, i1, v2, i2)])
        interaction_product = np.ones(state.shape[0])
        interaction_product *= state_config[:, int(i1+v1*(num_colors+1))]
        interaction_product *= state_config[:, int(i2+v2*(num_colors+1))]

        spin_mult.append(interaction_product)
    hamil_arr = np.array(hamiltoninan)
    spin_mult_arr = np.array(spin_mult)

    energy = np.sum(hamil_arr[:, np.newaxis] * spin_mult_arr, axis=0)+ B*len(nodes)

    return energy   

def calculate_ising_hamiltonian_matrix(Hamiltonian_matrix, state, nodes):
    # energy = 2*len(nodes) + torch.sum(state*(Hamiltonian_matrix@state.T).T, axis = 1, keepdims = True)
    if(Hamiltonian_matrix.shape[0]>50000):
        energy = 2*len(nodes) + torch.sum((state.to(torch.float16))*(Hamiltonian_matrix@(state.to(torch.float16)).T).T, axis = 1, keepdims = True)
    else:
        energy = 2*len(nodes) + torch.sum(state*(Hamiltonian_matrix@state.T).T, axis = 1, keepdims = True)

    return energy.squeeze()   

def calculate_ising_hamiltonian_matrix_optm(Hamiltonian_matrix, state, nodes, B):
    if(Hamiltonian_matrix.shape[0]>50000):
        energy = B*len(nodes) + torch.sum((state.to(torch.float16))*(Hamiltonian_matrix@(state.to(torch.float16)).T).T, axis = 1, keepdims = True)
    else:
        energy = B*len(nodes) + torch.sum(state*(Hamiltonian_matrix@state.T).T, axis = 1, keepdims = True)
    return energy.squeeze()  

# This term validates whether the sampled state is part of solution space 
def validate_solution(Q, spin_config, const):
    H = 0
    for (v, i) in spin_config:
        if (v, i) in Q:
            H += Q[(v, i)] * spin_config[(v, i)]

    # Quadratic terms
    for (v1, i1, v2, i2) in Q:
        if (v1, i1) in spin_config and (v2, i2) in spin_config:
            H += Q[(v1, i1, v2, i2)] * spin_config[(v1, i1)] * spin_config[(v2, i2)]
    H_net = H + const
    if H_net == 0:
        return True
    else:
        return False

def delta_energy(nodes, Q, spin_config, node, color):
    # Calculate current energy
    current_energy = calculate_ising_hamiltonian(nodes, Q, spin_config)
    
    # Flip the spin
    spin_config[(node, color)] =  1 - spin_config[(node, color)]
    
    # Calculate new energy
    new_energy = calculate_ising_hamiltonian(nodes, Q, spin_config)
    
    # Flip the spin back
    spin_config[(node, color)] =  1 - spin_config[(node, color)]
    
    # Calculate delta energy
    delta_E = new_energy - current_energy
    return delta_E



def delta_energy_batch(nodes, Q, state, state_space, node, color):

    delta_E = np.zeros(state.shape[0])

    flip_index = [i for i, t in enumerate(state_space) if t == (node, color)]

        # Calculate current energy

    current_energy = calculate_ising_hamiltonian_batch(nodes, Q, state, state_space)
    # Flip the spin

    state[:,flip_index] =  1 - state[:,flip_index]
    
    # Calculate new energy
    new_energy = calculate_ising_hamiltonian_batch(nodes, Q, state, state_space)
    
    # Flip the spin back
    state[:,flip_index] =  1 - state[:,flip_index]
    
    # Calculate delta energy
    delta_E = new_energy - current_energy

    return delta_E

def delta_energy_batch_parallel(nodes, Q, state, state_space, node, color):

    delta_E = np.zeros(state.shape[0])

    flip_index = [i for i, t in enumerate(state_space) if t == (node, color)]

        # Calculate current energy

    current_energy = calculate_ising_hamiltonian_batch_parallel(nodes, Q, state, state_space)
    # Flip the spin

    state[:,flip_index] =  1 - state[:,flip_index]
    
    # Calculate new energy
    new_energy = calculate_ising_hamiltonian_batch_parallel(nodes, Q, state, state_space)
    
    # Flip the spin back
    state[:,flip_index] =  1 - state[:,flip_index]
    
    # Calculate delta energy
    delta_E = new_energy - current_energy

    return delta_E
        

def delta_energy_matrix_torch(nodes, Hamiltonian_matrix, state, num_colors, node, color):

    # delta_E = torch.zeros(state.shape[0])

    flip_index = int((color-1)+(node-1)*num_colors) 
        # Calculate current energy

    current_energy = calculate_ising_hamiltonian_matrix(Hamiltonian_matrix, state, nodes)
    # Flip the spin

    state[:,flip_index] =  1 - state[:,flip_index]
    
    # Calculate new energy
    new_energy = calculate_ising_hamiltonian_matrix(Hamiltonian_matrix, state, nodes)
    # Flip the spin back
    state[:,flip_index] =  1 - state[:,flip_index]
    # Calculate delta energy
    delta_E = new_energy - current_energy
    return delta_E


def dict_to_vector(spin_dict, nodes, num_colors):
    n = len(nodes)
    spin_vector = np.zeros(n * num_colors, dtype=int)
    node_list = list(nodes)  # Convert set to list to maintain order
    for (node, color), value in spin_dict.items():
        index = node_list.index(node) * num_colors + (color - 1)
        spin_vector[index] = value
    return spin_vector

########### vectorized mapping #################
def logic_conversion_truth_table_graph_coloring(u, v, num_colors):
    logic_pow = int(math.floor(math.log2(num_colors-1) +1))
    variables_string = "" 
    variable_list = []
    truth_list =[]
    for i in range(logic_pow):
        variables_string = f"{variables_string} S{u}{int(logic_pow-1-i)}"
        variable_list.append(f"S{u}{int(logic_pow-1-i)}")
    for i in range(logic_pow):
        variables_string = f"{variables_string} S{v}{int(logic_pow-1-i)}" 
        variable_list.append(f"S{v}{int(logic_pow-1-i)}")  
    truth_table = np.zeros(int(math.pow(2, 2*logic_pow)), dtype=int)
    for j in range(num_colors):
        truth_table[int(j+math.pow(2, logic_pow)*j)] = 1
        truth_list.append(int(j+math.pow(2, logic_pow)*j))
    ### "any digit + col =1"
    num_arr = np.arange(0,int(math.pow(2, logic_pow)))
    for k in range(num_colors, int(math.pow(2, logic_pow))):
        truth_table[(math.pow(2, logic_pow)*num_arr+k).astype(int)] =1
        truth_table[(k*math.pow(2, logic_pow)+num_arr).astype(int)] = 1
        
        truth_list.append(list((math.pow(2, logic_pow)*num_arr+k).astype(int)))
        truth_list.append(list((k*math.pow(2, logic_pow)+num_arr).astype(int)))

    #### Edit ###########
    return variables_string, truth_table, truth_list, variable_list

def bool_to_ising_airth(bool_expression):
    ising_expression = bool_expression.replace("&", "*")
    ising_expression = ising_expression.replace("|", "+")
    ising_expression = re.sub(r'~(\w+)', r'(1 - \1)', ising_expression)
    # print(ising_expression)
    return ising_expression

def expand_expression(expression):
    result = expand(expression)
    return str(result)

def split_terms(expression):
    # Add space around + and - for splitting

    pattern = r'(?<!\S)[+\-]?\s*\w[\w\*\d]*'
    
    # Find all terms in the expression
    terms = re.findall(pattern, expression)
        
    # Initialize lists for positive and negative terms
    positive_terms = []
    negative_terms = []
    # Iterate over the terms and classify them
    for term in terms:
        if term.startswith('-'):
            negative_terms.append(term)
        elif term.startswith('+'):
            positive_terms.append(term.lstrip('+'))
        else:
            positive_terms.append(term)
    
    return positive_terms, negative_terms

def parse_vectorized_hamiltonian_expression(expression, hamiltonian, A):
    """
    Parse a string expression of an Ising model to generate a Hamiltonian dictionary.
    
    Parameters:
        expression (str): The Hamiltonian expression in the form of a string, such as 'S00*S10*S11*S01 + S10*S11 + S10'.
    
    Returns:
        dict: A dictionary where keys are tuples of spin tuples, and values are the interaction strengths.
    """
    # Normalize the expression: remove spaces and handle signs and create 2 separate arrays 
    positive_terms, negative_terms = split_terms(expression)
    for term in positive_terms:
        spins = re.findall(r'S(\d+)', term)
        spin_tuples = tuple(sorted((int(s[:-1]), int(s[-1])) for s in spins))

        match = re.search(r'^[+-]?\s*\d+', term) 
        if match:
            number_str = match.group().replace(' ', '')
            number = int(number_str)
            number = int(match.group().strip())
            if spin_tuples in hamiltonian:
                hamiltonian[spin_tuples] += A*number
            else:
                hamiltonian[spin_tuples] = A*number            
        else:
            if spin_tuples in hamiltonian:
                hamiltonian[spin_tuples] += A  
            else:
                hamiltonian[spin_tuples] = A
    

    for term in negative_terms:
        spins = re.findall(r'S(\d+)', term)
        match = re.search(r'^[+-]?\s*\d+', term)
        spin_tuples = tuple(sorted((int(s[:-1]), int(s[-1])) for s in spins))
        if match:
            number_str = match.group().replace(' ', '')
            number = int(number_str)
            if spin_tuples in hamiltonian:
                hamiltonian[spin_tuples] += A*number
            else:
                hamiltonian[spin_tuples] = A*number            
        else:
            if spin_tuples in hamiltonian:
                hamiltonian[spin_tuples] += -A  
            else:
                hamiltonian[spin_tuples] = -A

    return hamiltonian
def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list

def generate_vectorized_hamiltonian(nodes, edges, num_colors, A):
    n = max(nodes)  # Number of nodes
    num_nodes = len(nodes)
    Q = {}
    hamiltonian_expression = f"0 + "
    gate_expression = ""
    or_gate_count = 0
    for (u, v) in edges:
        variable, truth_table, truth_list, variable_list = logic_conversion_truth_table_graph_coloring(u, v, num_colors)
        variable_symbols_list = symbols(' '.join(variable_list))
        flattened_truth_list =flatten_list(truth_list)
        flattened_truth_list = [int(num) for num in flattened_truth_list]
        # print("SOP form",SOPform(variable_list, list(flattened_truth_list)))
        variables_bool = symbols(variable)
        
        # print(list(variable))
        # print(SOPform(list(variable), truth_list))
        # print("Variable", variable)
        # print("Truth Table", truth_table)
        # minimized_expression = boolean_minimization(truth_table, variables_bool)
        minimized_expression = SOPform(variable_list, list(flattened_truth_list))
        gate_expression += f"{minimized_expression} | "
        or_gate_count +=str(minimized_expression).count('|')
        ising_expression = bool_to_ising_airth(str(minimized_expression))
        # print(f"2*({ising_expression})-1")
        # expand_expr = expand_expression(f"2*({ising_expression})-1")
        expand_expr = expand_expression(ising_expression)
        hamiltonian_expression += f"{expand_expr} + "
        # print("Expanded Expression", expand_expr)
        Q = parse_vectorized_hamiltonian_expression(expand_expr, Q, A)
    hamiltonian_expression += f" 0"
    return Q, hamiltonian_expression, gate_expression, or_gate_count

def generate_vectorized_unique_spin_tuple(Q):
    unique_spins = []
    for k, v in Q.items():
        for k1 in k:
            if k1 not in unique_spins:
                unique_spins.append(k1)
    print(unique_spins)
    return unique_spins

def generate_vectorized_unique_spin_tuple_in_order(nodes, num_colors):
    unique_spins = []
    num_nodes = max(nodes)
    # print(num_nodes)
    logic_pow = int(math.floor(math.log2(num_colors-1) +1))
    for node in range(1, num_nodes + 1): #node in nodes: #
        for j in range(logic_pow):
            unique_spins.append((node,int(logic_pow-1-j)))
    return unique_spins

### Write a spin variable function in generate_vectorized function to account the spins used
### Same goes with the normal case

def calculate_ising_vectorized(hamiltonian, spin_states):
    """
    Calculate the energy of an Ising model given a Hamiltonian and spin states.
    
    Parameters:
        hamiltonian (dict): A dictionary where keys are tuples of tuples, each key representing
                            interacting spins and their coefficient as the value.
        spin_states (dict): A dictionary where each key is a tuple representing the spin's
                            position and the value is the spin's state (+1 or -1).
    
    Returns:
        float: The total energy of the system.
    """
    energy = 0
    for spins, coefficient in hamiltonian.items():
        # Calculate the product of spins for this interaction term
        interaction_product = coefficient
        for spin in spins:
            interaction_product *= spin_states[spin]
        energy += interaction_product
    return energy

def calculate_ising_vectorized_batch(hamiltonian, state, state_space):
    """
    Calculate the energy of an Ising model given a Hamiltonian and spin states.
    
    Parameters:
        hamiltonian (dict): A dictionary where keys are tuples of tuples, each key representing
                            interacting spins and their coefficient as the value.
        spin_states (dict): A dictionary where each key is a tuple representing the spin's
                            position and the value is the spin's state (+1 or -1).
    
    Returns:
        float: The total energy of the system.
    """
    energy = np.zeros(state.shape[0])
    for batch_indx in range (state.shape[0]) :
        state_config = {}
        state_config = {(v[0], v[1]): state[batch_indx,indx_state] for  indx_state, v in enumerate(state_space)}

        energy_int = 0
        for spins, coefficient in hamiltonian.items():
            # Calculate the product of spins for this interaction term
            interaction_product = coefficient
            for spin in spins:
                interaction_product *= state_config[spin]
            energy_int += interaction_product
        energy[batch_indx] = energy_int
    return energy


def calculate_ising_vectorized_parallel(hamiltonian,state, state_space):
    numpy_state=np.array(state_space)
    num_colors = np.max(numpy_state[:, 1])
    num_nodes = np.max(numpy_state[:, 0])
    # state_config = np.zeros((state.shape[0],(num_colors+1)*(num_nodes+1)))
    # for  indx_state, v in enumerate(state_space):
    #     state_config[:, int(v[1]+v[0]*(num_colors+1))] = state[:,indx_state]

    hamiltoninan = []
    spin_mult = []
    for spins, coefficient in hamiltonian.items():
        # Calculate the product of spins for this interaction term
        hamiltoninan.append(coefficient)
        interaction_product = np.ones(state.shape[0])
        for spin in spins:
            # interaction_product *= state_config[:, int(spin[1]+spin[0]*(num_colors+1))]
            interaction_product *= state[:, int((num_colors-spin[1])+(spin[0]-1)*(num_colors+1))] ## check the state space and state order

        spin_mult.append(interaction_product)
        # print(np.array(hamiltoninan).shape)
        # print(np.array(spin_mult).shape)
    hamil_arr = np.array(hamiltoninan)
    spin_mult_arr = np.array(spin_mult)

    energy = np.sum(hamil_arr[:, np.newaxis] * spin_mult_arr, axis=0)

    return energy

def generate_padded_indices_mask(hamiltonian, state_space):
    numpy_state=np.array(state_space)
    num_colors = np.max(numpy_state[:, 1])
    spins = [[int((num_colors-arg[1])+(arg[0]-1)*(num_colors+1)) for arg in k] for k in hamiltonian.keys()]
    coefficients = np.array(list(hamiltonian.values()), dtype=np.float32)
    max_len = max(len(sublist) for sublist in spins)
    padded_indices = np.array([sublist + [0] * (max_len - len(sublist)) for sublist in spins]) ## defined globally
    mask = np.array([[True] * len(sublist) + [False] * (max_len - len(sublist)) for sublist in spins]) ## defined globally
    return padded_indices, mask, coefficients, num_colors

def calculate_ising_vectorized_matrix(padded_indices, mask, state, coefficients):
    gathered_elements = state[:, padded_indices]

    gathered_elements[:, ~mask] = 1
    out = np.prod(gathered_elements, axis=2)
    hamil_arr = coefficients
    spin_mult_arr = np.array(out)

    energy = np.sum(hamil_arr[ np.newaxis, :] * spin_mult_arr, axis=1)
    return energy

def calculate_ising_vectorized_matrix_torch(padded_indices, mask, state, coefficients):

    gathered_elements = state[:, padded_indices]

    gathered_elements[:, ~mask] = 1
    out = torch.prod(gathered_elements, dim=2)
    hamil_arr = coefficients
    spin_mult_arr = out
    energy = torch.sum(hamil_arr.unsqueeze(0) * spin_mult_arr, dim=1)
    # stream = torch.cuda.Stream()
    # # Perform the operation within the stream
    # with torch.cuda.stream(stream):
    #     energy = torch.sum(hamil_arr.unsqueeze(0) * spin_mult_arr, dim=1)

    # # Wait for the stream to complete
    # stream.synchronize()
    return energy

def delta_energy_vectorized_matrix(col_width, padded_indices, mask, state, coefficients, node, color):
    delta_E = np.zeros(state.shape[0])
    flip_index = int((col_width-color)+(node-1)*(col_width+1))
    # Calculate current energy
    current_energy =  calculate_ising_vectorized_matrix(padded_indices, mask, state, coefficients)
    # Flip the spin
    state[:,flip_index] =  1 -  state[:,flip_index]
    # Calculate new energy
    new_energy =calculate_ising_vectorized_matrix(padded_indices, mask, state, coefficients)
    # Flip the spin back
    state[:,flip_index] =  1 -  state[:,flip_index]
    # Calculate delta energy
    delta_E = new_energy - current_energy
    return delta_E

def delta_energy_vectorized_matrix_torch(col_width, padded_indices, mask, state, coefficients, node, color):
    delta_E = torch.zeros(state.shape[0])
    flip_index = int((col_width-color)+(node-1)*(col_width+1))
    # Calculate current energy
    current_energy =  calculate_ising_vectorized_matrix_torch(padded_indices, mask, state, coefficients)
    # Flip the spin
    state[:,flip_index] =  1 -  state[:,flip_index]
    # Calculate new energy
    new_energy = calculate_ising_vectorized_matrix_torch(padded_indices, mask, state, coefficients)
    # Flip the spin back
    state[:,flip_index] =  1 -  state[:,flip_index]
    # Calculate delta energy
    delta_E = new_energy - current_energy
    return delta_E

def calculate_ising_vectorized_parallel_njit(hamiltonian,state, state_space):
    numpy_state=np.array(state_space)
    num_colors = np.max(numpy_state[:, 1])
    num_nodes = np.max(numpy_state[:, 0])

    spins = [[int((num_colors-arg[1])+(arg[0]-1)*(num_colors+1)) for arg in k] for k in hamiltonian.keys()]
    coefficients = np.array(list(hamiltonian.values()), dtype=np.float32)

    # indices = [[1, 0, 3, 2], [1, 0], [1, 3]]
    # st = np.array([2, 3, 4, 5])  # Example st array, replace with actual values

    # Pad sublists with ones to make them the same length
    max_len = max(len(sublist) for sublist in spins)
    padded_indices = np.array([sublist + [0] * (max_len - len(sublist)) for sublist in spins]) ## defined globally

    # Use advanced indexing to gather elements
    gathered_elements = state[:, padded_indices]

    # Set padding positions to 1
    mask = np.array([[True] * len(sublist) + [False] * (max_len - len(sublist)) for sublist in spins]) ## defined globally
    gathered_elements[:, ~mask] = 1
    print(gathered_elements.shape)
    # Compute the product along the last axis
    out = np.prod(gathered_elements, axis=2)
    hamil_arr = coefficients
    spin_mult_arr = np.array(out)

    energy = np.sum(hamil_arr[ np.newaxis, :] * spin_mult_arr, axis=1)

    return energy





def calculate_ising_vectorized_parallel_expression_eval(hamiltonian_expression, state, nodes, num_colors):
    #### Using sorted state space in this case 
    nodes_num = len(nodes)
    logic_pow = int(math.floor(math.log2(num_colors-1) +1))
    state_indx = 0
    hamil_eval = hamiltonian_expression
    for i in range(1, nodes_num + 1):
        for j in range(logic_pow):

            hamil_eval= hamil_eval.replace(f"*S{i}{int(logic_pow-1-j)}*", f"*state[:,{int(state_indx)}]*")
            hamil_eval= hamil_eval.replace(f"*S{i}{int(logic_pow-1-j)} ", f"*state[:,{int(state_indx)}] ")
            hamil_eval= hamil_eval.replace(f" S{i}{int(logic_pow-1-j)}*", f" state[:,{int(state_indx)}]*")
            hamil_eval= hamil_eval.replace(f" S{i}{int(logic_pow-1-j)} ", f" state[:,{int(state_indx)}] ")

            state_indx +=1
    hamil_eval= hamil_eval.replace(f"+ ", f"+")
    hamil_eval= hamil_eval.replace(f"- ", f"-")

    parts = re.findall(r'\S+', hamil_eval)

    # Split the parts into 7 roughly equal-sized groups
    # num_parts_per_group = state_indx // logic_pow
    num_parts_per_group = logic_pow
    grouped_parts = [parts[i:i+num_parts_per_group] for i in range(0, len(parts), num_parts_per_group)]
    sub_expressions = [' '.join(group) for group in grouped_parts]

    # Evaluate each sub-expression separately
    results = [eval(expr, {'state': state}) for expr in sub_expressions]
    final_result = sum(results)
    return np.array(final_result)



def delta_energy_vectorized(hamiltonian, spin_states, node, color):
    # Calculate current energy
    current_energy = calculate_ising_vectorized(hamiltonian, spin_states)
    
    # Flip the spin
    spin_states[(node, color)] =  1 - spin_states[(node, color)]
    
    # Calculate new energy
    new_energy = calculate_ising_vectorized(hamiltonian, spin_states)
    
    # Flip the spin back
    spin_states[(node, color)] =  1 - spin_states[(node, color)]
    
    # Calculate delta energy
    delta_E = new_energy - current_energy
    return delta_E


def delta_energy_vectorized_batch(hamiltonian, state, state_space, node, color):
    delta_E = np.zeros(state.shape[0])

    flip_index = [i for i, t in enumerate(state_space) if t == (node, color)]

    # Calculate current energy
    current_energy =  calculate_ising_vectorized_batch(hamiltonian, state, state_space)
    
    # Flip the spin
    state[:,flip_index] =  1 -  state[:,flip_index]
    
    # Calculate new energy
    new_energy = calculate_ising_vectorized_batch(hamiltonian, state, state_space)

    # Flip the spin back
    state[:,flip_index] =  1 -  state[:,flip_index]
    
    # Calculate delta energy
    delta_E = new_energy - current_energy
    return delta_E


def delta_energy_vectorized_batch_parallel(hamiltonian, state, state_space, node, color):
    delta_E = np.zeros(state.shape[0])

    flip_index = [i for i, t in enumerate(state_space) if t == (node, color)]
    # print(flip_index)
    # numpy_state=np.array(state_space)
    # num_colors = np.max(numpy_state[:, 1])
    # print(int((num_colors-color)+(node-1)*(num_colors+1)))

    # Calculate current energy
    current_energy =  calculate_ising_vectorized_parallel(hamiltonian, state, state_space)
    
    # Flip the spin
    state[:,flip_index] =  1 -  state[:,flip_index]
    
    # Calculate new energy
    new_energy = calculate_ising_vectorized_parallel(hamiltonian, state, state_space)

    # Flip the spin back
    state[:,flip_index] =  1 -  state[:,flip_index]
    
    # Calculate delta energy
    delta_E = new_energy - current_energy
    return delta_E

def delta_energy_vectorized_batch_parallel_expression_eval(hamiltonian_expression, state, state_space, node, color, nodes, num_colors):
    delta_E = np.zeros(state.shape[0])

    flip_index = [i for i, t in enumerate(state_space) if t == (node, color)]

    # Calculate current energy
    current_energy =  calculate_ising_vectorized_parallel_expression_eval(hamiltonian_expression, state, nodes, num_colors)
    
    # Flip the spin
    state[:,flip_index] =  1 -  state[:,flip_index]
    
    # Calculate new energy
    new_energy = calculate_ising_vectorized_parallel_expression_eval(hamiltonian_expression, state, nodes, num_colors)

    # Flip the spin back
    state[:,flip_index] =  1 -  state[:,flip_index]
    
    # Calculate delta energy
    delta_E = new_energy - current_energy
    return delta_E

def graph_plot_ising_coloring(state, state_space, ising_energy, num_choices, nodes, edges, colors, prob_name):
    """
    Randomly pick num_choices graphs out of all graphs 
    Zero energy implies valid graph
    """
    valid_graph = np.where(ising_energy == 0)
    if len(valid_graph[0]) < num_choices:
        num_choices = len(valid_graph[0])
    random_indices = np.random.choice(len(valid_graph[0]), size=num_choices, replace=False)
    color_array = np.zeros((num_choices, int(len(nodes)+1))) ## state space 1 - 9
    valid_states_array = np.zeros((num_choices, state.shape[2]))
    valid_states_array = state[valid_graph[0][random_indices], valid_graph[1][random_indices], :]


    for i in range(num_choices):
        for indx, j in enumerate(valid_states_array[i,:]):
            if (j== 1):
                color_array[i, state_space[indx][0]] = state_space[indx][1]


    color_correctness =  np.ones(num_choices, dtype=bool)
    for i in range(num_choices):
        for (u, v) in edges:
            if color_array[i, u]==color_array[i, v]:
                color_correctness[i] = False

        if(color_correctness[i] == True):
            G = nx.Graph()

            # Add edges to the graph
            G.add_edges_from(edges)
            # Create a layout for our nodes 
            layout_d = nx.spring_layout(G)
            layout = dict(sorted(layout_d.items()))
            # Draw the nodes
            nx.draw_networkx_nodes(G, layout,nodelist=np.arange(1,int(len(color_array[i,1:])+1),1), node_color=color_array[i,1:],cmap = plt.cm.plasma, node_size=500)

            # Draw the edges
            nx.draw_networkx_edges(G, layout, width=1.0, alpha=0.5)

            # Draw node labels
            nx.draw_networkx_labels(G, layout)
            prob = os.path.splitext(prob_name)[0]

            # Show the plot
            plt.title(f'Graph Coloring Ising Problem:{prob} \n Nodes:{len(nodes)} Colors:{colors}')
            plt.axis('off')
            plt.savefig(f"../Data/ising_graph_coloring/graph_coloring_{prob}_{i}.png")
            # plt.show()
            # plt.close()

    return color_correctness

def graph_plot_vectorized_coloring(state, state_space, ising_energy, num_choices, nodes, edges, colors, prob_name):
    """
    Randomly pick num_choices graphs out of all graphs 
    Zero energy implies valid graph
    """
    valid_graph = np.where(ising_energy == 0)
    if len(valid_graph[0]) < num_choices:
        num_choices = len(valid_graph[0])
    random_indices = np.random.choice(len(valid_graph[0]), size=num_choices, replace=False)
    color_array = np.zeros((num_choices, int(max(nodes)+1))) ## state space 1 - 9
    valid_states_array = np.zeros((num_choices, state.shape[2]))
    valid_states_array = state[valid_graph[0][random_indices], valid_graph[1][random_indices], :]
    log_color = int(math.log2(colors-1) + 1)
    for batch_indx in range (valid_states_array.shape[0]) :
        state_config = {}
        state_config = {(v[0], v[1]): valid_states_array[batch_indx,indx_state] for  indx_state, v in enumerate(state_space)}
        for node in nodes:
            for col_indx in range(log_color):
                color_array[batch_indx, int(node)] += state_config[(int(node), col_indx)]*int(math.pow(2, col_indx))
        # print(color_array)

    color_correctness =  np.ones(num_choices, dtype=bool)
    for i in range(num_choices):
        for (u, v) in edges:
            if color_array[i, u]==color_array[i, v]:
                print(f"Failed at (_{v}__{u}_edge" )
                color_correctness[i] = False

        if(color_correctness[i] == True):
            G = nx.Graph()

            # Add edges to the graph
            G.add_edges_from(edges)
            # Create a layout for our nodes 
            layout_d = nx.spring_layout(G)
            layout = dict(sorted(layout_d.items()))
            # Draw the nodes
            nx.draw_networkx_nodes(G, layout,nodelist=np.arange(1,int(len(color_array[i,1:])+1),1), node_color=color_array[i,1:],cmap = plt.cm.plasma, node_size=500)

            # Draw the edges
            nx.draw_networkx_edges(G, layout, width=1.0, alpha=0.5)

            # Draw node labels
            nx.draw_networkx_labels(G, layout)
            prob = os.path.splitext(prob_name)[0]

            # Show the plot
            plt.title(f'Graph Coloring Vectorized Problem:{prob} \nNodes:{len(nodes)} Colors:{colors}')
            plt.axis('off')

            plt.savefig(f"../Data/vectorized_graph_coloring/graph_coloring_{prob}_{i}.png")
            # plt.show()
            # plt.close()

    return color_correctness


def valid_invalid_state_space_ising(nodes,edges, hamiltonian, state_space, colors, states):
    # states = np.array(generate_all_possible_s_vectors(len(state_space)))
    states = np.array(states)
    valid_energy = []
    invalid_energy = []
    for i in range(states.shape[1]):
        possible_energy  = calculate_ising_hamiltonian_batch_parallel(nodes, hamiltonian,states[:,i,:], state_space)
        constrain_hamiltonian = create_constraint_terms_ising_hamiltonian(nodes, edges, colors, A=1)
        constraint_energy = calculate_ising_hamiltonian_batch_parallel(nodes, constrain_hamiltonian,states[:,i,:], state_space)
        validity = (constraint_energy==0)
        valid_energy.append(possible_energy[validity].tolist())
        invalid_energy.append(possible_energy[~validity].tolist())
    # print("State Array Shape", states.shape)
    # with np.printoptions(threshold=np.inf):
    #     print(states[~validity,:])

    return valid_energy, invalid_energy

def valid_invalid_state_space_ising_optm(nodes,edges, hamiltonian, state_space, colors, states, B):
    # states = np.array(generate_all_possible_s_vectors(len(state_space)))
    states = np.array(states)
    valid_energy = []
    invalid_energy = []
    for i in range(states.shape[1]):
        possible_energy  = calculate_ising_hamiltonian_batch_parallel_optm(nodes, hamiltonian,states[:,i,:], state_space, B)
        constrain_hamiltonian = create_constraint_terms_ising_hamiltonian_optm(nodes, edges, colors, B)
        constraint_energy = calculate_ising_hamiltonian_batch_parallel_optm(nodes, constrain_hamiltonian,states[:,i,:], state_space, B)
        validity = (constraint_energy==0)
        valid_energy.append(possible_energy[validity].tolist())
        invalid_energy.append(possible_energy[~validity].tolist())
    # print("State Array Shape", states.shape)
    # with np.printoptions(threshold=np.inf):
    #     print(states[~validity,:])

    return valid_energy, invalid_energy

def valid_invalid_state_space_vectorized(nodes,edges, hamiltonian, state_space, colors):
    states = np.array(generate_all_possible_s_vectors(len(state_space)))
    possible_energy  = calculate_ising_vectorized_parallel(hamiltonian,states, state_space)
    return possible_energy


### Minimum energy in each iteration and find the num of wrong edges
def failed_edges_count(state, state_space, ising_energy, num_choices, nodes, edges, colors, prob_name):
    """
    Randomly pick num_choices graphs out of all graphs 
    Zero energy implies valid graph
    """
    min_values = np.min(ising_energy, axis=1)
    # Find the index of the minimum value
    min_index = np.argmin(ising_energy, axis=1)
    state_min= np.zeros((state.shape[0], state.shape[2]))
    for indx, min in enumerate(min_index):
        state_min[indx,:] = state[indx,min,:]

    color_array = np.zeros((state.shape[0], int(max(nodes)+1))) 
    incorrect_edge_count = np.zeros(state.shape[0]) 
    for i in range(state.shape[0]):
        for indx, j in enumerate(state_min[i,:]):
                if (j== 1):
                    color_array[i, state_space[indx][0]] = state_space[indx][1]

    for i in range(state.shape[0]):
        for (u, v) in edges:
            if color_array[i, u]==color_array[i, v]:
                incorrect_edge_count[i] += 1
    return min_values, incorrect_edge_count


### Minimum energy in each iteration and find the num of wrong edges
def failed_edges_count_vectorized(state, state_space, ising_energy, num_choices, nodes, edges, colors, prob_name):

    min_values = np.min(ising_energy, axis=1)
    min_index = np.argmin(ising_energy, axis=1)
    state_min= np.zeros((state.shape[0], state.shape[2]))
    for indx, min in enumerate(min_index):
        state_min[indx,:] = state[indx,min,:]
    color_array = np.zeros((state.shape[0], int(max(nodes)+1))) 
    incorrect_edge_count = np.zeros(state.shape[0]) 

    log_color = int(math.log2(colors-1) + 1)
    for batch_indx in range (state.shape[0]) :
        state_config = {}
        state_config = {(v[0], v[1]): state_min[batch_indx,indx_state] for  indx_state, v in enumerate(state_space)}
        for node in nodes:
            for col_indx in range(log_color):
                color_array[batch_indx, int(node)] += state_config[(int(node), col_indx)]*int(math.pow(2, col_indx))



    for i in range(state.shape[0]):
        for (u, v) in edges:
            if color_array[i, u]==color_array[i, v]:
                incorrect_edge_count[i] += 1
    return min_values, incorrect_edge_count



#### Complete Matrix Implementation ############## This one works better #############

def generate_gc_hamiltonian(nodes, edges,lib, device): 
    if lib == 'numpy':
        Hamiltonian_matrix = np.zeros((max(nodes), max(nodes)))
    else:
        Hamiltonian_matrix = torch.zeros((max(nodes), max(nodes)), device = device)

    for (u, v) in edges:
        Hamiltonian_matrix[u-1, v-1] = 1
        Hamiltonian_matrix[v-1, u-1] = 1
    return Hamiltonian_matrix

def write_vecmul_gc(nodes, num_colors,lib,  device):
    num_nodes = max(nodes)
    num_color_bits = int(math.floor(math.log2(num_colors-1) +1))
    if lib == 'numpy':
        vecmul_array = np.zeros((num_nodes,int(math.pow(2, 2*num_color_bits))))
    else: 
        vecmul_array = torch.zeros((num_nodes,int(math.pow(2, 2*num_color_bits))),device = device)

    ## 1 if connected nodes give same color 
    for j in range(num_colors):
        vecmul_array[:, int(j+math.pow(2, num_color_bits)*(j))] = 1
    if lib == 'numpy':
        num_arr = np.arange(0,int(math.pow(2, num_color_bits)))
        for k in range(num_colors, int(math.pow(2, num_color_bits))):
            vecmul_array[:, (math.pow(2, num_color_bits)*num_arr+k).astype(int) ] = 1
            vecmul_array[:, (k*math.pow(2, num_color_bits)+num_arr).astype(int) ] = 1
    else:
        num_arr = torch.arange(0,int(math.pow(2, num_color_bits)))
        for k in range(num_colors, int(math.pow(2, num_color_bits))):
            vecmul_array[:, (math.pow(2, num_color_bits)*num_arr+k).to(torch.int)] = 1
            vecmul_array[:, (k*math.pow(2, num_color_bits)+num_arr).to(torch.int)] = 1

    return vecmul_array


def calculate_ising_vectorized_gc(lib, Hamiltonian_matrix, vecmul_array, state, num_colors_bits, problem_pubmed):
    if (lib == 'numpy'):
        state1 = state.reshape(state.shape[0], int(state.shape[1]/num_colors_bits), int(num_colors_bits))
        powers_of_two = 2 ** np.arange(state1.shape[-1])[::-1]
        state_int = np.dot(state1, powers_of_two)

        vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_colors_bits))*state_int[:, np.newaxis, :]
        mask1 = np.zeros_like(vecmul_inp_array)
        vecmul_array_expanded = np.expand_dims(vecmul_array, axis=0)  # Shape (1, 14, 256)
        vecmul_array_expanded = np.expand_dims(vecmul_array_expanded, axis=0)  # Shape (1, 14, 256)

        vecmul_inp_expanded = np.expand_dims(vecmul_inp_array, axis=-1)  # Shape (40, 14, 14, 1)

        # Gather elements using np.take_along_axis
        mask1 = np.take_along_axis(vecmul_array_expanded, vecmul_inp_expanded, axis=3)
        mask1 = np.squeeze(mask1, axis=-1)
        energy_array = np.where(mask1 == 1, Hamiltonian_matrix, mask1)
        return np.sum(energy_array, axis=(1,2))


    else:
        state1 = state.reshape(state.shape[0], int(state.shape[1]/num_colors_bits), int(num_colors_bits))
        state1 = state1.to(torch.float32)
        powers_of_two =  2 ** torch.arange(state1.shape[-1] - 1, -1, -1, dtype=torch.float32, device = state1.device)
        state_int = torch.matmul(state1, powers_of_two)
        if (problem_pubmed ==0):
            vecmul = vecmul_array.unsqueeze(0).unsqueeze(1).repeat(state.shape[0], int(state.shape[1]/num_colors_bits), 1, 1)

            vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_colors_bits))*state_int[:, np.newaxis, :]

            vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)
            # Gather along the last dimension (dim=3) of vecmul_array
            mask = vecmul.gather(3, vecmul_inp_array_expanded).squeeze(-1) 
            energy_array = torch.where(mask == 1, Hamiltonian_matrix, mask)

            return torch.sum(energy_array, axis=(1,2))
        else:
            energy =  0
            for i in range(int(state.shape[1]/num_colors_bits)):
                vecmul = vecmul_array.unsqueeze(0).repeat(state.shape[0], 1, 1)
                vecmul_inp_array = state_int[:, (i)].unsqueeze(1).repeat(1, state_int.shape[1]) + int(math.pow(2, num_colors_bits))*state_int
                vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)
                # Gather along the last dimension (dim=2) of vecmul_array
                mask = vecmul.gather(2, vecmul_inp_array_expanded).squeeze(-1) 
                energy_array = torch.where(mask == 1, Hamiltonian_matrix[(i),:], mask)
                energy =  energy + torch.sum(energy_array, axis=(1))
            return energy


def calculate_delta_ising_vectorized_gc(lib, Hamiltonian_matrix, vecmul_array, state, num_colors_bits,  node, color):
    flip_index = int((num_colors_bits-1-color)+(node-1)*(num_colors_bits))
    if (lib == 'numpy'):
        state1 = state.reshape(state.shape[0], int(state.shape[1]/num_colors_bits), int(num_colors_bits))
        powers_of_two = 2 ** np.arange(state1.shape[-1])[::-1]
        state_int = np.dot(state1, powers_of_two)

        vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_colors_bits))*state_int[:, np.newaxis, :]
        mask1 = np.zeros_like(vecmul_inp_array)
        vecmul_array_expanded = np.expand_dims(vecmul_array, axis=0)  # Shape (1, 14, 256)
        vecmul_array_expanded = np.expand_dims(vecmul_array_expanded, axis=0)  # Shape (1, 14, 256)

        vecmul_inp_expanded = np.expand_dims(vecmul_inp_array, axis=-1)  # Shape (40, 14, 14, 1)

        # Gather elements using np.take_along_axis
        mask1 = np.take_along_axis(vecmul_array_expanded, vecmul_inp_expanded, axis=3)
        mask1 = np.squeeze(mask1, axis=-1)
        energy_array_in = np.where(mask1 == 1, Hamiltonian_matrix, mask1)
        state[:,flip_index] =  1 -  state[:,flip_index]
        state1 = state.reshape(state.shape[0], int(state.shape[1]/num_colors_bits), int(num_colors_bits))
        powers_of_two = 2 ** np.arange(state1.shape[-1])[::-1]
        state_int = np.dot(state1, powers_of_two)

        vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_colors_bits))*state_int[:, np.newaxis, :]
        mask1 = np.zeros_like(vecmul_inp_array)
        vecmul_array_expanded = np.expand_dims(vecmul_array, axis=0)  # Shape (1, 14, 256)
        vecmul_array_expanded = np.expand_dims(vecmul_array_expanded, axis=0)  # Shape (1, 14, 256)

        vecmul_inp_expanded = np.expand_dims(vecmul_inp_array, axis=-1)  # Shape (40, 14, 14, 1)

        # Gather elements using np.take_along_axis
        mask1 = np.take_along_axis(vecmul_array_expanded, vecmul_inp_expanded, axis=3)
        mask1 = np.squeeze(mask1, axis=-1)
        energy_array_f = np.where(mask1 == 1, Hamiltonian_matrix, mask1)
        state[:,flip_index] =  1 -  state[:,flip_index]
        return (np.sum(energy_array_f, axis=(1,2)) - np.sum(energy_array_in, axis=(1,2)))
    else:
        state1 = state.reshape(state.shape[0], int(state.shape[1]/num_colors_bits), int(num_colors_bits))
        state1 = state1.to(torch.float32)
        powers_of_two =  2 ** torch.arange(state1.shape[-1] - 1, -1, -1, dtype=torch.float32, device = state1.device)
        state_int = torch.matmul(state1, powers_of_two)
        vecmul = vecmul_array.unsqueeze(0).unsqueeze(1).repeat(state.shape[0], int(state.shape[1]/num_colors_bits), 1, 1)
        vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_colors_bits))*state_int[:, np.newaxis, :]

        vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)
        # Gather along the last dimension (dim=3) of vecmul_array
        mask = vecmul.gather(3, vecmul_inp_array_expanded).squeeze(-1) 


        energy_array_in = torch.where(mask == 1, Hamiltonian_matrix, mask)
        # Calculate current energy
        # Flip the spin
        state[:,flip_index] =  1 -  state[:,flip_index]
        state1 = state.reshape(state.shape[0], int(state.shape[1]/num_colors_bits), int(num_colors_bits))
        state1 = state1.to(torch.float32)


        # state1[:,node-1, color] =  1 -  state1[:,node-1, color]

        powers_of_two =  2 ** torch.arange(state1.shape[-1] - 1, -1, -1, dtype=torch.float32, device = state1.device)
        state_int = torch.matmul(state1, powers_of_two)
        vecmul = vecmul_array.unsqueeze(0).unsqueeze(1).repeat(state.shape[0], int(state.shape[1]/num_colors_bits), 1, 1)
        vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_colors_bits))*state_int[:, np.newaxis, :]

        vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)
        # Gather along the last dimension (dim=3) of vecmul_array
        mask = vecmul.gather(3, vecmul_inp_array_expanded).squeeze(-1) 

        energy_array_f = torch.where(mask == 1, Hamiltonian_matrix, mask)


        # state1[:,node-1, color] =  1 -  state1[:,node-1, color]
        state[:,flip_index] =  1 -  state[:,flip_index]

        return (torch.sum(energy_array_f, axis=(1,2)) - torch.sum(energy_array_in, axis=(1,2)))


def calculate_delta_ising_vectorized_gc_optimized(lib, Hamiltonian_matrix, vecmul_array, state, num_colors_bits,  node, color):
    ### Only Implemented In Torch 
    flip_index = int((num_colors_bits-1-color)+(node-1)*(num_colors_bits))

    state1 = state.reshape(state.shape[0], int(state.shape[1]/num_colors_bits), int(num_colors_bits))
    state1 = state1.to(torch.float32)
    powers_of_two =  2 ** torch.arange(state1.shape[-1] - 1, -1, -1, dtype=torch.float32, device = state1.device)
    state_int = torch.matmul(state1, powers_of_two)
    # print(state_int.shape)
    vecmul = vecmul_array.unsqueeze(0).repeat(state.shape[0], 1, 1)
    # print("vecmul_shape", vecmul.shape)
    # print(state_int[:, (node-1)].shape)
    # print((state_int[:, (node-1)].repeat(1, state_int.shape[1])).shape)
    vecmul_inp_array = state_int[:, (node-1)].unsqueeze(1).repeat(1, state_int.shape[1]) + int(math.pow(2, num_colors_bits))*state_int

    vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)
    # Gather along the last dimension (dim=2) of vecmul_array
    mask = vecmul.gather(2, vecmul_inp_array_expanded).squeeze(-1) 
    energy_array_in = torch.where(mask == 1, Hamiltonian_matrix[(node-1),:], mask)
    # Calculate current energy
    # Flip the spin
    state[:,flip_index] =  1 -  state[:,flip_index]
    state1 = state.reshape(state.shape[0], int(state.shape[1]/num_colors_bits), int(num_colors_bits))
    state1 = state1.to(torch.float32)
    # state1[:,node-1, color] =  1 -  state1[:,node-1, color]
    powers_of_two =  2 ** torch.arange(state1.shape[-1] - 1, -1, -1, dtype=torch.float32, device = state1.device)
    state_int = torch.matmul(state1, powers_of_two)


    vecmul = vecmul_array.unsqueeze(0).repeat(state.shape[0], 1, 1)
    vecmul_inp_array = state_int[:, (node-1)].unsqueeze(1).repeat(1, state_int.shape[1]) + int(math.pow(2, num_colors_bits))*state_int

    vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)
    # Gather along the last dimension (dim=2) of vecmul_array
    mask = vecmul.gather(2, vecmul_inp_array_expanded).squeeze(-1) 
    energy_array_f = torch.where(mask == 1, Hamiltonian_matrix[(node-1),:], mask)


    # state1[:,node-1, color] =  1 -  state1[:,node-1, color]
    state[:,flip_index] =  1 -  state[:,flip_index]

    return 2*(torch.sum(energy_array_f, axis=(1)) - torch.sum(energy_array_in, axis=(1)))



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


#### RBM Implemenation Implemented Only in Pytorch ##########################


def generate_gc_hamiltonian_rbm(nodes, edges,lib, coupling, device): 
    if lib == 'numpy':
        Hamiltonian_matrix = np.zeros((len(nodes), len(nodes)))
    else:
        Hamiltonian_matrix = torch.zeros((len(nodes), len(nodes)), device = device)
    for (u, v) in edges:
        Hamiltonian_matrix[u-1, v-1] = 1
        Hamiltonian_matrix[v-1, u-1] = 1
    for i in range(len(nodes)):
        Hamiltonian_matrix[i, i] = -coupling*torch.sum(Hamiltonian_matrix[i, :]) ##0.5 -1 -2
    return Hamiltonian_matrix


def write_vecmul_rbm(nodes, num_colors, lib,  device):

    num_nodes = len(nodes)
    num_color_bits = int(math.floor(math.log2(num_colors-1) +1))
    if lib == 'numpy':
        vecmul_array = np.zeros((num_nodes,int(math.pow(2, 2*num_color_bits))))
    else: 
        vecmul_array = torch.zeros((num_nodes,int(math.pow(2, 2*num_color_bits))),device = device)

    ## 1 if connected nodes give same color 
    for j in range(num_colors):
        vecmul_array[:, int(j+math.pow(2, num_color_bits)*(j))] = 1
    if lib == 'numpy':
        num_arr = np.arange(0,int(math.pow(2, num_color_bits)))
        for k in range(num_colors, int(math.pow(2, num_color_bits))):
            vecmul_array[:, (math.pow(2, num_color_bits)*num_arr+k).astype(int) ] = 1
            vecmul_array[:, (k*math.pow(2, num_color_bits)+num_arr).astype(int) ] = 1
    else:
        num_arr = torch.arange(0,int(math.pow(2, num_color_bits)))
        for k in range(num_colors, int(math.pow(2, num_color_bits))):
            vecmul_array[:, (math.pow(2, num_color_bits)*num_arr+k).to(torch.int)] = 1
            vecmul_array[:, (k*math.pow(2, num_color_bits)+num_arr).to(torch.int)] = 1

    return vecmul_array


def sample_nodes(Hamiltonian_matrix, vecmul_array, state, num_colors_bits):

        # Repeat the tensor along the new axis to shape (batch, state_space, state_space)
        # print("state", state)
        expanded_state = state.unsqueeze(2)
        expanded_state = (expanded_state.repeat(1, 1, int(state.shape[1]))).transpose(1, 2)
        # expanded_state[:, torch.arange(int(state.shape[1])), torch.arange(int(state.shape[1]))] = 1
        expanded_state =  expanded_state.reshape(state.shape[0], int(state.shape[1]), int(state.shape[1]/num_colors_bits), int(num_colors_bits))
        expanded_state = expanded_state.to(torch.float32)
        # print(" expanded_state Int shape", expanded_state.shape)
        # print("expanded_state", expanded_state)

        powers_of_two =  2 ** torch.arange(expanded_state.shape[-1] - 1, -1, -1, dtype=torch.float32, device = state.device)
        state_int = torch.matmul(expanded_state, powers_of_two)
        # print("Integer State", state_int)
        # print(" State Int shape", state_int.shape)

        vecmul = vecmul_array.unsqueeze(0).unsqueeze(1).repeat(state.shape[0], state.shape[1], 1, 1)
        # print(" Vecmul  shape", vecmul.shape)
        state_int2 = state_int[:, torch.arange(state_int.shape[1]), torch.arange(state_int.shape[1])//num_colors_bits]

        # Reshape the extracted elements to shape (batch, state_space, 1)
        state_int2 = state_int2.unsqueeze(-1)

        vecmul_inp_array = state_int2 + int(math.pow(2, num_colors_bits))*state_int
        # print("vecmul_inp_array", vecmul_inp_array.shape)

        vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)
        # print("vecmul_inp_array_expanded", vecmul_inp_array_expanded.shape)
        # Gather along the last dimension (dim=3) of vecmul_array
        mask = vecmul.gather(3, vecmul_inp_array_expanded).squeeze(-1) 
        # print("mask", mask.shape)
        weight_matrix = torch.zeros((state_int.shape[1]), (state_int.shape[1])//num_colors_bits, device = state.device)

        # energy_array = torch.zeros(state.shape[0], int(state.shape[1]), device = state.device)
        ## Build adjacency or weight matrix according to num_colors repeated

        for i in range(num_colors_bits):
            index = ((np.arange(0,int(state.shape[1]),1)-i)%num_colors_bits ==0)
            weight_matrix[index, :] = Hamiltonian_matrix

        energy_array = torch.where(mask == 1, weight_matrix, mask)
        energy_array =  torch.sum(energy_array, axis=2)
        # print(energy_array.shape)


        # Repeat the tensor along the new axis to shape (batch, state_space, state_space)
        expanded_state = state.unsqueeze(2)
        expanded_state = (expanded_state.repeat(1, 1, int(state.shape[1]))).transpose(1, 2)
        expanded_state = expanded_state.int()
        expanded_state[:, torch.arange(int(state.shape[1])), torch.arange(int(state.shape[1]))] = 1 - expanded_state[:, torch.arange(int(state.shape[1])), torch.arange(int(state.shape[1]))]
        expanded_state =  expanded_state.reshape(state.shape[0], int(state.shape[1]), int(state.shape[1]/num_colors_bits), int(num_colors_bits))
        expanded_state = expanded_state.to(torch.float32)
        # print(" expanded_state Int shape", expanded_state.shape)
        # print("expanded_state", expanded_state)

        powers_of_two =  2 ** torch.arange(expanded_state.shape[-1] - 1, -1, -1, dtype=torch.float32, device = state.device)
        state_int = torch.matmul(expanded_state, powers_of_two)
        # print("Integer State", state_int)
        # print(" State Int shape", state_int.shape)

        vecmul = vecmul_array.unsqueeze(0).unsqueeze(1).repeat(state.shape[0], state.shape[1], 1, 1)
        # print(" Vecmul  shape", vecmul.shape)
        state_int2 = state_int[:, torch.arange(state_int.shape[1]), torch.arange(state_int.shape[1])//num_colors_bits]

        # Reshape the extracted elements to shape (batch, state_space, 1)
        state_int2 = state_int2.unsqueeze(-1)

        vecmul_inp_array = state_int2 + int(math.pow(2, num_colors_bits))*state_int
        # print("vecmul_inp_array", vecmul_inp_array.shape)

        vecmul_inp_array_expanded = (vecmul_inp_array.unsqueeze(-1)).to(torch.int64) # shape: (:, :, :, 1)
        # print("vecmul_inp_array_expanded", vecmul_inp_array_expanded.shape)
        # Gather along the last dimension (dim=3) of vecmul_array
        mask = vecmul.gather(3, vecmul_inp_array_expanded).squeeze(-1) 
        # print("mask", mask.shape)
        weight_matrix = torch.zeros((state_int.shape[1]), (state_int.shape[1])//num_colors_bits, device = state.device)

        # energy_array = torch.zeros(state.shape[0], int(state.shape[1]), device = state.device)
        ## Build adjacency or weight matrix according to num_colors repeated

        for i in range(num_colors_bits):
            index = ((np.arange(0,int(state.shape[1]),1)-i)%num_colors_bits ==0)
            weight_matrix[index, :] = Hamiltonian_matrix

        energy_array_f = torch.where(mask == 1, weight_matrix, mask)
        energy_array_f =  torch.sum(energy_array_f, axis=2)

        return energy_array_f.int() - energy_array.int()


def calculate_ising_vectorized_gc_rbm( Hamiltonian_matrix, vecmul_array, state, num_colors_bits):
        state1 = state.reshape(state.shape[0], int(state.shape[1]/num_colors_bits), int(num_colors_bits))
        powers_of_two = 2 ** np.arange(state1.shape[-1])[::-1]
        state_int = np.dot(state1, powers_of_two)
        vecmul_inp_array = state_int[:, :, np.newaxis] + int(math.pow(2, num_colors_bits))*state_int[:, np.newaxis, :]

        mask1 = np.zeros_like(vecmul_inp_array)
        vecmul_array_expanded = np.expand_dims(vecmul_array, axis=0)  # Shape (1, 14, 256)
        vecmul_array_expanded = np.expand_dims(vecmul_array_expanded, axis=0)  # Shape (1, 14, 256)

        vecmul_inp_expanded = (np.expand_dims(vecmul_inp_array, axis=-1)).astype(int)  # Shape (40, 14, 14, 1)


        # Gather elements using np.take_along_axis
        mask1 = np.take_along_axis(vecmul_array_expanded, vecmul_inp_expanded, axis=3)
        mask1 = np.squeeze(mask1, axis=-1)
        # print("Mask 1",mask1)
        # print("Hamiltonian Matrix", Hamiltonian_matrix)
        energy_array = np.where(mask1 == 1, Hamiltonian_matrix, mask1)
        return np.sum(energy_array, axis=(1,2))



       
def check_double_count_edge(edges):
    edges_set = set(map(tuple, edges))
    # Initialize a set to keep track of seen edges
    seen_edges = set()
    # Initialize a list to collect duplicate edges
    duplicate_edges = []

    # Check each edge in the set
    for edge in edges_set:
        v, u = edge
        # Check if the reverse edge (u, v) is already seen
        if (u, v) in seen_edges:
            duplicate_edges.append((v, u))
            return True
        else:
            seen_edges.add(edge)
    return False


##########################################################################

def create_ising_hamiltonian_optm(nodes, edges, num_colors,device, A , B):
    n = max(nodes)  # Number of nodes
    num_nodes = len(nodes)
    
    # Initialize the coefficients for the quadratic terms
    Q = {}
    unique_spins = []
    ### Ordered list of nodes
    sorted_nodes = np.sort(list(nodes))
    # for node in sorted_nodes:
    for node in range(min(sorted_nodes), max(sorted_nodes)+min(sorted_nodes)):
        for j in range(1, num_colors+1):
            unique_spins.append((node, j))
    # print(max(sorted_nodes))
    # print(len(sorted_nodes))
    # Hamiltonian_matrix = torch.zeros((len(unique_spins),len(unique_spins)), device = device)
    Hamiltonian_matrix = torch.zeros((max(sorted_nodes)*num_colors,max(sorted_nodes)*num_colors), device = device)

    # Add the term to ensure each node gets exactly one color
    for v in nodes:
        for i in range(1, num_colors+1):
            Q[(v, i, v, i)] = Q.get((v, i, v, i), 0) - B
            for j in range(1, num_colors+1):
                if i != j:
                    Q[(v, i, v, j)] = Q.get((v, i, v, j), 0) + B
    
    # Add the term to ensure adjacent nodes do not get the same color
    for (u, v) in edges:
        for i in range(1, num_colors+1):
            Q[(u, i, v, i)] = Q.get((u, i, v, i), 0) + A
            Q[(v, i, u, i)] = Q.get((v, i, u, i), 0) + A
            # if (u, i) not in unique_spins:
            #     unique_spins.append((u, i))
            # if (v, i) not in unique_spins:
            #     unique_spins.append((v, i))
    for  (v1, i1, v2, i2) in Q:
        Hamiltonian_matrix[int((i1-1)+(v1-1)*num_colors), int((i2-1)+(v2-1)*num_colors)] = Q[(v1, i1, v2, i2)]
    return Q, unique_spins, Hamiltonian_matrix

def failed_edges_count_sbm_v1(state, state_space, nodes, edges):
    """
    Randomly pick num_choices graphs out of all graphs 
    Zero energy implies valid graph
    """


    color_array = np.zeros((state.shape[0], int(max(nodes)+1))) 
    incorrect_edge_count = np.zeros(state.shape[0]) 
    invalid_edge = []
    for i in range(state.shape[0]):
        for indx, j in enumerate(state[i,:]):
                if (j== 1):
                    if( color_array[i, state_space[indx][0]] ==1):
                        invalid_edge.append(state_space[indx][0])
                    color_array[i, state_space[indx][0]] = state_space[indx][1]
                    

    for i in range(state.shape[0]):
        for (u, v) in edges:
            if(u in invalid_edge) or (v in invalid_edge):
                incorrect_edge_count[i] += 1
            elif color_array[i, u]==color_array[i, v]:
                incorrect_edge_count[i] += 1
    return incorrect_edge_count

def failed_edges_count_sbm(state, state_space, nodes, edges):
    """
    Randomly pick num_choices graphs out of all graphs 
    Zero energy implies valid graph
    """


    color_array = np.zeros((state.shape[0], int(max(nodes)+1))) 
    incorrect_edge_count = np.zeros(state.shape[0]) 
    for i in range(state.shape[0]):
        for indx, j in enumerate(state[i,:]):
                if (j== 1):
                    color_array[i, state_space[indx][0]] = state_space[indx][1]

    for i in range(state.shape[0]):
        for (u, v) in edges:
            if color_array[i, u]==color_array[i, v]:
                incorrect_edge_count[i] += 1
    return incorrect_edge_count

# def hamiltonian_cim(nodes, edges, num_colors,device, A , B):
#     ## convert the [0, 1] binary spins into [1, -1] spin values
#     n = max(nodes)  # Number of nodes
#     num_nodes = len(nodes)
    
#     # Initialize the coefficients for the quadratic terms
#     Q = {}
#     unique_spins = []
#     ### Ordered list of nodes
#     sorted_nodes = np.sort(list(nodes))
#     for node in sorted_nodes:
#         for j in range(1, num_colors+1):
#             unique_spins.append((node, j))

#     Hamiltonian_matrix = torch.zeros((len(unique_spins),len(unique_spins)), device = device)
#     # Add the term to ensure each node gets exactly one color
#     constant_constraint = 0
#     for v in nodes:
#         constant_constraint += B*math.pow(num_colors-2, 2) +B*num_colors
#         for i in range(1, num_colors+1):
#             Q[(v, i, v, i)] = Q.get((v, i, v, i), 0) + 2*B*(num_colors-2)
#             for j in range(1, num_colors+1):
#                 if i != j:
#                     Q[(v, i, v, j)] = Q.get((v, i, v, j), 0) + 2*B
    
#     # Add the term to ensure adjacent nodes do not get the same color
#     constant_edge = 0
#     for (u, v) in edges:
#         constant_edge += 2*A
#         for i in range(1, num_colors+1):
#             Q[(u, i, u, i)] = Q.get((u, i, u, i), 0)+2*A
#             Q[(v, i, v, i)]=  Q.get((v, i, v, i), 0)+2*A

#             Q[(u, i, v, i)] = Q.get((u, i, v, i), 0) + A
#             Q[(v, i, u, i)] = Q.get((v, i, u, i), 0) + A
#             # if (u, i) not in unique_spins:
#             #     unique_spins.append((u, i))
#             # if (v, i) not in unique_spins:
#             #     unique_spins.append((v, i))
#     for  (v1, i1, v2, i2) in Q:
#         Hamiltonian_matrix[int((i1-1)+(v1-1)*num_colors), int((i2-1)+(v2-1)*num_colors)] = Q[(v1, i1, v2, i2)]
#     return Q, unique_spins, Hamiltonian_matrix, constant_edge+constant_constraint

    # Hamiltonian_matrix_convert = torch.zeros((Hamiltonian_matrix.shape[0],Hamiltonian_matrix.shape[1]), device = device)
    # diag_part = torch.diag(torch.diagonal(Hamiltonian_matrix))
    # non_diag_part = Hamiltonian_matrix - diag_part
    # non_diag_sum = torch.diag(torch.sum(non_diag_part, dim =1)/2)
    # non_diag_part = non_diag_part/4
    # Hamiltonian_matrix_convert = non_diag_part - 0.5*(non_diag_sum+diag_part)
    # return Hamiltonian_matrix_convert


def hamiltonian_cim(nodes, edges, num_colors,device, A , B):
    ## convert the [0, 1] binary spins into [1, -1] spin values
    n = max(nodes)  # Number of nodes
    num_nodes = len(nodes)
    
    # Initialize the coefficients for the quadratic terms
    Q = {}
    unique_spins = []
    ### Ordered list of nodes
    sorted_nodes = np.sort(list(nodes))
    for node in sorted_nodes:
        for j in range(1, num_colors+1):
            unique_spins.append((node, j))

    Hamiltonian_matrix = torch.zeros((len(unique_spins),len(unique_spins)), device = device)
    # Add the term to ensure each node gets exactly one color
    for v in nodes:
        for i in range(1, num_colors+1):
            Q[(v, i, v, i)] = Q.get((v, i, v, i), 0) + (num_colors-1)*(B/2)
            for j in range(1, num_colors+1):
                if i != j:
                    Q[(v, i, v, j)] = Q.get((v, i, v, j), 0) + B/4
    
    # Add the term to ensure adjacent nodes do not get the same color
    for (u, v) in edges:
        for i in range(1, num_colors+1):
            Q[(u, i, u, i)] = Q.get((u, i, u, i), 0)+A/4
            Q[(v, i, v, i)]=  Q.get((v, i, v, i), 0)+A/4

            Q[(u, i, v, i)] = Q.get((u, i, v, i), 0) + A/8
            Q[(v, i, u, i)] = Q.get((v, i, u, i), 0) + A/8

    for  (v1, i1, v2, i2) in Q:
        Hamiltonian_matrix[int((i1-1)+(v1-1)*num_colors), int((i2-1)+(v2-1)*num_colors)] = Q[(v1, i1, v2, i2)]
    return Q, unique_spins, Hamiltonian_matrix, 0

def failed_edges_count_cim(state, state_space, nodes, edges):
    """
    Randomly pick num_choices graphs out of all graphs 
    Zero energy implies valid graph
    """


    color_array = np.zeros( int(max(nodes)+1))
     

    for indx, j in enumerate(state[:]):
            if (j== 1):
                color_array[ state_space[indx][0]] = state_space[indx][1]
    incorrect_edge_count =0

    for (u, v) in edges:
        if color_array[ u]==color_array[ v]:
            incorrect_edge_count += 1
    return incorrect_edge_count
