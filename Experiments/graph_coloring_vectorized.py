import numpy as np

import scipy as sp
import os
import sys
import itertools
import matplotlib.pyplot as plt
from scipy.stats import mode
import time
parentdir = os.path.dirname(os.getcwd())
sys.path.append(parentdir)

from src.scheduler import *
from src.update import *
from src.sampler import *
from src.problems import *
from src.datasets import *
from src.graph_coloring_mapping_util import *
from src import settings

################################## CPU #################################

# lib = 'numpy'
# device = 'cpu'
# settings.init(lib, device)

################################## GPU #################################

lib = 'pytorch'
device = 'cuda:0'
settings.init(lib, device)

################################## Problem-Set #################################

spin_system = 0    # 1: For Ising spins 0/1  0: For Ising spins +1/-1
problem_set_path = "../Dataset/graph_coloring_dataset/"
problem_set = Graph_Coloring_vectorized(problem_set_path)

################################# Scheduler #####################################

temp0 = 0.2    # Initial temperature
scheduler = Constant({'temp0':temp0})

################################# Update ########################################

update = Gibbs_Update()

################################ Sampler ########################################
time_steps = 1000
N = 1
batch_size = 200
problem = None    # set according to the problem
batch_loop = 1
for ii, problem in enumerate(problem_set):
    print("Problem Name", problem.file_prefix)
    # print("Nodes ", problem.nodes, " Edges ",problem.edges)
    # print("\nHamiltonian ", problem.hamiltonian, " state_space ",problem.state_space)
    states = np.zeros((batch_size, time_steps, len(problem.state_space)))
    ising_energy = np.ones((batch_size, time_steps))
    for lp in range(batch_loop):
        sampler = Gibbs_Sampler(time_steps, N, (batch_size//batch_loop), scheduler, update, problem)
        sampler.set_problem(problem)
        states_arr, ising_energy_arr = sampler.Run(return_states = True)
        states[lp*(batch_size//batch_loop):(lp+1)*(batch_size//batch_loop),:,:] = states_arr
        ising_energy[lp*(batch_size//batch_loop):(lp+1)*(batch_size//batch_loop),:] = ising_energy_arr

    mode_columns = mode(ising_energy, axis=1)

    num_choices =5
    min_values, incorrect_edge_count = failed_edges_count_vectorized(states, problem.state_space, ising_energy, num_choices, problem.nodes, problem.edges, problem.num_colors, problem.file_prefix)
    
    sampler_tts = Gibbs_Sampler(10, N, batch_size, scheduler, update, problem)
    sampler_tts.set_problem(problem)
    t0 = time.time()
    _, _ = sampler_tts.Run(return_states = True)
    t1 = time.time()
    # ######### Analysis #################
    np.savez(f'../Data/vectorized_graph_coloring_solution/{os.path.splitext(problem.file_prefix)[0]}.npz', 
             problem_name=os.path.splitext(problem.file_prefix)[0], 
             nodes=problem.nodes, 
             num_nodes=len(problem.nodes),
             edges=problem.edges,
             num_edges = len(problem.edges),
             num_colors = problem.num_colors,
             implemented_ising_nodes = len(problem.state_space),
             hamiltonian = problem.hamiltonian,
             hamiltonian_expression = problem.hamiltonian_expression,
             state_space = problem.state_space,
             logic_expression = problem.gate_expression,
             logic_terms =  problem.or_gate_count,
             sampled_states = states,
             ising_energy = ising_energy,
             valid_energy_states = np.unique(ising_energy),
             mode_energy_samples = mode(ising_energy, axis=1),
             min_energy_samples = np.min(ising_energy, axis=1),
             batch_size = batch_size,
             time_steps = time_steps,
             beta = temp0,
             incorrect_edge_count = incorrect_edge_count,
             time_per_sample = (t1 - t0)/10 
             )