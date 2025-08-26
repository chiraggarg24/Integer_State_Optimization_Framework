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
# device = 'cpu'
settings.init(lib, device)
parameters = pt_hyperparameter()
################################## Problem-Set #################################

spin_system = 0    # 1: For Ising spins 0/1  0: For Ising spins +1/-1
problem_set_path = "../Dataset/graph_coloring_dataset/"
problem_set = Graph_Coloring_vectorized(problem_set_path)



N = 1
problem = None    # set according to the problem
batch_loop = 10
for ii, problem in enumerate(problem_set):
    print("Problem Name", problem.file_prefix)

    ################################# Scheduler #####################################


    batch_size = parameters[problem.file_prefix][0]
    temp0 = 0.01    # Initial temperature
    tempn = 40
    num =batch_size
    scheduler = Parallel_Tempering_Schedular({'temp0': temp0, 'tempn': tempn, 'num': num})

    ################################# Update ########################################

    update = Gibbs_Update()

    ################################ Sampler ########################################
    time_steps = 1000
    pt_interval = parameters[problem.file_prefix][1]
    ################################################################################################

    states = np.zeros((batch_size*batch_loop, time_steps, len(problem.state_space)))
    ising_energy = np.ones((batch_size*batch_loop, time_steps))
    for lp in range(batch_loop):
        sampler = Gibbs_Sampler_Parallel_Tempering(time_steps,pt_interval, N, (batch_size), scheduler, update, problem)
        sampler.set_problem(problem)
        states_arr, ising_energy_arr = sampler.Run(return_states = True)
        states[lp*(batch_size):(lp+1)*(batch_size),:,:] = states_arr
        ising_energy[lp*(batch_size):(lp+1)*(batch_size),:] = ising_energy_arr
    mode_columns = mode(ising_energy, axis=1)

    num_choices =5
    # truth = graph_plot_vectorized_coloring(states, problem.state_space, ising_energy, num_choices, problem.nodes, problem.edges, problem.num_colors,problem.file_prefix)
    min_values, incorrect_edge_count = failed_edges_count_vectorized(states, problem.state_space, ising_energy, num_choices, problem.nodes, problem.edges, problem.num_colors, problem.file_prefix)
    if(check_double_count_edge(problem.edges)==0):
        incorrect_edge_count = incorrect_edge_count
    else: 
        incorrect_edge_count = incorrect_edge_count/2
    plot_ising_energy(ising_energy)


    sampler_tts = Gibbs_Sampler_Parallel_Tempering(50,pt_interval, N, (batch_size), scheduler, update, problem)
    
    sampler_tts.set_problem(problem)
    t0 = time.time()
    _ = sampler_tts.Run(return_states = True)
    t1 = time.time()
    time_per_sample = (t1 - t0)/50 
    
    min_tts_vectorized = np.min(ising_energy, axis=0)
    absolute_min = np.min(ising_energy)
    tts_first_occurance_vectorized = np.where(min_tts_vectorized==absolute_min)[0][0]

    # ######### Analysis #################
    np.savez(f'../Data/vectorized_graph_coloring_solution_pt/{os.path.splitext(problem.file_prefix)[0]}.npz', 
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
             temp0 = temp0,
             tempn = tempn,
             time_steps = time_steps,
             pt_interval = pt_interval,
             beta = temp0,
             incorrect_edge_count = incorrect_edge_count,
             time_per_sample = (t1 - t0)/50,
             tts_first_occurance_vectorized = tts_first_occurance_vectorized
            )