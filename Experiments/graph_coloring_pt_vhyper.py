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
parameters = pt_hyperparameter()

################################## Problem-Set #################################

spin_system = 0    # 1: For Ising spins 0/1  0: For Ising spins +1/-1
problem_set_path = "../Dataset/graph_coloring_dataset/"
hyperparameter = ising_opt
problem_set = Graph_Coloring_hyperparam(problem_set_path, hyperparameter, hyperparameter)

N = 1
problem = None    # set according to the problem

for ii, problem in enumerate(problem_set):
    print("Problem Name", problem.file_prefix)
    print(hyperparameter[problem.file_prefix.split(".")[0]])
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

    sampler = Gibbs_Sampler_Parallel_Tempering(time_steps,pt_interval, N, batch_size, scheduler, update, problem)
    sampler.set_problem(problem)
    states, ising_energy = sampler.Run(return_states = True)
    mode_columns = mode(ising_energy, axis=1)
    num_choices =5
    valid_energy, invalid_energy = valid_invalid_state_space_ising_optm(problem.nodes,problem.edges, problem.hamiltonian, problem.state_space, problem.num_colors, states, hyperparameter[problem.file_prefix.split(".")[0]][1])

    min_values, incorrect_edge_count = failed_edges_count(states, problem.state_space, ising_energy, num_choices, problem.nodes, problem.edges, problem.num_colors, problem.file_prefix)

    sampler_tts = Gibbs_Sampler_Parallel_Tempering(10,pt_interval, N, batch_size, scheduler, update, problem)

    sampler_tts.set_problem(problem)
    t0 = time.time()
    _, _ = sampler_tts.Run(return_states = True)
    t1 = time.time()
    ########## Analysis #################
    np.savez(f'../Data/ising_graph_coloring_solution_pt/{os.path.splitext(problem.file_prefix)[0]}.npz', 
             problem_name=os.path.splitext(problem.file_prefix)[0], 
             nodes=problem.nodes, 
             num_nodes=len(problem.nodes),
             edges=problem.edges,
             num_edges = len(problem.edges),
             num_colors = problem.num_colors,
             implemented_ising_nodes = len(problem.state_space),
             hamiltonian = problem.hamiltonian,
             state_space = problem.state_space,
             interaction_order = 2,
             hamiltonian_terms = len(problem.hamiltonian.values()),
             sampled_states = states,
             ising_energy = ising_energy,
             valid_energy_states = list(itertools.chain.from_iterable(valid_energy)),
             invalid_energy_states = list(itertools.chain.from_iterable(invalid_energy)),
             mode_energy_samples = mode(ising_energy, axis=1),
             min_energy_samples = np.min(ising_energy, axis=1),
             batch_size = batch_size,
             time_steps = time_steps,
             beta = temp0,
             incorrect_edge_count = incorrect_edge_count,
             time_per_sample = (t1 - t0)/10 
             )
