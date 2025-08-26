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
from src.tsp_util import *
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

################################## Problem-Set #################################

spin_system = 0    # 1: For Ising spins 0/1  0: For Ising spins +1/-1
problem_set_path = "../Dataset/tsp_datset/tsp/burma_dataset/"
problem_set = TSP_vectorized(problem_set_path)

################################# Scheduler #####################################

temp0 = 0.2    # Initial temperature
scheduler = Constant({'temp0':temp0})

################################# Update ########################################

update = Gibbs_Update()

################################ Sampler ########################################
time_steps = 4000 
N = 1
batch_size = 500  
problem = None    # set according to the problem
tsp_vec_param = tsp_vec_param

for ii, problem in enumerate(problem_set):
    print("Problem Name", problem.file_prefix)

    problem.A = tsp_vec_param[problem.file_prefix]
    print("wt_factor", tsp_vec_param[problem.file_prefix])
    sampler = Gibbs_Sampler(time_steps, N, batch_size, scheduler, update, problem)
    sampler.set_problem(problem)
    states, ising_energy = sampler.Run(return_states = True)


    mode_columns = mode(ising_energy, axis=1)
    percent_correct_states, min_energy, tour_cost, states_val = solution_quality(problem.lib, states, time_steps-1, ising_energy, int(len(problem.nodes)), problem.num_time_len)
    optimal_tour, optimal_tour_cost, state_int_array, sample_correctness, sample_tour_cost = tsp_tour_cost(problem.lib, states,ising_energy, 100, int(len(problem.nodes)), problem.num_time_len)
    tour_array = np.array(tour_cost)
    tour_array = tour_array.reshape(batch_size, time_steps-1)
    

    sampler_tts = Gibbs_Sampler(10, N, 1, scheduler, update, problem)
    sampler_tts.set_problem(problem)
    t0 = time.time()
    _, _ = sampler_tts.Run(return_states = True)
    t1 = time.time()


    # ######### Analysis #################
    np.savez(f'../Data/tsp/{os.path.splitext(problem.file_prefix)[0]}.npz', 
             problem_name=os.path.splitext(problem.file_prefix)[0], 
             nodes=problem.nodes, 
             num_nodes=len(problem.nodes),
             edges=problem.edges,
             tour_cost = tour_cost,
             min_tour_cost = np.min(tour_array),
             states_val = states_val,
             time_steps = time_steps,
             batch_size = batch_size,
             wt_factor =tsp_vec_param[problem.file_prefix],
             temp0 =temp0,
             time_per_sample = (t1 - t0)/10 
    )

