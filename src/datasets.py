import numpy as np
import os
import sys
from os import listdir
from os.path import join
from src.problems import *
from src.graph_coloring_mapping_util import *
from src import settings
import tsplib95


__all__ = ["Graph_Coloring", "Graph_Coloring_vectorized", "TSP_vectorized", "TSP_ising", "Graph_Coloring_benchmark", "Graph_Coloring_hyperparam"]

class DataSet():
    r"""Dataset class that iterates over problems given in a folder
    
      Args:
          path(str): Path to the folder containing problems
      Info: 
          .col files are dimac files containing the connectivity information
          parse_dimacs function returns 
          nodes as dictionary
          edges as a list of tupple 
          num_colors is defined in chromatic_numbers in utils
      """
    
    
    def __init__(self, path):
        self.path = path
        self.dirlist = listdir(path)    # Contains the list of files with a certain extension in the directory
        self.numlist = len(self.dirlist)    # Number of files in directory
      
    def __iter__(self):
        r""" Iterates over all the examples in the problem set and return weights"""
        self.current_index = -1 
        return self
    
    def __next__(self):
        if(self.current_index < self.numlist - 1):
            self.current_index += 1
            file_path = join(self.path, self.dirlist[self.current_index])
            file_prefix = file_path.split('.')[0]
            nodes, edges = parse_dimacs(file_path)
            num_colors = chromatic_numbers[file_prefix]
            self.problem = Ising_Graph_Coloring(nodes, edges, num_colors)
            return self.problem
        else:
            raise StopIteration
    

    
class Graph_Coloring(DataSet):
    r"""Dataset class that handles examples of max_cut ising problems given in the folder G-set_W/
    
      Args:
          path(str): Path to the folder containing G-set max cut problems
          
      """
    def __init__(self, path):
        super(Graph_Coloring, self).__init__(path)
      
    def __next__(self):

        self.dirlist = ['anna.col', 'david.col',  'huck.col', 'myciel3.col', 'myciel4.col', 'myciel5.col', 'myciel6.col', 'myciel7.col', 'queen11_11.col', 'queen13_13.col', 'queen5_5.col', 'queen6_6.col', 'queen7_7.col', 'queen8_12.col', 'queen8_8.col', 'queen9_9.col']
        # self.dirlist = ['cora.col', 'citeseer.col', 'pubmed_graph.col']

        self.numlist = len(self.dirlist)
        if(self.current_index < self.numlist - 1):
            self.current_index += 1
            file_path = join(self.path, self.dirlist[self.current_index])
            filename = file_path.split('/')[-1]
            file_prefix = filename ##.split('.')[0]
            nodes, edges = parse_dimacs(file_path)
            num_colors = chromatic_numbers[file_prefix]
            self.problem = Ising_Graph_Coloring(file_prefix, nodes, edges, num_colors)
            return self.problem
        else:
            raise StopIteration
      
      

  
    

class Graph_Coloring_vectorized(DataSet):
    r"""Dataset class that handles examples of max_cut ising problems given in the folder G-set_W/
    
      Args:
          path(str): Path to the folder containing G-set max cut problems
          
      """
    def __init__(self, path):
        super(Graph_Coloring_vectorized, self).__init__(path)
        
      
    def __next__(self):
        self.dirlist = ['anna.col', 'david.col',  'huck.col', 'myciel3.col', 'myciel4.col', 'myciel5.col', 'myciel6.col', 'myciel7.col', 'queen11_11.col', 'queen13_13.col', 'queen5_5.col', 'queen6_6.col', 'queen7_7.col', 'queen8_12.col', 'queen8_8.col', 'queen9_9.col']
        # self.dirlist = ['cora.col', 'citeseer.col', 'pubmed_graph.col']

        self.numlist =len(self.dirlist)
        if(self.current_index < self.numlist - 1):
            self.current_index += 1
            file_path = join(self.path, self.dirlist[self.current_index])
            filename = file_path.split('/')[-1]
            file_prefix = filename ##.split('.')[0]
            nodes, edges = parse_dimacs(file_path)
            num_colors = chromatic_numbers[file_prefix]
            self.problem = Ising_Graph_Coloring_vectorized(file_prefix, nodes, edges, num_colors)
            return self.problem
        else:
            raise StopIteration
class Graph_Coloring_hyperparam(DataSet):
    r"""Dataset class that handles examples of max_cut ising problems given in the folder G-set_W/
    
      Args:
          path(str): Path to the folder containing G-set max cut problems
          val1(int)
          val2(int)
          
      """
    def __init__(self, path, val1, val2):
        super(Graph_Coloring_hyperparam, self).__init__(path)
        self.val1 = val1
        self.val2 = val2
        
      
    def __next__(self):
        self.dirlist = ['anna.col', 'david.col',  'huck.col', 'myciel3.col', 'myciel4.col', 'myciel5.col', 'myciel6.col', 'myciel7.col', 'queen11_11.col', 'queen13_13.col', 'queen5_5.col', 'queen6_6.col', 'queen7_7.col', 'queen8_12.col', 'queen8_8.col', 'queen9_9.col']
        # self.dirlist = ['cora.col', 'citeseer.col', 'pubmed_graph.col']
        self.numlist =len(self.dirlist)
        if(self.current_index < self.numlist - 1):
            self.current_index += 1
            file_path = join(self.path, self.dirlist[self.current_index])
            
            filename = file_path.split('/')[-1]
            file_prefix = filename ##.split('.')[0]
            nodes, edges = parse_dimacs(file_path)
            num_colors = chromatic_numbers[file_prefix]
            self.problem = Ising_Graph_Coloring_hyperparam(file_prefix, nodes, edges, num_colors, self.val1[filename.split('.')[0]][0], self.val2[filename.split('.')[0]][1])
            # self.problem = Ising_Graph_Coloring_hyperparam(file_prefix, nodes, edges, num_colors, self.val1, self.val2)
            return self.problem
        else:
            raise StopIteration

class Graph_Coloring_benchmark(DataSet):
    r"""Dataset class that handles examples of max_cut ising problems given in the folder G-set_W/
    
      Args:
          path(str): Path to the folder containing G-set max cut problems
          val1(int)
          val2(int)
          
      """
    def __init__(self, path, problem_index, val1, val2):
        super(Graph_Coloring_benchmark, self).__init__(path)
        self.val1 = val1
        self.val2 = val2
        self.dirlist = [problem_index]
      
    def __next__(self):

        self.numlist =len(self.dirlist)
        if(self.current_index < self.numlist - 1):
            self.current_index += 1
            file_path = join(self.path, self.dirlist[self.current_index])
            print
            filename = file_path.split('/')[-1]
            file_prefix = filename ##.split('.')[0]
            nodes, edges = parse_dimacs(file_path)
            num_colors = chromatic_numbers[file_prefix]
            self.problem = Ising_Graph_Coloring_benchmark(file_prefix, nodes, edges, num_colors, self.val1, self.val2)
            return self.problem
        else:
            raise StopIteration
        

  
    
class TSP_vectorized(DataSet):
    r"""Dataset class that handles examples of max_cut ising problems given in the folder G-set_W/
    
      Args:
          path(str): Path to the folder containing G-set max cut problems
          
      """
    def __init__(self, path):
        super(TSP_vectorized, self).__init__(path)
        

    def __next__(self):
        self.dirlist = ['burma4.tsp', 'burma6.tsp', 'burma8.tsp', 'burma10.tsp', 'burma12.tsp', 'burma14.tsp']

        self.numlist = len(self.dirlist)
        if(self.current_index < self.numlist - 1):
            self.current_index += 1
            file_path = join(self.path, self.dirlist[self.current_index])
            dataset_load = tsplib95.load(file_path)
            self.problem =  Ising_TSP_vectorized(dataset_load) 
            return self.problem
        else:
            raise StopIteration
        
class TSP_ising(DataSet):
    r"""Dataset class that handles examples of max_cut ising problems given in the folder G-set_W/
    
      Args:
          path(str): Path to the folder containing G-set max cut problems
          
      """
    def __init__(self, path, A, B):
        super(TSP_ising, self).__init__(path)
        self.A = A
        self.B = B

    def __next__(self):
        self.dirlist = ['burma4.tsp', 'burma6.tsp', 'burma8.tsp', 'burma10.tsp', 'burma12.tsp', 'burma14.tsp']

        self.numlist = len(self.dirlist)
        if(self.current_index < self.numlist - 1):
            self.current_index += 1
            file_path = join(self.path, self.dirlist[self.current_index])
            filename = file_path.split('/')[-1]
            dataset_load = tsplib95.load(file_path)
            # self.problem = Ising_TSP(dataset_load, self.A, self.B) 
            self.problem = Ising_TSP(dataset_load, self.A[filename.split('.')[0]][1], self.B[filename.split('.')[0]][0])
            
            return self.problem
        else:
            raise StopIteration