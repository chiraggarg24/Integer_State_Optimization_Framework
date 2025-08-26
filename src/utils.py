import numpy as np  
import torch
import math
import itertools
import sys





  

  

def sigmoid(x):

    """ To avoid Overflow errors """

    if x >= 0:

        z = math.exp(-x)

        sig = 1 / (1 + z)

        return sig

    else:

        z = math.exp(x)

        sig = z / (1 + z)

        return sig




  

    