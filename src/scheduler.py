import math
import numpy as np

__all__ = ["Constant",
           "Multiplicative_Exponential",
           "Multiplicative_Logarithmic",
           "Multiplicative_Linear",
           "Multiplicative_Quadratic",
           "Additive_Linear",
           "Additive_Quadratic",
           "Additive_Exponential",
           "Additive_Trignometric",
           "Parallel_Tempering_Schedular"]


class Constant():
    r""" Returns constant temperature 
    
      Args:
        param (Dict): Dicionary of the scheduler parameters. Extract the constant temperature
        ii seems to be th iteration number
    """
    
    def __init__(self, param):
        self.temp0 = param["temp0"]
    
    def scheduled_temp(self, ii = 0):
        return self.temp0
        

class Multiplicative_Exponential():
    r""" Returns temperature according to  
      
      math :: Tk = T0 \alpha^k    (0.8 <= \alpha <= 0.9)
      
      Args:
        param (Dict): Dicionary of the scheduler parameters                    
                      temp0 (float64): Initial temperature
                      alpha (float64): Multiplicative rate
                  
    """
    
    def __init__(self, param):
        self.temp0 = param["temp0"]
        self.alpha = param["alpha"]
      
    def scheduled_temp(self, ii = 0):     
        return self.temp0*self.alpha**ii
    
  


class Multiplicative_Logarithmic(): 
    r""" Returns temperature according to  
      
      math :: Tk = T0/(1 + \alpha \log(1 + k))    (\alpha > 1.0)
      
      Args:
        param (Dict): Dicionary of the scheduler parameters    
                    temp0 (float64): Initial temperature
                    alpha (float64): Decrease rate
    
    """
    
    def __init__(self, param):
        self.temp0 = param["temp0"]
        self.alpha = param["alpha"]
      
    def scheduled_temp(self, ii = 0):
        return self.temp0/(1 + self.alpha*math.log(1 + ii))    
    
    
class Multiplicative_Linear():
    r""" Returns temperature according to  
      
      math :: Tk = T0/(1 + \alpha k)    (\alpha > 0.0)
      
      Args:
        param (Dict): Dicionary of the scheduler parameters    
                      temp0 (float64): Initial temperature
                      alpha (float64): Decrease rate
    
    """
    
    def __init__(self, param):
        self.temp0 = param["temp0"]
        self.alpha = param["alpha"]
      
    def scheduled_temp(self, ii = 0):     
        return self.temp0/(1 + self.alpha*ii)
    
    
class Multiplicative_Quadratic():
    r""" Returns temperature according to  
      
      math :: Tk = T0/(1 + \alpha k**2)    (\alpha > 0.0)
      
      Args:
        param (Dict): Dicionary of the scheduler parameters
                      temp0 (float64): Initial temperature
                      alpha (float64): Decrease rate
    
    """
    
    def __init__(self, param):
        self.temp0 = param["temp0"]
        self.alpha = param["alpha"]
      
    def scheduled_temp(self, ii = 0):     
        return self.temp0/(1 + self.alpha*(ii**2))
  
  
class Additive_Linear():
    r""" Returns temperature according to  
      
      math :: Tk = Tn + (T0 - Tn)((n-k)/n)
      
      Args:
        param (Dict): Dicionary of the scheduler parameters
                      temp0 (float64): Initial temperature
                      tempn (float64): Final Temperature
                      num (float64) : Number of cooling cycles
    
    """
    
    def __init__(self, param):
        self.temp0 = param["temp0"]
        self.tempn = param["tempn"]
        self.num = param["num"]
      
    def scheduled_temp(self, ii = 0):     
        return self.tempn + (self.temp0 - self.tempn)*((self.num - ii)/self.num)
    
    
class Additive_Quadratic():
    r""" Returns temperature according to  
      
      math :: Tk = Tn + (T0 - Tn)((n-k)/n)^2
      
      Args:
        param (Dict): Dicionary of the scheduler parameters
                      temp0 (float64): Initial temperature
                      tempn (float64): Final temperature
                      num (float64) : Number of cooling cycles
    
    """
    
    def __init__(self, param):
        self.temp0 = param["temp0"]
        self.tempn = param["tempn"]
        self.num = param["num"]
      
    def scheduled_temp(self, ii = 0):     
        return self.tempn + (self.temp0 - self.tempn)*((self.num - ii)/self.num)**2
    
class Additive_Exponential():
    r""" Returns temperature according to  
      
      math :: Tk = Tn + (T0 - Tn)(1/(1 + exp(2*log(T0 - Tn)/n)(k - n/2)))
      
      Args:
        param (Dict): Dicionary of the scheduler parameters
                      temp0 (float64): Initial temperature
                      tempn (float64): Final temperature
                      num (float64) : Number of cooling cycles
    
    """
    
    def __init__(self, param):
        self.temp0 = param["temp0"]
        self.tempn = param["tempn"]
        self.num = param["num"]
      
    def scheduled_temp(self, ii = 0):     
        return self.tempn + (self.temp0 - self.tempn)*(1/(1 + math.exp(2*math.log(self.temp0 - self.tempn)/self.num*(ii - self.num/2))))
    
    
class Additive_Trignometric():
    r""" Returns temperature according to  
      
      math :: Tk = Tn + 1/2(T0 - Tn)(1 + cos(k\pi/n))
      
      Args:
        param (Dict): Dicionary of the scheduler parameters
                      temp0 (float64): Initial temperature
                      tempn (float64): Final temperature
                      num (float64) : Number of cooling cycles
                  
    """
    
    def __init__(self, param):
        self.temp0 = param["temp0"]
        self.tempn = param["tempn"]
        self.num = param["num"]
      
    def scheduled_temp(self, ii = 0):     
        return self.tempn + 1/2*(self.temp0 - self.tempn)*(1 + math.cos(ii*math.pi/self.num))
    

class Parallel_Tempering_Schedular():
    r""" Returns temperature according to  
      
      math :: Tk = Tn + 1/2(T0 - Tn)(1 + cos(k\pi/n))
      
      Args:
        param (Dict): Dicionary of the scheduler parameters
                      temp0 (float64): Initial temperature
                      tempn (float64): Final temperature
                      num (float64) : Number of geometrically spaced temperature
                  
    """
    
    def __init__(self, param):
        self.temp0 = param["temp0"]
        self.tempn = param["tempn"]
        self.num = param["num"]
      
    def scheduled_temp(self):     
        return np.geomspace(self.temp0, self.tempn, self.num)

    
  
  
  
    
    
    
    
    