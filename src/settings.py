# def init(device = 'cpu', num_threads = 2):
#     global properties
#     properties =  {'DEVICE' : device, 'NUM_THREADS': num_threads}
    
def init(lib = 'numpy', device = 'cpu', num_threads = 2):
  global properties
  properties =  {'LIB' : lib, 'DEVICE' : device, 'NUM_THREADS': num_threads}
  if(lib == 'pytorch'):
    import torch
    torch.set_num_threads(num_threads)