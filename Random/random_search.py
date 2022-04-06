import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
import gpytorch
import numpy as np
import time
import math

class RANDOM_SEARCH:
    def __init__(self,
                 f,
                 n_init=50,
                 max_evals=100):
        
        self.current_obj_fun = f
        self.dim = f.dim

        self.no_iterations = max_evals + n_init # we run from the start

        
        self.device = torch.device("cpu")
        self.dtype = torch.float

    def eval_objective_function(self, x, obj_fun):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return obj_fun(unnormalize(x, obj_fun.bounds))

    def optimize(self):

        starttime = time.time()
        best_random = [] 
        for iteration in range(self.no_iterations):

            random_candidate_x = torch.rand(self.dim, dtype=self.dtype, device=self.device)
            next_random_y = self.eval_objective_function(random_candidate_x, self.current_obj_fun)
            next_random_best = next_random_y.item()
            best_random.append(next_random_best)

        
        endtime = time.time()
        print(f"Time taken {endtime-starttime} seconds")
   
        # Print current status
        print(
            f"{len(best_random)}) Best value: {np.max(best_random)}"
        )

        running_time = endtime-starttime

        fx = np.maximum.accumulate(best_random)
        
        return running_time, torch.Tensor(fx)
       
if __name__ == '__main__':
    
    device = torch.device("cpu")
    dtype = torch.float
    
    NO_DIMENSIONS=10
    rosenbrock = Rosenbrock(dim=NO_DIMENSIONS, negate=True).to(dtype=dtype, device=device)
    rosenbrock.bounds[0, :].fill_(-10)
    rosenbrock.bounds[1, :].fill_(10)
    
    levy = Levy(dim=NO_DIMENSIONS, negate=True).to(dtype=dtype, device=device)
    levy.bounds[0, :].fill_(-10)
    levy.bounds[1, :].fill_(10)
    
    rastrigin = Rastrigin(dim=NO_DIMENSIONS, negate=True).to(dtype=dtype, device=device)
    rastrigin.bounds[0, :].fill_(-5.12)
    rastrigin.bounds[1, :].fill_(5.12)
    
    ackley = Ackley(dim=NO_DIMENSIONS, negate=True).to(dtype=dtype, device=device)
    ackley.bounds[0, :].fill_(-5)
    ackley.bounds[1, :].fill_(10)
    
    obj_functions = {#"levy" : levy,
                     #"rastrigin" : rastrigin,
                      #"rosenbrock" : rosenbrock#,
                      "ackley": ackley
                     }
    
    random_search = RANDOM_SEARCH(f=ackley, n_init=20, max_evals=200)
    running_time, fx = random_search.optimize()
    print(fx.shape)

    print(f"Best value found: {fx[-1]}")
    print(fx.squeeze(-1).numpy())
            

            

            

        
        
        