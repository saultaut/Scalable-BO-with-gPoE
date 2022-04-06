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

class BO:
    def __init__(self,
                 f,
                 n_init=50,
                 max_evals=100,
                 n_candidates=5000,
                 batch_size=5000):
        
        self.current_obj_fun = f
        self.dim = f.dim
        self.beta = torch.tensor(1.0)
        self.n_init = n_init  # 2*dim, which corresponds to 5 batches of 4
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        
        self.no_iterations = max_evals
        
        self.device = torch.device("cpu")
        self.dtype = torch.float

    def get_initial_points(self, dim, n_pts):
        sobol = SobolEngine(dimension=dim, scramble=True)
        X_init = sobol.draw(n=n_pts).to(dtype=self.dtype, device=self.device)
        return X_init

    def eval_objective_function(self, x, obj_fun):
        """This is a helper function we use to unnormalize and evalaute a point"""
        return obj_fun(unnormalize(x, obj_fun.bounds))

    def get_fitted_model(self, train_X, train_Y):
        """
        Get a single task GP. The model will be fit unless a state_dict with model 
            hyperparameters is provided.
        """
        
        model = SingleTaskGP(train_X, train_Y)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_model(mll)
        return model

    def get_untrained_model(self, train_X, train_Y):
        """
        Get a single task GP. The model will be fit unless a state_dict with model 
            hyperparameters is provided.
        """
        
        model = SingleTaskGP(train_X, train_Y)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        return model

    def optimize(self):

        X_train = self.get_initial_points(self.dim, self.n_init)
        Y_train = torch.tensor(
            [self.eval_objective_function(x, self.current_obj_fun) for x in X_train], dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        

        starttime = time.time() 
        for iteration in range(self.no_iterations):
           # Fit a GP model
            try:
                model = self.get_fitted_model(X_train, Y_train)
            except Exception as ex:
                print(f"ERROR: {ex}")
                print("using untrained model")

                model = self.get_untrained_model(X_train, Y_train)
            # Create a batch
        
            #ucb posterior
            X_test = self.get_initial_points(self.dim, self.n_candidates)
            posterior = model.posterior(X_test)
            y_pred = model.likelihood(posterior.mvn)
            
            mu_s = y_pred.mean.cpu().detach()
            var_s = y_pred.variance.cpu().detach()
            
            ucb = mu_s + torch.sqrt(self.beta) * var_s.sqrt()
            best_index = ucb.argmax()
            
            # get best prediction
            X_next = X_test[best_index, :].clone().unsqueeze(0)
            
            Y_next = torch.tensor(
                [self.eval_objective_function(x, self.current_obj_fun) for x in X_next], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            
            # Append data
            X_train = torch.cat((X_train, X_next), dim=0)
            Y_train = torch.cat((Y_train, Y_next), dim=0)
            
            # Print current status
            print(
                f"{len(X_train)}) Best value: {Y_train.max()}"
            )
        
        endtime = time.time()
        print(f"Time taken {endtime-starttime} seconds")
        
        running_time = endtime-starttime

        fx = np.maximum.accumulate(Y_train.cpu())
        
        return running_time, fx
       
if __name__ == '__main__':
    
    device = torch.device("cpu")
    dtype = torch.float
    
    NO_DIMENSIONS=3
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
    
    bo = BO(f=ackley, n_init=10, max_evals=30, n_candidates=5000, batch_size=5000)
    bo.optimize()
            

            

            

        
        
        