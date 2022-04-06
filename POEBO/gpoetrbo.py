import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.constraints.constraints import Interval

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import pandas as pd

import time
from dataclasses import dataclass
from copy import deepcopy

device = torch.device("cpu")
dtype = torch.float

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        print("restart..........")
        # state.length  = state.length_min
        state.restart_triggered = True
    return state

class GPOETRBO:
    def __init__(self,
                 f,
                 points_per_expert = 20,
                 n_init=50,
                 prob_perturb = 0.2,
                 perturb_rate = 200,
                 max_evals=100,
                 n_candidates=5000,
                 batch_size=5000):
        
        self.current_obj_fun = f
        self.dim = f.dim
        self.beta = torch.tensor(1.0)
        
        self.prob_perturb = prob_perturb
        self.perturb_rate = perturb_rate
        
        self.n_init = n_init  # 2*dim, which corresponds to 5 batches of 4
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        
        self.POINTS_PER_EXPERT = points_per_expert
        self.partition_type = 'random'
        
        self.no_iterations = max_evals + n_init
        self.n_evals = 0

        # Save the full history
        self.X = torch.zeros((0, self.dim))
        self.fX = torch.zeros((0, 1))
        
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
        fit_gpytorch_model(mll, options={'maxiter': 500})
        return model
    
    def normalize_weights(self, weight_matrix):
        """ Compute unnormalized weight matrix
        """
        
        sum_weights = torch.sum(weight_matrix, axis=0)
        weight_matrix = weight_matrix / sum_weights
        
        return weight_matrix

    def compute_weights(self, mu_s, var_s, weighting, prior_var=None, softmax=False, power=10):
        
        """ Compute unnormalized weight matrix
        """
        
        if weighting == 'uniform':
            weight_matrix = torch.ones(mu_s.shape) / mu_s.shape[0]
    
        if weighting == 'diff_entr':
            weight_matrix = 0.5 * (torch.log(prior_var) - torch.log(var_s))
    
        if weighting == 'variance':
            weight_matrix = torch.exp(-power * var_s)
            
        if weighting == 'no_weights':
            weight_matrix = 1
        
        return weight_matrix

    def optimize(self):


        X_init = self.get_initial_points(self.dim, self.n_init)
        fX_init = torch.tensor(
            [self.eval_objective_function(x, self.current_obj_fun) for x in X_init], dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        
        # Update budget and set as initial data for this TR
        self.n_evals += self.n_init
        X_turbo = deepcopy(X_init)
        Y_turbo = deepcopy(fX_init)

        # Append data to the global history
        self.X = torch.cat((self.X, deepcopy(X_turbo)), dim=0)
        self.fX = torch.cat((self.fX, deepcopy(Y_turbo)), dim=0)

        state = TurboState(dim=self.dim, batch_size=1)
        ones_matrix = np.ones((self.n_candidates, self.dim))
        starttime = time.time() 

        while self.n_evals < self.no_iterations:

            # RESTART LOGIC
            if state.restart_triggered:
                print(f"{self.n_evals}) Restarting with fbest = {self.fX.max():.4}")
                state = TurboState(dim=self.dim, batch_size=1)

                X_init = self.get_initial_points(self.dim, self.n_init)
                fX_init = torch.tensor(
                    [self.eval_objective_function(x, self.current_obj_fun) for x in X_init], dtype=self.dtype, device=self.device
                ).unsqueeze(-1)
                
                # Update budget and set as initial data for this TR
                self.n_evals += self.n_init
                X_turbo = deepcopy(X_init)
                Y_turbo = deepcopy(fX_init)

                # Append data to the global history
                self.X = torch.cat((self.X, deepcopy(X_turbo)), dim=0)
                self.fX = torch.cat((self.fX, deepcopy(Y_turbo)), dim=0)

            # Compute number of experts 
            N_EXPERTS = int(np.max([X_turbo.shape[0] / self.POINTS_PER_EXPERT, 1]))
            # Compute number of points experts 
            N = int(X_turbo.shape[0] / N_EXPERTS)
            # If random partition, assign random subsets of data to each expert
            if self.n_evals % 20 == 0:
                print(f'Number of experts:{N_EXPERTS}')
            
            partition = []
            
            if self.partition_type == 'random':
                partition = np.random.choice(X_turbo.shape[0], size=(N_EXPERTS, N), replace=False)            

            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
            best_Y_index = Y_turbo.argmax()
                
    
            X_best = X_turbo[best_Y_index, :]
            
            batched_train_X = torch.stack([X_turbo[partition[k]] for k in range(N_EXPERTS)]).to(device=self.device, dtype=self.dtype)
            batched_train_Y = torch.stack([train_Y[partition[k]] for k in range(N_EXPERTS)]).to(device=self.device, dtype=self.dtype)
            
            try:
                model = self.get_fitted_model(batched_train_X, batched_train_Y)
            except:
                continue

            model = model.to(device=self.device, dtype=self.dtype)

            
            # ####  Turbo algorithm
            weights_algorithm = "other-uniform"
            if weights_algorithm == "uniform":
                weights = torch.ones(self.dim)
            else:
                # weights = model.covar_module.base_kernel.lengthscale.cpu().detach()[0, 0, :]
                weights = model.covar_module.base_kernel.lengthscale.detach()[0, 0, :]
                
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
            # weights = weights.to(device=device, dtype=dtype)
            x_center = X_best
            tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
            

            pert = self.get_initial_points(self.dim, self.n_candidates)#.cpu().detach().numpy()
                
            pert = tr_lb + (tr_ub - tr_lb) * pert
            #X_test[0,:] = x_center #include current best
                    # Create a perturbation mask
            if self.n_evals % self.perturb_rate == 0 or self.n_evals == self.n_init:
                prob_perturb = self.prob_perturb #min(2 / self.dim, 1.0) # used to be set as 0.2
                mask = np.random.rand(self.n_candidates, self.dim) <= prob_perturb
                ind = np.where(np.sum(mask, axis=1) == 0)[0]
                mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1
            

    
            # # Create candidate points
            X_test = x_center.numpy().copy() * ones_matrix
            X_test[mask] = pert[mask]
            X_test = torch.tensor(X_test).to(device=device, dtype=dtype)
        
            
            #these values can be pre-allcoated at the begining with maximum size values, but efficiency is not clear.
            mu_s = torch.zeros(N_EXPERTS, self.n_candidates).to(device=self.device, dtype=self.dtype)
            var_s = torch.zeros(N_EXPERTS, self.n_candidates).to(device=self.device, dtype=self.dtype)
            prior = torch.zeros(N_EXPERTS, self.n_candidates).to(device=self.device, dtype=self.dtype)

            num_batches = math.ceil(self.n_candidates/self.batch_size)            
            for i in range(num_batches):
                start_i, end_i = i*self.batch_size, (i+1)*self.batch_size
                try:
                    # get prior
                    with gpytorch.settings.prior_mode(True):
                        #y_pred = f + noise
                        y_prior = model.likelihood(model(X_test[start_i : end_i]))        
                        prior[:, start_i : end_i] = y_prior.variance.detach()  
                    
                    #get posterior
                    posterior = model.posterior(X_test[start_i : end_i])
                    y_pred = model.likelihood(posterior.mvn)
                    
                    mu_s[:, start_i : end_i] = y_pred.mean.cpu().detach()
                    var_s[:, start_i : end_i] = y_pred.variance.cpu().detach()
                    
                except:
                    #default
                    print("exception")            

            del batched_train_X, batched_train_Y
            weight_matrix = self.compute_weights(mu_s, var_s, weighting='diff_entr', prior_var=prior, softmax=False)
            del prior
            # Compute individual precisions - dim: n_experts x n_test_points
            prec_s = 1/var_s
            weight_matrix = self.normalize_weights(weight_matrix)
        
            prec = torch.sum(weight_matrix * prec_s, axis=0)
            
                            
            var = 1 / prec
            mu = var * torch.sum(weight_matrix * prec_s * mu_s, axis=0)
            
            del mu_s, var_s, 
            
            mu = torch.reshape(mu, (-1, 1))
            var = torch.reshape(var, (-1, 1))
            
            # #######  UCB ######
            ucb = mu + torch.sqrt(self.beta) * var.sqrt()
            best_index = ucb.argmax()
            # ###################
            
           
            X_next = X_test[best_index, :].clone().unsqueeze(0)
            del X_test
            # print(f'Current best X: {mu[best_index, :].clone()}')
            # print(f'Current best X: {X_next}')
            Y_next = torch.tensor(
                [self.eval_objective_function(x, self.current_obj_fun) for x in X_next], dtype=self.dtype, device=self.device
            ).unsqueeze(-1)
            
            
            # Update state
            state = update_state(state=state, Y_next=Y_next)
        

            # Update budget and append data
            self.n_evals += 1
            X_turbo = torch.cat((X_turbo, X_next), dim=0)
            Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)
            

            # Append data to the global history
            self.X = torch.cat((self.X, deepcopy(X_next)), dim=0)
            self.fX = torch.cat((self.fX, deepcopy(Y_next)), dim=0)

            # Print current status
            print(
                f"{len(X_turbo)}) Best value: {Y_turbo.max()}  # {len(self.fX)} Global Best value: {self.fX.max()}"
            )

        
        endtime = time.time()
        print(f"Time taken {endtime-starttime} seconds")
        
        running_time = endtime-starttime

        fx = np.maximum.accumulate(self.fX.cpu())
        
        return running_time, fx
       
if __name__ == '__main__':
    
    device = torch.device("cpu")
    dtype = torch.float
    
    NO_DIMENSIONS=50
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
    
    gpoetrbo = GPOETRBO(f=ackley, points_per_expert=100, n_init=100, prob_perturb=0.4, max_evals=1000, n_candidates=5000, batch_size=5000)
    gpoetrbo.optimize()
            

            

            

        
        
        