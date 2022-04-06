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

class GPOEBO:
    def __init__(self,
                 f,
                 points_per_expert = 20,
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
        
        self.POINTS_PER_EXPERT = points_per_expert
        #self.FIXED_N_EXPERTS = 16
        self.partition_type = 'random'
        
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

    def get_unfitted_model(self, train_X, train_Y):
        """
        Get a single task GP. The model will be fit unless a state_dict with model 
            hyperparameters is provided.
        """
        
        model = SingleTaskGP(train_X, train_Y)
        model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_model(mll)
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

        X_turbo = self.get_initial_points(self.dim, self.n_init)
        Y_turbo = torch.tensor(
            [self.eval_objective_function(x, self.current_obj_fun) for x in X_turbo], dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        

        starttime = time.time() 
        for iteration in range(self.no_iterations):
            # Compute number of experts 
            N_EXPERTS = int(np.max([X_turbo.shape[0] / self.POINTS_PER_EXPERT, 1]))
            # Compute number of points experts 
            N = int(X_turbo.shape[0] / N_EXPERTS)
            # If random partition, assign random subsets of data to each expert
            if iteration % 20 == 0:
                print(f'Number of experts:{N_EXPERTS}')
            
            partition = []
            
            if self.partition_type == 'random':
                partition = np.random.choice(X_turbo.shape[0], size=(N_EXPERTS, N), replace=False)            

            train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
            
            batched_train_X = torch.stack([X_turbo[partition[k]] for k in range(N_EXPERTS)]).to(device=self.device, dtype=self.dtype)
            batched_train_Y = torch.stack([train_Y[partition[k]] for k in range(N_EXPERTS)]).to(device=self.device, dtype=self.dtype)
            
            try:
                model = self.get_fitted_model(batched_train_X, batched_train_Y)
            except Exception as ex:
                print(f"ERROR: {ex}")
                print("using untrained model")

                model = self.get_unfitted_model(batched_train_X, batched_train_Y)     

            model = model.to(device=self.device, dtype=self.dtype)
            # posterior prediction and sampling           
            X_test = self.get_initial_points(self.dim, self.n_candidates)
            
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
        
            # Append data
            X_turbo = torch.cat((X_turbo, X_next), dim=0)
            Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)
            

            # Print current status
            print(
                f"{len(X_turbo)}) Best value: {Y_turbo.max()}"
            )
        
        endtime = time.time()
        print(f"Time taken {endtime-starttime} seconds")
        
        running_time = endtime-starttime

        fx = np.maximum.accumulate(Y_turbo.cpu())
        
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
    
    gpoebo = GPOEBO(f=ackley, points_per_expert=400, n_init=1000, max_evals=20, n_candidates=5000, batch_size=5000)
    gpoebo.optimize()
            

            

            

        
        
        