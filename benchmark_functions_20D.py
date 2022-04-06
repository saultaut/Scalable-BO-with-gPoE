import torch
from POEBO.gpoebo import GPOEBO
from Random.random_search import RANDOM_SEARCH
from POEBO.gpoetrbo import GPOETRBO
from BO.bo_ucb import BO
import pandas as pd
import traceback
import os

from botorch.test_functions import Ackley, Rosenbrock, Levy, Rastrigin

def transform_data(name, arr):
    data = {'Model' : [name] * len(arr),
            'Iteration' : list(range(len(arr))),
            'Values' : arr}
    return data

if __name__ == '__main__':
    
    device = torch.device("cpu")
    dtype = torch.float
    
    NO_DIMENSIONS=20
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

    obj_functions = {"Levy" : levy,
                      "Rastrigin" : rastrigin,
                      "Rosenbrock" : rosenbrock,
                     "Ackley": ackley
                     }

    model_list = ["gpoetrbo"]

    currentDirectory = os.getcwd()
    path = currentDirectory + "/results/"
    experiment_no = 1

    max_evals = 500
    n_init = 50
    points_per_expert = 50
    for function_name, opt_function in obj_functions.items():
                  
        N_TRIALS = 10

        for model_name in model_list:
            
            time_history = []
            opt_history = []
            for trial in range(N_TRIALS):

                models = {
                  "gpoebo_y" : GPOEBO(f=opt_function, points_per_expert=points_per_expert, n_init=n_init, max_evals=max_evals, n_candidates=5000, batch_size=5000),
                  "bo" : BO(f=opt_function, n_init=n_init, max_evals=max_evals, n_candidates=5000, batch_size=5000),
                  "gpoetrbo" : GPOETRBO(f=opt_function, points_per_expert=points_per_expert, n_init=n_init, prob_perturb=1.0, perturb_rate=20, max_evals=max_evals, n_candidates=5000, batch_size=5000),
                  "Random_Search" : RANDOM_SEARCH(f=opt_function, n_init=n_init, max_evals=max_evals)
                  }

                print(f"Model: {model_name} Trial : {str(trial)}")
                model = models[model_name]

                try:
                  running_time, fx = model.optimize()
                  time_history.append(running_time)
                  opt_history.append(fx.squeeze(-1).numpy())
                except Exception as ex:
                  print(f"ERROR in model: {model_name}")
                  traceback.print_exc()            
            

                
            model_name_file= model_name + "_" + function_name + "_" + str(NO_DIMENSIONS) + "D_" + str(experiment_no)
            
            results = [pd.DataFrame(transform_data(model_name, arr)) for arr in opt_history]
            df = pd.concat(results)
            df.to_pickle(path+model_name_file+"_results.pkl")
        
        
            
            best_values = [arr[-1] for arr in opt_history]
            
            d = {"Model" : [model_name] * len(best_values),
                  "time" : time_history,
                  "best_values" : best_values}
            df_time = pd.DataFrame(d)
            df_time.to_pickle(path+model_name_file+"_results_time.pkl")
