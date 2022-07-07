
"""

Copyright (c) 2017, Mostapha Kalami Heris & Yarpiz (www.yarpiz.com)
All rights reserved. Please read the "LICENSE" file for usage terms.
__________________________________________________________________________

Project Code: YPEA127
Project Title: Implementation of Particle Swarm Optimization in Python
Publisher: Yarpiz (www.yarpiz.com)

Developer: Mostapha Kalami Heris (Member of Yarpiz Team)

Cite as:
Mostapha Kalami Heris, Particle Swarm Optimization (PSO) in Python (URL: https://yarpiz.com/463/ypea127-pso-in-python), Yarpiz, 2017.

Contact Info: sm.kalami@gmail.com, info@yarpiz.com

"""

import numpy as np
import pandas as pd
import copy, os, multiprocessing, dill
import concurrent.futures
# from pathos.multiprocessing import ProcessingPool as Pool

# Particle Swarm Optimization
class PSO:
    def __init__(self, CostFunction, search, MaxIter = 100, PopSize = 2, c1 = 1.4962, c2 = 1.4962, w = 0.7298, wdamp = 1.0):
        self.MaxIter = MaxIter
        self.PopSize = PopSize
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.wdamp = wdamp
        self.search = search
        self.CostFunction = CostFunction
        self.update = False
  


        self.nVar = len(search)

        self.gbest = {'position': None, 'cost': np.NINF}

        empty_particle = {
            'position': search.copy(), 
            'velocity':search.copy(), 
            'cost': None, 
            'best_position': None, 
            'best_cost': np.NINF, 
        }

        self.pop = []
        for i in range(0, self.PopSize):
            self.pop.append(copy.deepcopy(empty_particle))

            for j in range(self.nVar):            
                var_name = list(search.keys())[j]

                VarMin, VarMax = search[var_name]
                self.pop[i]['position'][var_name] = np.random.uniform(VarMin, VarMax)###
                self.pop[i]['velocity'][var_name] = 1                      

    def particle_process(self, i):
        # print(f"\tpid ({os.getpid()})")
        particle = self.pop[i]
        if self.update == True:
            for j in range(self.nVar):
                var_name = list(self.search.keys())[j]
                
                particle['velocity'][var_name] = self.w*particle['velocity'][var_name] \
                    + self.c1*np.random.rand()*(particle['best_position'][var_name] - particle['position'][var_name]) \
                    + self.c2*np.random.rand()*(self.gbest['position'][var_name] - particle['position'][var_name])

                particle['position'][var_name] += particle['velocity'][var_name]
                VarMin, VarMax = self.search[list(self.search.keys())[j]]
                particle['position'][var_name] = np.maximum(particle['position'][var_name], VarMin)
                particle['position'][var_name] = np.minimum(particle['position'][var_name], VarMax)

        # print(particle['position'])
        # print(CostFunction.__dict__)
        particle['cost'] = self.CostFunction( **particle['position'])
        
        if particle['cost'] > particle['best_cost']:
            particle['best_position'] = copy.deepcopy(particle['position'])
            particle['best_cost'] = particle['cost']

            if particle['best_cost'] > self.gbest['cost']:
                self.gbest['position'] = copy.deepcopy(particle['best_position'])
                self.gbest['cost'] = particle['best_cost']
        # print(f' particle : {particle}'    )
        self.pop[i] = particle 
        return particle      

    def __call__(self,):
        log = []
        log_best = []
        ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", default=multiprocessing.cpu_count()))
        pool = multiprocessing.Pool(processes=ncpus)
        result = []
        for i in range(0, self.PopSize):
            result.append( apply_async(pool, self.particle_process, args=(i,)) )
        
        pool.close()
        pool.join()
        self.pop = [p.get() for p in result]
        log = copy.deepcopy(self.pop)
        self.update = True
        self.updating_gbest()
        log_best.append(copy.deepcopy(self.gbest))
        # PSO Loop
        print(f"Iteration -1 : Best Cost = {self.gbest['cost']} best position: {self.gbest['position']}")
        for it in range(0, self.MaxIter):
            

            ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", default=multiprocessing.cpu_count()))
            pool = multiprocessing.Pool(processes=ncpus)
            result = []
            for i in range(0, self.PopSize):
                result.append( apply_async(pool, self.particle_process, args=(i,)) )
                # self.particle_process(i)
            pool.close()
            pool.join()
            self.pop = [p.get() for p in result]
            self.updating_gbest()
            log += self.pop 
            log_best.append(self.gbest)

            

            self.w *= self.wdamp
            print(f"Iteration {it}: Best Cost = {self.gbest['cost']} best position: {self.gbest['position']}")

        return pd.DataFrame(log_best), pd.DataFrame(log)
    
    def updating_gbest(self,):
        for i in range(0, self.PopSize):
            if self.pop[i]['best_cost'] > self.gbest['cost']:
                self.gbest['position'] = copy.deepcopy(self.pop[i]['best_position'])
                self.gbest['cost'] = self.pop[i]['best_cost']

    @staticmethod
    def bacc(ys, yhats, positive=True):
        """Computes a contingency table for given predictions.

        :param ys: true labels
        :type ys: iterable
        :param yhats: predicted labels
        :type yhats: iterable
        :param positive: the positive label

        :return: TP, FP, TN, FN

        >>> ys =    [True, True, True, True, True, False]
        >>> yhats = [True, True, False, False, False, True]
        >>> tab = contingency_table(ys, yhats, 1)
        >>> print(tab)
        (2, 1, 0, 3)

        """
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for y, yhat in zip(ys, yhats):
            if y == positive:
                if y == yhat:
                    TP += 1
                else:
                    FN += 1
            else:
                if y == yhat:
                    TN += 1
                else:
                    FP += 1
        sp = (TN / (FP + TN + 1e-33)) * 100
        se = (TP / (FN + TP + 1e-33)) * 100
        return (sp + se)/2

def apply_async(pool, fun, args, ):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


def run_dill_encoded(what):
    fun, args = dill.loads(what)
    return fun(*args)


# def particle_process(CostFunction, gbest, search, particle, w, c1, c2):
#     print(f"\tpid ({os.getpid()})")
#     nVar = len(search)
#     for j in range(nVar):
#         var_name = list(search.keys())[j]
        

#         particle['velocity'][var_name] = w*particle['velocity'][var_name] \
#             + c1*np.random.rand()*(particle['best_position'][var_name] - particle['position'][var_name]) \
#             + c2*np.random.rand()*(gbest['position'][var_name] - particle['position'][var_name])


#         particle['position'][var_name] += particle['velocity'][var_name]
#         VarMin, VarMax = search[list(search.keys())[j]]
#         particle['position'][var_name] = np.maximum(particle['position'][var_name], VarMin)
#         particle['position'][var_name] = np.minimum(particle['position'][var_name], VarMax)

#     print(particle['position'])
#     print(CostFunction.__dict__)
#     particle['cost'] = CostFunction(**particle['position'])
    
#     if particle['cost'] > particle['best_cost']:
#         particle['best_position'] = particle['position'].copy()
#         particle['best_cost'] = particle['cost']

#         # if pop[i]['best_cost'] > gbest['cost']:
#         #     gbest['position'] = pop[i]['best_position'].copy()
#         #     gbest['cost'] = pop[i]['best_cost']
#     print(f' particle : {particle}'    )
#     return particle

# def main():
#     """

#     Copyright (c) 2017, Mostapha Kalami Heris & Yarpiz (www.yarpiz.com)
#     All rights reserved. Please read the "LICENSE" file for usage terms.
#     __________________________________________________________________________

#     Project Code: YPEA127
#     Project Title: Implementation of Particle Swarm Optimization in Python
#     Publisher: Yarpiz (www.yarpiz.com)

#     Developer: Mostapha Kalami Heris (Member of Yarpiz Team)

#     Cite as:
#     Mostapha Kalami Heris, Particle Swarm Optimization (PSO) in Python (URL: https://yarpiz.com/463/ypea127-pso-in-python), Yarpiz, 2017.

#     Contact Info: sm.kalami@gmail.com, info@yarpiz.com

#     """

#     # import pso

#     # A Sample Cost Function
#     def Sphere(x):
#         return sum(x**2)

#     # Define Optimization Problem
#     problem = {
#             'CostFunction': Sphere, 
#             'nVar': 10, 
#             'VarMin': -5,   # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
#             'VarMax': 5,    # Alternatively you can use a "numpy array" with nVar elements, instead of scalar
#         }
#     search = {
#             'knn': {'n_neighbors': [1, 20]},
#             'svm-linear': {'logC': [-4, 3]},
#             'svm-rbf': {'logGamma': [-6, 0], 'logC': [-4, 3]},
#             'svm-poly': {'logGamma': [2, 5], 'logC': [-4, 3], 'coef0': [0, 1]},
#             'rf': {'n_estimators': [20, 120], 'max_features': [5, 25]},
#             'if': {'n_estimators': [20, 120], 'max_features': [5, 25]},
#             'ocsvm': { 'nu': [0, 1]},
#             'svdd': {'nu': [0, 1], 'logGamma': [-6, 0]},
#             'tm': None,
#             'lda': None,
#         }
    
#     print(search['svm-rbf'])
#     # gbest, pop = PSO(problem, MaxIter = 200, PopSize = 50, c1 = 1.5, c2 = 2, w = 1, wdamp = 0.995)
#     breakpoint()



        
      


#     # Running PSO
#     pso.tic()
#     print('Running PSO ...')
#     gbest, pop = pso.PSO(problem, MaxIter = 200, PopSize = 50, c1 = 1.5, c2 = 2, w = 1, wdamp = 0.995)
#     print()
#     pso.toc()
#     print()

#     # Final Result
#     print('Global Best:')
#     print(gbest)
#     print()

# if __name__ == "__main__":
#     main()