import numpy as np
import sys

class Optimizer:
    def __call__(self, population, environment):
        population_new = population.copy()
        population_new.generation += 1
        dim = population_new.dim

        Y = np.stack([np.random.multivariate_normal(np.zeros(dim), population_new.C) for _ in range(population_new.lam)], axis = 0)
        X = np.stack([population_new.mean + population_new.sigma*y for y in Y], axis = 0)
        F = environment.get_score(X)

        sort_idx = np.argsort(F)[:population_new.mu]
        weight = population_new.weight[:population_new.mu].reshape((-1, 1))
        Y_sort = Y[sort_idx]
        y_step = np.sum(weight*Y_sort, axis = 0)

        population_new.mean += population_new.c_m*population_new.sigma*y_step

        eig, B = np.linalg.eig(population_new.C)
        D_inv = np.diag(1./eig)
        C_sqrtm = B@D_inv@(B.T)

        population_new.p_sigma = (
            (1. - population_new.c_sigma)*population_new.p_sigma
            +
            np.sqrt(population_new.c_sigma*(2. - population_new.c_sigma)*population_new.mu_eff)*C_sqrtm@y_step
            )
        
        chiN = np.sqrt(dim)*(1. - 0.25/dim + 1./21./(dim**2))
        ps_norm = np.linalg.norm(population_new.p_sigma)
        population_new.sigma *= np.exp(
            population_new.c_sigma/population_new.d_sigma
            *(ps_norm/chiN - 1.)
            )

        h_sigma = 1. if (ps_norm/np.sqrt(1. - (1. - population_new.c_sigma)**(2.*population_new.generation)) < (1.4 + 2./(dim + 1.))*chiN) else 0.
        
        population_new.p_c = (
            (1. - population_new.c_c)*population_new.p_c
            +
            h_sigma*np.sqrt(population_new.c_c*(2. - population_new.c_c)*population_new.mu_eff)*y_step
        )

        weight_pre = []
        for w, y in zip(population_new.weight, Y_sort):
            if w > 0.:
                weight_pre.append(w)
            else:
                weight_pre.append(dim/(np.linalg.norm(C_sqrtm@y)**2)*w)
        weight_pre = np.array(weight_pre)
        C_new = (
            (1. + (1. - h_sigma)*population_new.c_c*(2. - population_new.c_c)
                - population_new.c_1 - population_new.c_mu*np.sum(population_new.weight)
            )*population_new.C
            +
            population_new.c_1*(population_new.p_c.reshape((-1, 1)))@(population_new.p_c.reshape((1, -1)))
            +
            population_new.c_mu*1
            )
        population_new.C = 0.5*(C_new + C_new.T)

        return population_new