import numpy as np
from copy import deepcopy

class Population:
    def __init__(self,
                mean:np.ndarray,
                sigma:float,
                lam:int = None,
                weight:np.ndarray = None,
                c_m:float = None,
                c_sigma:float = None,
                d_sigma:float = None,
                c_c:float = None,
                c_1:float = None,
                c_mu:float = None,
                p_c:np.ndarray = None,
                p_sigma:np.ndarray = None,
                C:np.ndarray = None,
                generation:int = 0,
                set_default:bool = True
                ):
        
        self.generation = generation
        self.mean = mean
        self.sigma = sigma
        self.p_c = np.zeros(self.dim) if p_c is None else p_c
        self.p_sigma = np.zeros(self.dim) if p_sigma is None else p_sigma
        self.C = np.eye(self.dim) if C is None else C

        if set_default:
            self.set_default_values()
        else:
            self.lam = lam
            self.weight = weight
            self.c_m = c_m
            self.c_sigma = c_sigma
            self.d_sigma = d_sigma
            self.c_c = c_c
            self.c_1 = c_1
            self.c_mu = c_mu
    
    def set_default_values(self):
        self.lam = int(4 + np.floor(3.*np.log(self.dim)))
        self.c_m = 1.
        self.c_sigma = (self.mu_eff + 2.)/(self.dim + self.mu_eff + 5.)
        self.d_sigma = 1. + 2.*max(0., np.sqrt((self.mu_eff - 1.)/(self.dim + 1.)) - 1.) + self.c_sigma
        self.c_c = (4. + self.mu_eff/self.dim)/(self.dim + 4. + 2.*self.mu_eff/self.dim)
        self.c_1 = 2./((self.dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1. - self.c_1, 2.*(0.25 + self.mu_eff + 1./self.mu_eff - 2.)/((self.dim+2)**2 + self.mu_eff))
        
        w_pre = np.array([np.log(0.5*(self.lam + 1.)) - np.log(i) for i in range(1, self.lam + 1)])
        alpha_mu = 1. + self.c_1/self.c_mu

        mu = int(np.floor(self.lam*0.5))
        mu_eff_m = (np.sum(w_pre[mu:])**2)/np.sum(w_pre[mu:]**2)
        alpha_mueff = 1. + 2.*mu_eff_m/(self.mu_eff + 2.)
        alpha_pos = (1. - self.c_1 - self.c_mu)/(self.dim*self.c_mu)

        alpha = min(alpha_mu, alpha_mueff, alpha_pos)

        wpre_p = np.sum(w_pre[w_pre > 0.])
        wpre_m = -np.sum(w_pre[w_pre < 0.])

        w = []
        for wp in w_pre:
            if wp > 0.:
                w.append(wp/wpre_p)
            else:
                w.append(wp*alpha/wpre_m)
        self.weight = np.array(w)

    def copy(self):
        mean_cpy = deepcopy(self.mean)
        weight_cpy = deepcopy(self.weight)
        p_c_cpy = deepcopy(self.p_c)
        p_sigma_cpy = deepcopy(self.p_sigma)
        C_cpy = deepcopy(self.C)

        return type(self)(mean_cpy, self.sigma, self.lam, weight_cpy, self.c_m,
                        self.c_sigma, self.d_sigma, self.c_c, self.c_1, self.c_mu,
                        p_c_cpy, p_sigma_cpy, C_cpy, self.generation, False)
    @property
    def mu_eff(self):
        w_pre = np.array([np.log(0.5*(self.lam + 1.)) - np.log(i) for i in range(1, self.lam + 1)])
        mu = int(np.floor(self.lam*0.5))
        return (np.sum(w_pre[:mu])**2)/np.sum(w_pre[:mu]**2)
    
    @property
    def mu(self):
        return int(np.floor(self.lam*0.5))
        
    @property
    def dim(self):
        return len(self.mean)