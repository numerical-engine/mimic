import numpy as np

def roulette_wheel(selection_num:int, prob:np.ndarray)->tuple[int]:
    indice = np.random.choice(len(prob), selection_num, replace = True, p = prob)
    return tuple(indice)

def SUS(selection_num:int, prob:np.ndarray)->tuple[int]:
    indice = []
    r = np.random.uniform(0., 1./selection_num)
    prob_cum = np.cumsum(prob)

    check_point = 0
    while len(indice) < selection_num:
        while True:
            if r <= prob_cum[check_point]:
                indice.append(check_point)
                r += 1./selection_num
            else:
                break
        check_point += 1
    
    return tuple(indice)