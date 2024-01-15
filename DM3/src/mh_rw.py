import numpy as np

def target_distribution(x):
    if x>0:
        return 1 / (x + 1e-6)
    else:
        return 0

def metropolis_hastings_rw(n_iter, burn_in_period, initial_value, scale):
    samples = [initial_value]

    for t in range(n_iter):
        x = samples[-1]
        epsilon = np.random.normal(0, scale)
        y = x + epsilon

        p_x = target_distribution(x)
        p_y = target_distribution(y)
        if (p_x==0)&(p_y==0):
            ratio_pdf = 1
        elif (p_x==0)&(p_y>0):
            ratio_pdf = p_y/(p_x+1e-6)
        else:
            ratio_pdf = p_y/p_x
        acceptance_ratio = min(1, ratio_pdf)
        b = np.random.binomial(n=1, p=acceptance_ratio)
        if b==1:
            proposed_sample = y
        else:
            proposed_sample = x
        
        samples.append(proposed_sample)

    del samples[0]
    return np.array(samples)[burn_in_period:]