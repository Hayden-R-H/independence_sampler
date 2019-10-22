"""
:@title: Independence Sampler
:@author: Hayden Reece Hohns
:@date: 23/09/2019
:@brief: This is an example of an independence sampler, the simplest form of a Markov Chain Monte Carlo (MCMCM) algorithm. This script is based off of the example found in:

Kroese, Dirk P.. Statistical Modeling and Computation (p. 216). Springer New York. Kindle Edition. 

The independence sampler works by choosing a proposal distribution that is 
independent of X, i.e., q(y|x) = g(y). Although this simplifies the process, it 
also produces dependent samples. Note that g(y) should be chosen to be close to 
the target distribution if possible.

"""

import math
import matplotlib.pyplot as plt
import numpy as np 


def independence_sampler(numSamples: int):
    """
    BRIEF

    This function generates samples from the target distribution given a 
    required number of samples.

    ARGUMENTS

    :@numSamples: An integer representing the number of samples.

    RETURNS

    :@X: A list containing the samples from the target distribution in the form of a Markov chain.

    """

    # f is unnormalised target pdf
    f = lambda x: x ** 2 * math.exp(-x **2 + math.sin(x))
    # g is proposal pdf
    g = lambda x: math.exp(-np.abs(x)) / 2
    # acceptance probability
    alpha = lambda x, y: np.min([(f(y) * g(x)) / (f(x) * g(y)), 1])
    xt = 0
    X = [] # empty Markov chain

    for t in range(1, numSamples):
        # draw a proposal
        y = -math.log(np.random.uniform(low = 0.0, high = 1.0)) * (2 * (np.random.uniform(low = 0.0, high = 1.0) < 0.5) - 1)
        if np.random.uniform(low = 0.0, high = 1.0) < alpha(xt, y):
            xt = y
        X.append(xt)

    return X



def main():
    """
    This function runs the main loop for the simulation.
    """

    X = independence_sampler(numSamples=10000)
    print(X)

    return X





if __name__ == "__main__":
    main()