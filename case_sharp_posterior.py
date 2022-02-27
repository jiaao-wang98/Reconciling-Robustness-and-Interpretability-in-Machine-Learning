import logging
from numbers import Number
import statistics

from sklearn.metrics import euclidean_distances
import heston
import numpy as np
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, Discrete, InputConnector
from abcpy.statistics import Statistics

from heston_model import Heston, HestonStatistics, setup_backend


def infer_parameters(backend, steps=100, n_sample=500,
                     n_samples_per_param=100, logging_level=logging.INFO):
    """Perform inference for this example.
    Parameters
    ----------
    steps : integer, optional
        Number of iterations in the sequential PMCABC algoritm ("generations"). The default value is 3
    n_samples : integer, optional
        Number of posterior samples to generate. The default value is 250.
    n_samples_per_param : integer, optional
        Number of data points in each simulated data set. The default value is 10.
    Returns
    -------
    abcpy.output.Journal
        A journal containing simulation results, metadata and optionally intermediate results.
    """
    logging.basicConfig(level=logging_level)

    # define prior
    from abcpy.continuousmodels import Uniform
    kappa = Uniform([[0], [500]], name="kappa")
    theta = Uniform([[0], [0.0005]], name="theta")
    xi = Uniform([[0], [0.0005]], name='xi')
    r = Uniform([[0], [0.2]], name='r')
    rho = Uniform([[-0.05], [0.05]], name='rho')

    # define the model
    from abcpy.continuousmodels import Normal as Gaussian
    steps = 500
    heston_graph = Heston([r, kappa, theta, rho, xi], 100, 1,
                    0.0001, steps, False, name='heston')
    print(heston_graph.heston_sim)

    graph_obs = heston_graph.forward_simulate([0.05, 100, 0.0001, 0, 0.0001], 1)

    statistics_calculator = HestonStatistics()

    stats = statistics_calculator.statistics(graph_obs)

    from abcpy.distances import  Euclidean
    distance_calculator = Euclidean(statistics_calculator)


    # define kernel
    from abcpy.perturbationkernel import DefaultKernel
    kernel = DefaultKernel([kappa, theta, xi, r, rho])

    # define sampling scheme
    from abcpy.inferences import PMCABC, RejectionABC
    sampler = PMCABC([heston_graph], [distance_calculator], backend, seed=1)

    # sample from scheme
    journal = sampler.sample(
        [graph_obs],
        steps=4,
        n_samples=200,
        n_samples_per_param=1,
        epsilon_init=[100.],
        epsilon_percentile=30
    )

    return journal


def analyse_journal(journal):
    # output parameters and weights
    print(journal.get_accepted_parameters())
    print(journal.get_weights())

    # do post analysis
    print(journal.posterior_mean())
    print(journal.posterior_cov())

    # print configuration
    print(journal.configuration)

    # plot posterior
    journal.plot_posterior_distr(path_to_save="sharp_posterior.png")

    # save and load journal
    journal.save("sharp_posterior_experiments.jnl")

    from abcpy.output import Journal
    new_journal = Journal.fromFile('sharp_posterior_experiments.jnl')


if __name__ == "__main__":
    backend = setup_backend(True, process_per_model=4)
    journal = infer_parameters(backend, logging_level=logging.DEBUG)
    analyse_journal(journal)
