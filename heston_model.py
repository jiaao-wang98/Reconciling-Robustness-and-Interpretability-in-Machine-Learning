import logging
from numbers import Number
import statistics

from sklearn.metrics import euclidean_distances
import heston
import numpy as np
from functools import partial
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, Discrete, InputConnector
from abcpy.statistics import Statistics

def setup_backend(mpi=False, process_per_model=1):
    from abcpy.backends import BackendMPI 
    from abcpy.backends import BackendDummy 

    if mpi:
        backend = BackendMPI(process_per_model)
    else:
        backend = BackendDummy()

    return backend

def moving_average(a, n=3) :
    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:,n:] = ret[:,n:] - ret[:,:-n]
    return ret[:,n - 1:] / n

class HestonStatistics(Statistics):
    """Statistics for Heston model.
    """

    def __init__(self, degree=1, cross=False, reference_simulations=None, previous_statistics=None, window_size=100):
        self.window_size = window_size
        super().__init__(degree, cross, reference_simulations, previous_statistics)


    def statistics(self, data):
        """
        Parameters
        ----------
        data: python list
            Contains n data sets with length p.
        Returns
        -------
        numpy.ndarray
            nx(d+degree*d+cross*nchoosek(d,2)) matrix where for each of the n data points with length p you apply the
            linear transformation to get to dimension d, from where (d+degree*d+cross*nchoosek(d,2)) statistics are
            calculated.
        """

        # need to call this first which takes care of calling the previous statistics if that is defined and of properly
        # formatting data
        data = self._preprocess(data)


        # Calculate the statistics
        means = np.mean(data, axis=1, keepdims=True)
        dev = np.power(data-means, 2)
        mean_dev = np.sqrt(np.mean(dev, axis=1, keepdims=True))
        dev_window = np.sqrt(moving_average(dev, n=self.window_size))
        dev_dev = np.power(dev_window - mean_dev, 2)
        mean_dev_dev = np.sqrt(np.mean(dev_dev, axis=1, keepdims=True))
        trunc_data = data[:,self.window_size-1:]
        corr = np.mean((trunc_data-means)*(dev_window-mean_dev), axis=1, keepdims=True)


        data = np.hstack((means, mean_dev, mean_dev_dev, corr))


        # Expand the data with polynomial expansion
        result = self._polynomial_expansion(data)

        # now call the _rescale function which automatically rescales the different statistics using the standard
        # deviation of them on the training set provided at initialization.
        result = self._rescale(result)

        return result

class Heston(ProbabilisticModel):
    """
    """

    def __init__(self, parameters, S_0, T, v_0,
                 steps, return_vol, name='heston'):

        self.heston_sim = partial(heston.generate_heston_paths, S_0=S_0,
                                  T=T, v_0=v_0, steps=steps, return_vol=return_vol)

        if not isinstance(parameters, list):
            raise TypeError('Input of Heston model is of type list')

        if len(parameters) != 5:
            raise RuntimeError(
                'Input list must be of length 5, containing [...FILL...].')

        input_connector = InputConnector.from_list(parameters)
        super().__init__(input_connector, name)

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != 5:
            raise ValueError('Number of parameters of Heston model must be 5.')

        return True

    def _check_output(self, values):
        if not isinstance(values, Number):
            raise ValueError(
                'Output of the normal distribution is always a number.')

        # At this point values is a number (int, float); full domain for Normal
        # is allowed
        return True

    def get_output_dimension(self):
        return 1

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        # Extract the input parameters
        r = input_values[0]
        kappa = input_values[1]
        theta = input_values[2]
        rho = input_values[3]
        xi = input_values[4]

        # Do the actual forward simulation
        vector_of_k_samples = self.heston_sim(
            r=r, kappa=kappa, theta=theta, rho=rho, xi=xi, Npaths=k)

        # Format the output to obey API
        result = [x for x in vector_of_k_samples]
        return result


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
        steps=6,
        n_samples=300,
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
    journal.plot_posterior_distr(path_to_save="posterior.png")

    # save and load journal
    journal.save("experiments.jnl")

    from abcpy.output import Journal
    new_journal = Journal.fromFile('experiments.jnl')


# def setUpModule():
#     '''
#     If an exception is raised in a setUpModule then none of 
#     the tests in the module will be run. 
    
#     This is useful because the slaves run in a while loop on initialization
#     only responding to the master's commands and will never execute anything else.
#     On termination of master, the slaves call quit() that raises a SystemExit(). 
#     Because of the behaviour of setUpModule, it will not run any unit tests
#     for the slave and we now only need to write unit-tests from the master's 
#     point of view. 
#     '''
#     setup_backend()

if __name__ == "__main__":
    backend = setup_backend(True, process_per_model=4)
    journal = infer_parameters(backend, logging_level=logging.DEBUG)
    analyse_journal(journal)
