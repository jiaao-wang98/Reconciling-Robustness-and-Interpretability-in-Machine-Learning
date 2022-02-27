"""This Python file contains functions which generate
price graphs from Heston model.
"""
import numpy as np


def generate_heston_paths(S_0, T, r, kappa, theta, v_0, rho, xi,
                          steps, Npaths, return_vol=False):
    """Generate Heston path.

    This function generates a number of Heston paths for a given
    set of Heston model parameters.

    Parameters
    ----------
        Si_0 (float): initial price
        T : float
            total running time
        r : float
            risk free rate / drift term
        kappa : float
            rate of mean reversion
        theta : float
            long term average volatility
        v_0 : float
            initial volatility
        rho : float
            correlation of Wiener processes
        xi : float
            volatility of volatility
        steps : int
            number of time steps performed
        Npaths : int
            number of Heston paths returned
        return_vol : bool
            whether to return volatility (default False)


    Returns
    -------
    np.ndarray
        Npaths * steps dimensional array with sample Heston paths in each row
    np.ndarray (if return_vol is True)
        Npaths * steps dimensional array with sample Heston paths' volatilities
        in each row

    """
    dt = T / steps
    size = (Npaths, steps)
    prices = np.zeros(size)
    sigs = np.zeros(size)
    S_t = S_0
    v_t = v_0
    for t in range(steps):
        WT = np.random.multivariate_normal(np.array([0, 0]),
                                           cov=np.array([[1, rho],
                                                         [rho, 1]]),
                                           size=Npaths) * np.sqrt(dt)

        S_t = S_t * (np.exp((r - 0.5 * v_t) * dt + np.sqrt(v_t) * WT[:, 0]))
        v_t = np.abs(v_t + kappa * (theta - v_t) *
                     dt + xi * np.sqrt(v_t) * WT[:, 1])
        prices[:, t] = S_t
        sigs[:, t] = v_t

    if return_vol:
        return prices, sigs

    return prices


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(1)

    kappa = 100
    theta = 0.0001
    v_0 = 0.0001
    xi = 0.0001
    r = 0.05
    S = 100
    paths = 3
    steps = 2000
    T = 1
    rho = 0
    prices, sigs = generate_heston_paths(S, T, r, kappa, theta,
                                         v_0, rho, xi, steps, paths,
                                         return_vol=True)

    plt.figure(figsize=(7, 6))
    # ax = plt.axes()
    # ax.set_facecolor("#fffffa")
    plt.plot(prices.T)
    plt.title('Heston Price Paths Simulation')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.grid()
    
    plt.show()

    plt.figure(figsize=(7, 6))
    plt.plot(np.sqrt(sigs).T)
    plt.axhline(np.sqrt(theta), color='black', label=r'$\sqrt{\theta}$')
    plt.title('Heston Stochastic Vol Simulation')
    plt.xlabel('Time Steps')
    plt.ylabel('Volatility')
    plt.legend(fontsize=15)
    plt.show()
