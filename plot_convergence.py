from abcpy.output import Journal
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
plt.rc('font', size=14) 

filename = 'standard_experiments3_n=%d.jnl'
resolutions = [1000, 2000, 5000, 10000, 20000, 30000, 40000]
# resolutions = [100, 1000, 5000, 10000, 20000,30000,40000]

ranges = [
    (-0.6, 0.6),
    (0, 20),
    (0,0.12),
    (-1,1),
    (0,5.5)
]
All_set = []
for i, label in enumerate(['mu', 'kappa', 'sigma', 'rho', 'xi']):
    plt.figure()
    theta_set = []
    for resolution in resolutions:
        journal = Journal.fromFile(filename % resolution)
        # print(journal.get_accepted_parameters())

        thetas = [item[i][0] for item in journal.get_accepted_parameters()]
        theta_set.append(thetas)
        kernel = stats.gaussian_kde(thetas)

        X = np.linspace(ranges[i][0], ranges[i][1], 200)
        Y = kernel(X)
        plt.plot(X,Y, label='N=%d' % resolution)
    All_set.append(theta_set)


    plt.legend()
    plt.xlabel('$\%s$' % label)
    plt.ylabel('$\pi_{N,\\varepsilon}$')
    plt.grid()
    plt.tight_layout()
    # plt.title('Posterior plots for case 3')
    plt.savefig('posteriors_case_3_%s.png' % label)
    plt.close()
    # exit(0)


def hell_distance(y0, y1, a, b, n):

    eval_points = np.linspace(a, b, n)
    
    # Evaluate pdf of Gaussian kde
    p0 = stats.gaussian_kde(y0)(eval_points)
    p1 = stats.gaussian_kde(y1)(eval_points)
    # Evaluate continuous hell distance
    step = (b-a)/(n-1)
    d = continuous_hellinger(p0, p1, step) - 1 + 0.5 * (trapezoidal(eval_points, p0) + trapezoidal(eval_points, p1))
    return d

def continuous_hellinger(f,g, step):
    """
    Given equally spaced values of pdf f,g, calculate Hellinger distance using Trapezoidal rule.
    """
    #print(f-g)
    sq_fg = np.sqrt(f*g)
    return 1 - step*sum(sq_fg[1:-1])-step*(1/2)*(sq_fg[0]+sq_fg[-1])


def l2_distance(y0, y1, a, b, n):

    eval_points = np.linspace(a, b, n)
    
    # Evaluate pdf of Gaussian kde
    p0 = stats.gaussian_kde(y0)(eval_points)
    p1 = stats.gaussian_kde(y1)(eval_points)
    # Evaluate continuous hell distance
    step = (b-a)/(n-1)
    p = np.power(p1-p0,2)
    d = np.sqrt(np.sum((p[1:]+p[:-1]/2))*step)
    return d

def trapezoidal(x,y):
    n = len(x)
    integral = (np.dot( (x[1:]-x[:-1]) , (y[1:]+y[:-1]) ))/2
    return integral

def monte_carlo(y0, y1, h, squeeze = True):
    if squeeze:
        y0 = np.array(np.squeeze(y0))
        y1 = np.array(np.squeeze(y1))
    n = y0.shape[0]
    return np.abs(1/n * (np.sum(1/y0) - np.sum(1/y1)))

def h(x, c):
    return np.exp(-(x-c)**2) 

all_D = []
for j, label in enumerate(['mu', 'kappa', 'sigma', 'rho', 'xi']):
    D = []
    for i, resolution in enumerate(resolutions):
        D.append(hell_distance(All_set[j][i], All_set[j][-1], -1, 1, 400))
    all_D.append(D)


for j, label in enumerate(['mu', 'kappa', 'sigma', 'rho', 'xi']):
    plt.loglog(resolutions[:-1], all_D[j][:-1], label='$\%s$' % label)
plt.xlabel('number of samples $N$')
plt.ylabel('hellinger distance')
plt.grid()
plt.legend()
plt.tight_layout()
# plt.title('Convergence study case 3')
plt.savefig('convergence_case_3_log.png', dpi=400)
# plt.show()
plt.close()

for j, label in enumerate(['mu', 'kappa', 'sigma', 'rho', 'xi']):
    plt.plot(resolutions, all_D[j], label='$\%s$' % label)
plt.xlabel('number of samples $N$')
plt.ylabel('hellinger distance')
plt.grid()
plt.legend()
plt.tight_layout()
# plt.title('Convergence study case 3')
plt.savefig('convergence_case_3.png', dpi=400)
# plt.show()
plt.close()


all_D = []
for j, label in enumerate(['mu', 'kappa', 'sigma', 'rho', 'xi']):
    D = []
    for i, resolution in enumerate(resolutions):
        D.append(l2_distance(All_set[j][i], All_set[j][-1], -1, 1, 400))
    all_D.append(D)


for j, label in enumerate(['mu', 'kappa', 'sigma', 'rho', 'xi']):
    plt.loglog(resolutions[:-1], all_D[j][:-1], label='$\%s$' % label)
plt.xlabel('number of samples $N$')
plt.ylabel('$L_2$ distance')
plt.grid()
plt.legend()
plt.tight_layout()
# plt.title('Convergence study case 3')
plt.savefig('convergence_case_3_log_l2.png', dpi=400)
# plt.show()
plt.close()

for j, label in enumerate(['mu', 'kappa', 'sigma', 'rho', 'xi']):
    plt.plot(resolutions, all_D[j], label='$\%s$' % label)
plt.xlabel('number of samples $N$')
plt.ylabel('$L_2$ distance')
plt.grid()
plt.legend()
plt.tight_layout()
# plt.title('Convergence study case 3')
plt.savefig('convergence_case_3_l2.png', dpi=400)
# plt.show()
plt.close()