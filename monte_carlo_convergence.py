from abcpy.output import Journal
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
plt.rc('font', size=14) 

filename = 'standard_experiments%d_n=%d.jnl'
# filename2 = 'standard_experiments4_n=%d.jnl'
resolutions = [1000, 2000, 5000, 10000, 15000, 20000]
# resolutions = [100, 1000, 5000, 10000, 20000,30000,40000]

statistics = np.array([
    [94.81808701,  5.14334496,  1.84792104, -1.46720583],
    [96.70280141,  2.62173462,  0.77414173,  0.57861121],
    [97.83363005,  3.22756218,  1.08080703, -0.48925518],
    [98.21057293,  3.7612552,   1.19455838, -0.73564671],
    [98.39904437,  4.0607704,   1.25760627, -0.82598126],
    [98.58751581,  4.37621463,  1.32851105, -0.89538125]
])

l2s = []
for i in range(6):
    l2s.append(np.mean(np.square(statistics[i]-statistics[-1])))

print(l2s)

def h_func(theta):
    return theta


def h_exp(theta):
    return np.exp(theta)


for j,i in enumerate([4,5,6,7,8,3]):
    mean_hs = []
    for resolution in resolutions:
        journal = Journal.fromFile(filename % (i, resolution))
        # print(journal.get_accepted_parameters())

        mus = [item[0][0] for item in journal.get_accepted_parameters()]
        mean_hs.append(np.mean(mus))
        

    plt.plot(resolutions, mean_hs, label='$d = %.3f$' % l2s[j])

plt.legend()
plt.xlabel('number of samples $N$')
plt.ylabel('$h(\\theta)$ = $\hat{\mu}$')
plt.tight_layout()
# plt.title('Monte Carlo convergence for mean $\mu$')
plt.savefig('monte_carlo_convergence.png', dpi=400)
# plt.show()
plt.close()


for j,i in enumerate([4,5,6,7,8,3]):
    mean_hs = []
    for resolution in resolutions:
        journal = Journal.fromFile(filename % (i, resolution))
        # print(journal.get_accepted_parameters())

        mus = np.array([item[0][0] for item in journal.get_accepted_parameters()])
        mean_hs.append(np.mean(np.exp(mus)))
        

    plt.plot(resolutions, mean_hs, label='$d = %.3f$' % l2s[j])

plt.legend()
plt.xlabel('number of samples $N$')
plt.ylabel('mean exp($\mu$)')
plt.tight_layout()
# plt.title('Monte Carlo convergence for mean exp $\mu$')
plt.savefig('monte_carlo_exp_convergence.png', dpi=400)
# plt.show()
plt.close()

    
markers = ['o', 'v', 'x', 's', '*']
for j, resolution in enumerate(resolutions):
    mean_hs = []
    for i in [4,5,6,7,8,3]:
        journal = Journal.fromFile(filename % (i, resolution))
        # print(journal.get_accepted_parameters())

        mus = [item[0][0] for item in journal.get_accepted_parameters()]
        mean_hs.append(np.mean(mus))
    mean_hs = np.array(mean_hs)
    mean_hs = np.abs(mean_hs-mean_hs[-1])
    # print(mean_hs)

    plt.loglog(l2s, mean_hs, label='$N = %d$' % resolution)

plt.legend()
plt.xlabel('Euclidean distance')
plt.ylabel('|$\hat{\mu} - \hat{\mu}_0$|')
plt.tight_layout()
plt.grid()
# plt.title('Monte Carlo convergence for mean $\mu$')
plt.savefig('monte_carlo_convergence_l2_loglog.png', dpi=400)
# plt.show()
plt.close()

markers = ['o', 'v', 'x', 's', '*']
for j, resolution in enumerate(resolutions):
    mean_hs = []
    for i in [4,5,6,7,8,3]:
        journal = Journal.fromFile(filename % (i, resolution))
        # print(journal.get_accepted_parameters())

        mus = [item[0][0] for item in journal.get_accepted_parameters()]
        mean_hs.append(np.mean(mus))
    mean_hs = np.array(mean_hs)
    mean_hs = np.abs(mean_hs-mean_hs[-1])
    # print(mean_hs)

    plt.loglog(l2s, mean_hs, label='$N = %d$' % resolution)

plt.legend()
plt.xlabel('Euclidean distance')
plt.ylabel('|$\hat{\mu} - \hat{\mu}_0$|')
plt.xlim([-0.2, 0.8])
plt.ylim([-0.004, 0.024])
plt.tight_layout()
plt.grid()
# plt.title('Monte Carlo convergence for mean $\mu$')
plt.savefig('monte_carlo_convergence_l2_close_up_loglog.png', dpi=400)
# plt.show()
plt.close()