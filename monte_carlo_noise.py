from abcpy.output import Journal
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
plt.rc('font', size=14) 


filename = 'standard_experiments3_n=%d.jnl'
all_mus = []
for resolution in [1000,2000,4000,5000,10000,12000,15000,20000,30000,40000]:
    journal = Journal.fromFile(filename % resolution)
    # print(journal.get_accepted_parameters())

    mus = [item[0][0] for item in journal.get_accepted_parameters()]
    all_mus.extend(mus)



def h_func(theta):
    return theta


def h_exp(theta):
    return np.exp(theta)




print(len(all_mus))
resolutions = [100,200,500,1000,2000,4000]
stds = []
vars = []
plt.figure(figsize=(9,5))
for resolution in resolutions:
    mean_hs = []
    for i in range(40000//resolution):
        mean_hs.append(np.mean(all_mus[i*resolution:(i+1)*resolution]))
    mean_hs = mean_hs[:35]
    stds.append(np.std(mean_hs))
    vars.append(np.var(mean_hs))
    plt.hist(mean_hs, bins=15, alpha=0.5, range=(-0.04, 0.04), label='N = %d' % resolution)
plt.legend()
plt.xlabel('$\hat{\mu}_0$')
plt.tight_layout()
plt.savefig('histogram_monte_carlo.png', dpi=400)
# plt.show()
plt.close()


resolutions = [100,200,500,1000,2000,4000]
stds = []
vars = []
fig, axs = plt.subplots(3, 2, sharex=True, figsize=(9,10))
for j, resolution in enumerate(resolutions):
    mean_hs = []
    for i in range(139000//resolution):
        mean_hs.append(np.mean(all_mus[i*resolution:(i+1)*resolution]))
    mean_hs = mean_hs[:34]
    stds.append(np.std(mean_hs))
    vars.append(np.var(mean_hs))
    axs[j%3, j//3].hist(mean_hs, bins=25, range=(-0.04, 0.04))
    axs[j%3, j//3].set_title('number of samples $N$ = %d' % resolution)
    axs[j%3, j//3].set_ylim([0,17])
# plt.legend()
axs[2,0].set_xlabel('$\hat{\mu}_0$')
axs[2,1].set_xlabel('$\hat{\mu}_0$')
plt.tight_layout()
plt.savefig('histograms_monte_carlo.png', dpi=400)
# plt.show()
plt.close()






fig, ax1 = plt.subplots() 
  
ax1.set_xlabel('resolution') 
ax1.set_ylabel('std') 
ax1.plot(resolutions, stds) 
ax1.tick_params(axis ='y') 
  
# Adding Twin Axes

ax2 = ax1.twinx() 
  
ax2.set_ylabel('var') 
ax2.plot(resolutions, vars, ':') 
ax2.tick_params(axis ='y') 
 
# Show plot

# plt.show()
plt.close()

trend = 1/np.sqrt(np.array(resolutions))
plt.loglog(resolutions, stds, label='std')
plt.loglog(resolutions, trend, label='$1/\sqrt{N}$')
# plt.loglog(resolutions, vars, label='var')
plt.grid(True)
plt.xlabel('number of samples $N$')
plt.legend()
plt.tight_layout()
plt.savefig('variance_convergence')
plt.show()
plt.close()

for i in range(len(vars)):
    print(resolutions[i], ' & ', '{:.2e}'.format(vars[i]), ' & ', '{:.2e}'.format(stds[i]), '\\\\')