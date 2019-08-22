import numpy as np
import matplotlib.pyplot as plt

E = np.loadtxt('E.txt')
plt.plot(E[:,0], E[:,1])
plt.plot(E[:,0], E[:,2])

drvcpl = np.loadtxt('drvcpl.txt')
plt.plot(drvcpl[:,0], -drvcpl[:,1]/50)

plt.show()

