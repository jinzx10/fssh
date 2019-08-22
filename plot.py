import numpy as np
import matplotlib.pyplot as plt

E = np.loadtxt('E.txt')
plt.plot(E[:,0], E[:,1])
#plt.plot(E[:,0], E[:,2])

drvcpl = np.loadtxt('drvcpl.txt')
plt.plot(drvcpl[:,0], drvcpl[:,1]/50)
#plt.plot(drvcpl[:,0], drvcpl[:,2]/50)

berry = np.loadtxt('berry.txt')
plt.plot(berry[:,0], berry[:,1])
#plt.plot(berry[:,0], berry[:,2])

F = np.loadtxt('F.txt')
plt.plot(F[:,0], F[:,1])
#plt.plot(F[:,0], F[:,2])

plt.show()

