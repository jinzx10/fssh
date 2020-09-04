import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(1)
ax1 = fig.add_subplot(121)

x = np.linspace(-10,10,1000)
dx = x[2] - x[1]
E = np.loadtxt('E.txt')
ax1.plot(x, E[:,0])
ax1.plot(x, E[:,1])
#plt.plot(E[:,0], E[:,2])

dc01 = np.loadtxt('dc01.txt')
ax1.plot(x, dc01/50)
#plt.plot(drvcpl[:,0], drvcpl[:,2]/50)

ax2 = fig.add_subplot(122)
F = np.loadtxt('F.txt')
plt.plot(x, F[:,0])
plt.plot(x, F[:,1])
plt.plot(x[1:], -(E[1:,0]-E[0:-1,0])/dx)
plt.plot(x[1:], -(E[1:,1]-E[0:-1,1])/dx)

plt.show()

