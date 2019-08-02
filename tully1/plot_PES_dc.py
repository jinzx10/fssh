import numpy as np
import matplotlib.pyplot as plt

eigval = np.loadtxt("eigval.txt")
dc_real = np.loadtxt("dc_real.txt")
dc_imag = np.loadtxt("dc_imag.txt")

plt.plot(eigval[:,0], eigval[:,1])
plt.plot(eigval[:,0], eigval[:,2])
plt.plot(dc_real[:,0], dc_real[:,1]/50)
plt.plot(dc_imag[:,0], dc_imag[:,1])

plt.show()
