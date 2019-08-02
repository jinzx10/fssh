from numpy import loadtxt
import matplotlib.pyplot as plt

eigval = loadtxt("eigval.txt")
plt.plot(eigval[:,0], eigval[:,1])
plt.plot(eigval[:,0], eigval[:,2])

plt.show()
