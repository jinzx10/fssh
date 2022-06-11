import h5py
import numpy as np
import matplotlib.pyplot as plt

fh = h5py.File('tully1_var1.h5', 'r')
k = np.asarray(fh['k'])
r0 = np.asarray(fh['r0']) 
r1 = np.asarray(fh['r1'])
t0 = np.asarray(fh['t0'])
t1 = np.asarray(fh['t1'])

fig, (ax1,ax2,ax3) = plt.subplots(1,3)

ax1.plot(k, r0)
ax1.set_title('r0')

ax2.plot(k, t0)
ax2.set_title('t0')

ax3.plot(k, t1)
ax3.set_title('t1')

plt.show()

