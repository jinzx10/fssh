import numpy as np
import matplotlib.pyplot as plt
import h5py

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

class FSSH1990():

    def __init__(self, model, mass, dtc, max_ntc):
        self.model = model
        self.mass = mass
        self.dtc = dtc          # dt for a classical time step
        self.max_ntc = max_ntc  # maximum number of classical time steps
        self.reset()

    def reset(self):
        self.x = None
        self.v = None
        self.state = None
        self.rho = None

        self.rcq = None         # number of quantum steps within a classical step, must be an integer
        self.dtq = None         # dt for a quantum step
        self.has_hop = False    # 
        self.H_elec = None      # electronic Hamiltonian
        self.T_drvcpl = None    # time-derivative coupling

        # counter for classical steps
        self.itc = 0

        # trajectory histories
        self.x_t = np.zeros(max_ntc)
        self.v_t = np.zeros(max_ntc)
        self.E_t = np.zeros(max_ntc)
        self.state_t = np.zeros(max_ntc, dtype=int)
        self.num_frustrated_hops = 0

    def propagate(self, state0, x0, v0, rho0):
        # initialize a trajectory
        self.reset()
        self.x = x0
        self.v = v0
        self.state = state0
        self.rho = rho0

        self.update_prop()
        self.collect()

        # propagate
        for self.itc in range(1, self.max_ntc):
            self.evolve_nucl()
            self.update_prop()
            self.calc_dtq()

            '''
            self.has_hop = False
            for iq in range(0, self.rcq):
                self.evolve_elec()
                if not self.has_hop:
                    self.hop()
            '''
            #################################
            # test-1
            for iq in range(0, self.rcq):
                self.evolve_elec()
                self.hop()
            # test-2 would be to use dtc instead of dtq and keep has_hop check
            #################################

            self.collect()

            if self.model.terminate(self.x):
                # trajectory finishes wihtin max_ntc steps
                np.resize(self.x_t, self.itc+1)
                np.resize(self.v_t, self.itc+1)
                np.resize(self.E_t, self.itc+1)
                np.resize(self.state_t, self.itc+1)
                return 0

        # trajectory does not terminate within max_ntc steps
        return 1

    def collect(self):
        self.state_t[self.itc] = self.state
        self.x_t[self.itc] = self.x
        self.v_t[self.itc] = self.v
        self.E_t[self.itc] = self.tot_energy()

    def evolve_nucl(self):
        # velocity-Verlet
        F = self.model.force(self.x, self.state)
        a = F / self.mass
        self.x += self.v * self.dtc + 0.5 * a * self.dtc * self.dtc
        F = self.model.force(self.x, self.state)
        a_new = F / self.mass
        self.v += 0.5 * (a + a_new) * self.dtc

    def evolve_elec(self):
        # 4th-order Runge-Kutta
        k1 = self.dtq * self.drho_dt(self.rho)
        k2 = self.dtq * self.drho_dt(self.rho + 0.5*k1)
        k3 = self.dtq * self.drho_dt(self.rho + 0.5*k2)
        k4 = self.dtq * self.drho_dt(self.rho + k3)
        self.rho += (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    def update_prop(self):
        # update time-derivative coupling & electronic Hamiltonian
        self.H_elec = self.model.H_elec(self.x)
        self.T_drvcpl = self.v * self.model.drvcpl(self.x)

    def drho_dt(self, rho_tmp):
        # drho/dt = (-i)[H-iT, rho]
        H_tmp = self.H_elec - 1j*self.T_drvcpl
        return -1j * (H_tmp @ rho_tmp - rho_tmp @ H_tmp)

    def calc_dtq(self):
        #dtq1 = 0.02 / np.max(np.abs( self.T_drvcpl ))
        #
        #e, _ = np.linalg.eigh(self.H_elec)
        #dtq2 = 0.02 / np.max(np.abs( e - np.mean(e) ))

        #raw_dtq = min(self.dtc, dtq1, dtq2)
        #self.rcq = int(self.dtc/raw_dtq)
        #self.dtq = self.dtc/self.rcq if self.rcq > 1 else self.dtc

        self.rcq = 1
        self.dtq = self.dtc

    def tot_energy(self):
        return 0.5*self.mass*self.v*self.v + self.H_elec[self.state,self.state]

    def hop(self):
        # (d/dt)rho_mm = -\sum_l g_lm
        # g is antisymmetric
        # g_lm = 2*Re(T_ml*rho_lm)
        g = 2.0 * np.real(self.T_drvcpl[self.state,:] * self.rho[:,self.state])
        occ = np.real(self.rho[self.state,self.state])

        # hopping probability to each state 
        P_hop = self.dtq * g * (g>0) / occ

        # determne the final state
        P_cumu = np.cumsum(P_hop)
        final_state = 0

        r = np.random.rand()
        for final_state in range(0, self.model.sz_elec):
            if r < P_cumu[final_state]:
                break
            if final_state == self.model.sz_elec-1: # no hopping happens
                return

        # hopping may happen, check whether frustrated or not
        dE = self.H_elec[final_state,final_state] - self.H_elec[self.state,self.state]
        if dE < 0.5 * self.mass * self.v * self.v:
            # successfull hop
            v_sign = 1 if self.v > 0 else -1
            self.v = v_sign * np.sqrt(self.v*self.v - 2.0*dE/self.mass)
            self.state = final_state
            self.has_hop = True
        else:
            self.num_frustrated_hops += 1



class Tully1():
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0

    sz_elec = 2

    def terminate(self, x):
        return True if (x > 5 or x < -5) else False

    ###############################################################
    #                   common interface
    ###############################################################
    # H_elec, elec_energy, force & drvcpl are the quantities in the same representation
    # where FSSH is used (not necessarily adiabatic representation)

    # electronic Hamiltonian
    def H_elec(self, x):
        v0 = self.V00(x)
        v1 = self.V11(x)
        e0 = 0.5*(v0+v1) - np.sqrt( (0.5*(v0-v1))**2 + self.V01(x)**2 )
        e1 = 0.5*(v0+v1) + np.sqrt( (0.5*(v0-v1))**2 + self.V01(x)**2 )
        return np.array([[e0, 0], [0,e1]])

    # electronic energy (diagonal elements of H_elec)
    def elec_energy(self, x, state):
        v0 = self.V00(x)
        v1 = self.V11(x)
        if state == 0:
            return 0.5*(v0+v1) - np.sqrt( (0.5*(v0-v1))**2 + self.V01(x)**2 )
        elif state == 1:
            return 0.5*(v0+v1) + np.sqrt( (0.5*(v0-v1))**2 + self.V01(x)**2 )
        else:
            print('energy: state error!')
            return None

    # negative gradient of elec_energy
    def force(self, x, state):
        v0 = self.V00(x)
        v1 = self.V11(x)
        w = self.V01(x)
        dv0 = self.dV00(x)
        dv1 = self.dV11(x)
        dw = self.dV01(x)
        if state == 0:
            return -0.5*(dv0+dv1) + 1. / np.sqrt( (0.5*(v0-v1))**2 + w*w ) * (0.25*(v0-v1)*(dv0-dv1) + w*dw)
        elif state == 1:
            return -0.5*(dv0+dv1) - 1. / np.sqrt( (0.5*(v0-v1))**2 + w*w ) * (0.25*(v0-v1)*(dv0-dv1) + w*dw)
        else:
            print('energy: state error!')
            return None

    # derivative coupling
    def drvcpl(self, x):
        v0 = self.V00(x)
        v1 = self.V11(x)
        w = self.V01(x)
        dc01 = (1 if w > 0 else -1 ) * ( w*(self.dV00(x)-self.dV11(x)) - (v0-v1)*self.dV01(x) ) / ( (v0-v1)**2 + 4*w*w )
        return np.array([[0,dc01], [-dc01, 0]])

    ###############################################################
    #                   internal functions
    ###############################################################
    def V00(self, x):
        return self.A*(1.0-np.exp(-self.B*x)) if x > 0 else -self.A*(1.0-np.exp(self.B*x))

    def V11(self, x):
        return -self.V00(x)

    def V01(self, x):
        return self.C*np.exp(-self.D*x*x)


    def dV00(self, x):
        return self.A*self.B*np.exp((-self.B if x>0 else self.B)*x)
    
    def dV11(self, x):
        return -self.dV00(x)
    
    def dV01(self, x):
        return -2.0*self.D*x*self.V01(x)


###############################################################
#                   main starts
###############################################################

mode = 'MODE'
model = Tully1()

'''
nx = 500
x = np.linspace(-10,10,nx)

dc = np.zeros(nx)
e0 = np.zeros(nx)
e1 = np.zeros(nx)
f0 = np.zeros(nx)
f1 = np.zeros(nx)
for ix in range(nx):
    xi = x[ix]
    dc[ix] = model.drvcpl(xi)[0,1]
    e0[ix] = model.elec_energy(xi,0)
    e1[ix] = model.elec_energy(xi,1)
    f0[ix] = model.force(xi, 0)
    f1[ix] = model.force(xi, 1)

plt.plot(x, e0, color='C0')
plt.plot(x, e1, color='C1')
plt.plot(x, f0, color='C0', linestyle=':')
plt.plot(x, f1, color='C1', linestyle=':')
plt.plot(x, dc/50, color='C2')

plt.show()
exit()
'''

mass = 2000
max_ntc = MAX_NTC if mode == 'production' else 10000

n_trajs = NUM_TRAJS if mode == 'production' else 10
n_trajs_local = int(n_trajs/nprocs)
rem = n_trajs % nprocs
if rank < rem:
    n_trajs_local += 1

# use v0-dependent dtc later
sh = FSSH1990(model, mass, 1.0, max_ntc)

klist = [3, 3.5, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, \
        5.5, 6, 6.5, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, \
        8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, \
        9, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, \
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

klist = [20,25,30]

nk = len(klist)

r0_k = np.zeros(nk)
r1_k = np.zeros(nk)
t0_k = np.zeros(nk)
t1_k = np.zeros(nk)

np.random.seed()

MPI.Wtime()

for ik in range(0, nk):
    k = klist[ik]
    sh.dtc = 10.0 / k
    v0 = k / mass
    x0 = -4.95
    rho0 = np.array([[1.0,0.0],[0.0,0.0]], dtype=complex)

    # r: reflection; t: transmission
    r0_local = 0
    r1_local = 0
    t0_local = 0
    t1_local = 0

    for it in range(0, n_trajs_local):
        flag = sh.propagate(state0=0, x0=x0, v0=v0, rho0=rho0)
        if flag == 0: # successfully terminate
            if sh.x > 5: # transmission
                if sh.state == 0:
                    t0_local += 1
                else:
                    t1_local += 1
            else:
                if sh.state == 0:
                    r0_local += 1
                else:
                    r1_local += 1
        else: # does not terminate within max_ntc
            print('trajectory does not terminate!')

    t0_k[ik] = comm.reduce(t0_local, op=MPI.SUM, root=0)
    t1_k[ik] = comm.reduce(t1_local, op=MPI.SUM, root=0)
    r0_k[ik] = comm.reduce(r0_local, op=MPI.SUM, root=0)
    r1_k[ik] = comm.reduce(r1_local, op=MPI.SUM, root=0)


    if rank == 0:
        print('k = %4.1f finished, time elapsed = %8.1f seconds'%(k, MPI.Wtime()))

    comm.Barrier()

if rank == 0:
    t0_k /= n_trajs
    t1_k /= n_trajs
    r0_k /= n_trajs
    r1_k /= n_trajs

    fh = h5py.File('tully1_var1.h5', 'w')
    fh['k'] = klist
    fh['t0'] = t0_k
    fh['t1'] = t1_k
    fh['r0'] = r0_k
    fh['r1'] = r1_k
    fh.close()

MPI.Finalize()




