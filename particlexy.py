import numpy as np
import os
from pyDOE import lhs
from math import atan2
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
import pickle

"""
Implement particle filter for estimating center of the target's circular movement
using noisy distance measurements.
"""
class SingleSenseParticle(object):

    def __init__(self, d, T, M, sensor_pos):
        """
        :d: State dimensions
        :T: Number of algorithm iterations
        :M: Number of particles in the filter 
        :sense_pos: Position of single sensor
        """
        self.d = d
        self.T = T 
        self.M = M
        self.dt = 0.5
        self.x_true = 1.
        self.y_true = 4.
        self.r_true = 2.
        self.theta0_true = 0.
        self.omega_true = 0.2
        self.sensor_pos = sensor_pos 
        
    def generate_particles(self):
        """
        Generate points for center (x, y) using Latin hypercube sampling
        """
        xyz_0 = lhs(self.d, int(self.M))
        xyz_0[:,0] = 10*xyz_0[:,0] - 2# Center x
        xyz_0[:,1] = 10*xyz_0[:,1] - 2# Center y
        return xyz_0

    def position(self, t, x, y):
        """
        Return position after one time step
        """
        # Theta at next time step
        theta = self.theta0_true + self.omega_true*t*self.dt
        noise = np.random.normal(0, 0.01, 2)
        # Position at next time step
        return (x+self.r_true*np.cos(theta)+noise[0], \
                y+self.r_true*np.sin(theta)+noise[1])

    def generate_true_target(self):
        """
        Generate sequence of true target positions
        """
        true_pos = np.zeros((self.T+1,self.d))
        true_pos[0,:] = (self.x_true+self.r_true*np.cos(self.theta0_true), 
                        self.y_true+self.r_true*np.sin(self.theta0_true))

        theta0_t = self.theta0_true
        for t in range(int(self.T)):
            (x1,x2) = self.position(t, self.x_true, self.y_true)
            true_pos[t+1,:] = (x1,x2)
        return true_pos

    def generate_observations(self):
        """
        Generate noisy relative distance observations between the sensor and the target
        """
        true_pos = self.generate_true_target()
        true_obs = np.zeros((self.T,1))
        dist = np.zeros((self.T,1))

        for t in range(T):
            dist[t,0] = np.linalg.norm(self.sensor_pos-true_pos[t,:])
            true_obs[t,0] = np.random.normal(dist[t,0], 0.3)
        return true_obs

    def single_sense(self):
        # Generate target positions and observations 
        true_pos = self.generate_true_target()
        true_obs = self.generate_observations()
        # Define particles 
        xyz = self.generate_particles()
        xyz_0 = self.generate_particles()
        
        estimated_mean = np.random.rand(self.T, self.d)
        estimated_cov = np.random.rand(self.T, self.d, self.d)
        # Define weights for particles 
        alfa = [1./self.M for i in range(int(self.M))]
        new_alfa = [1./self.M for i in range(int(self.M))]
        
        weight_ratio = [0. for i in range(self.T)]
        for t in range(self.T): #T
            # Update particle weights 
            for s_id in range(int(self.M)):
                target_pos = self.position(t, xyz[s_id,0], xyz[s_id,1])
                new_alfa[s_id] = norm.pdf(true_obs[t,0], np.linalg.norm(self.sensor_pos-target_pos), 0.5)
            alfa = new_alfa/np.sum(new_alfa)
            # Weight ratio is a proxy for setting T
            weight_ratio[t] = np.max(alfa)/np.min(alfa)
            
            # Stratified resampling
            j = 0
            c = alfa[0]
            for k in range(int(self.M)):
                u = 1./int(self.M)*np.random.rand(1)
                beta = u + k/self.M
                while beta > c:
                    j = j+1
                    c = c+alfa[j]
                new_alfa[k] = 1./self.M
                xyz_0[k,:] = xyz[j,:] + 0.02*np.random.normal(0,1,2)
            
            estimated_mean[t,:] = np.mean(xyz_0, axis=0)
            estimated_cov[t,:,:] = np.cov(xyz_0.T, bias=True)
            alfa = new_alfa
            xyz = xyz_0
            
            # Plot particle distributions after certain iterations 
            if (t%100 == 0):
                circle = plt.Circle((self.x_true, self.y_true), self.r_true, color='b', fill=False)
                print ('Iteration:', t, 'Weight ratio:', weight_ratio[t])
                fig, ax1 = plt.subplots()
                ax1.scatter(xyz[:,0], xyz[:,1])
                ax1.scatter(self.x_true, self.y_true, s=50, marker='^', label='Center')
                ax1.scatter(true_pos[t,0], true_pos[t,1], s=50, marker='x', label='Target location')
                ax1.scatter(self.sensor_pos[0], self.sensor_pos[1], s=50, marker='s', label='Sensor location')
                ax1.add_patch(circle)
                ax1.set_title('Estimate spread at time = %i'% int(t+1))
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_xlim([-2, 6])
                ax1.set_ylim([-2, 6])
                ax1.legend()
                ax1.legend([circle], ['Trajectory'])
        spfilter_ = { "xy": xyz, "emean": estimated_mean, "ecov":estimated_cov }
        pickle.dump( spfilter_, open( "spfilter.p", "wb" ) )
        return xyz, weight_ratio, estimated_mean, estimated_cov

class MultiSenseParticle(object):

    def __init__(self, d, T, M, sensor_pos):
        """
        :d: State dimensions
        :T: Number of algorithm iterations
        :M: Number of particles in the filter 
        :sense_pos: Position of sensors in the network
        """
        self.d = d
        self.T = T 
        self.M = M
        self.dt = 0.5
        self.x_true = 1.
        self.y_true = 4.
        self.r_true = 2.
        self.theta0_true = 0.
        self.omega_true = 0.2
        self.sensor_pos = sensor_pos
        self.n = np.shape(self.sensor_pos)[0]
        self.true_pos = self.generate_true_target()

    def generate_particles(self):
        """
        Generate points for center (x, y) using Latin hypercube sampling
        """
        xyz_0 = lhs(self.d, int(self.M))
        xyz_0[:,0] = 10*xyz_0[:,0]-2 # Center x
        xyz_0[:,1] = 10*xyz_0[:,1]-2 # Center y
        return xyz_0

    def position(self, t, x, y):
        """
        Return position after one time step
        """
        # Theta at next time step
        theta = self.theta0_true + self.omega_true*t*self.dt
        noise = np.random.normal(0, 0.01, 2)
        return (x+self.r_true*np.cos(theta)+noise[0], \
                y+self.r_true*np.sin(theta)+noise[1])

    def rand_network(self):
        """
        Generate a n-node network with arbitrary connections
        Represent the network as a row stochastic matrix
        """
        n = self.n
        A = np.eye(n)
        E_MIN = n-1
        E_MAX = 0.5*(n**2 - n)
        n_edges = np.random.randint(E_MIN, E_MAX)
        for i in range(n_edges):
            i1 = np.random.randint(1,n+1)-1
            i2 = np.random.randint(1,n+1)-1
            while (i2 == i1):
                i2 = np.random.randint(1,n)
            A[i1, i2] = 1
        A = A/np.sum(A, axis=1).reshape(-1,1)
        return A
        
    def generate_true_target(self):
        """
        Generate sequence of true target positions
        """
        true_pos = np.zeros((self.T+1,self.d))
        true_pos[0,:] = (self.x_true+self.r_true*np.cos(self.theta0_true), 
                        self.y_true+self.r_true*np.sin(self.theta0_true))

        theta0_t = self.theta0_true
        for t in range(int(self.T)):
            (x1,x2) = self.position(t, self.x_true, self.y_true)
            true_pos[t+1,:] = (x1,x2)
        return true_pos

    def generate_observations(self, t, sense_pos):
        """
        Generate noisy relative distance observations between each sensor and the target
        """
        dist = np.linalg.norm(self.sensor_pos-self.true_pos[t,:])
        true_obs = np.random.normal(dist, 0.3)
        return true_obs

    def multi_sense(self):
        # Generate agent-specific particles representing target position
        xyz = np.zeros((self.n, int(self.M), self.d))
        xyz_0 = np.zeros((self.n, int(self.M), self.d))
        for i in range(self.n):
            xyz[i, :,:] = self.generate_particles()
            xyz_0[i, :,:] = self.generate_particles()
        
        estimated_mean = np.random.rand(self.n, self.T, self.d)
        estimated_cov = np.random.rand(self.n, self.T, self.d, self.d)
        
        # Define particle weights 
        alfa = [[1./self.M for i in range(int(self.M))] for i in range(self.n)]
        new_alfa = [[1./self.M for i in range(int(self.M))] for i in range(self.n)]
        
        for t in range(self.T):
            At = self.rand_network()
            for i in range(self.n):
                true_obs = self.generate_observations(t, self.sensor_pos[i,:])
                # Update particle weights and normalize
                for s_id in range(int(self.M)):
                    target_pos = self.position(t, xyz[i,s_id,0], xyz[i,s_id,1])
                    new_alfa[i][s_id] = norm.pdf(true_obs, np.linalg.norm(self.sensor_pos-target_pos), 0.5)
                alfa[i] = new_alfa[i]/np.sum(new_alfa[i])
            
            for i in range(self.n):
                neighbor = np.where(At[i,:]>0)[0]
                # Create mixed alpha
                for it_idx, jn in enumerate(neighbor):
                    if it_idx == 0:
                        weights = [At[i,jn]*elem for elem in alfa[jn]]
                        xyz_appended = xyz[jn, :,:]
                    else:
                        weights = weights + [At[i,jn]*elem for elem in alfa[jn]]
                        xyz_appended = np.vstack((xyz_appended, xyz[jn, :,:]))
                
                # Stratified mixed resampling
                j = 0
                c = weights[0]
                for k in range(int(self.M)):
                    u = 1./len(weights)*np.random.rand(1)
                    beta = u + np.float(k)/(len(weights))
                    while beta > c:
                        j = j+1
                        c = c+weights[j]
                    new_alfa[i][k] = 1./(self.M)
                    xyz_0[i,k,:] = xyz_appended[j,:] + 0.01*np.random.normal(0,1,2)
                estimated_mean[i,t,:] = np.mean(xyz_0[i,:,:], axis=0)
                estimated_cov[i,t,:,:] = np.cov(xyz_0[i,:,:].T, bias=True)
            alfa = new_alfa
            xyz = xyz_0
            
            # Plot particle distributions after certain iterations 
            if (t%100 == 0):
                print (np.max(alfa)/np.min(alfa))
                circle = plt.Circle((self.x_true, self.y_true), self.r_true, color='b', fill=False)
                print ('Iteration:', t)
                # Add plot
                fig, ax1 = plt.subplots()
                ax1.scatter(xyz[:,:,0].flatten(), xyz[:,:,1].flatten())
                ax1.scatter(self.x_true, self.y_true, marker='^', label='Center')
                ax1.scatter(self.true_pos[t,0], self.true_pos[t,1], marker='x', label='Target location')
                ax1.scatter(self.sensor_pos[:,0], self.sensor_pos[:,1], marker='s', label='Sensor location')
                for i in range(self.n):
                    for j in range(self.n):
                        if At[i,j] > 0:
                            ax1.arrow(sensor_pos[i,0], sensor_pos[i,1], 
                                    sensor_pos[j,0]-sensor_pos[i,0], sensor_pos[j,1]-sensor_pos[i,1], color='red',
                                    ls='-.', alpha=0.5, head_width=0.2, head_length=0.5, length_includes_head=True)
                ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax1.add_patch(circle)
                ax1.set_title('Estimate spread at time = %i'% int(t))
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.set_xlim([-2, 6])
                ax1.set_ylim([-2, 6])
                ax1.legend()
                ax1.legend([circle], ['Trajectory'])
                
        mpfilter_ = { "xy": xyz, "emean": estimated_mean, "ecov":estimated_cov }
        pickle.dump( mpfilter_, open( "mpfilter.p", "wb" ) )
        return xyz, estimated_mean, estimated_cov
        
if __name__ == '__main__':
    d = 2
    M = 4e2
    T = 401
    m = 4
    sensor_pos = np.array([[1, 1], [1, 2], [2, 4], [2, 3]])
    
    logmaxeigcov = lambda cov: np.log(np.abs(np.max(np.linalg.eig(cov)[0])))
    
    for j in range(0):
        spfilter = SingleSenseParticle(d, T, M, sensor_pos[j, :])
        xyz, weight_ratio, estimated_mean, estimated_cov = spfilter.single_sense()
        fig, ax = plt.subplots(1,2)
        ax[0].plot(estimated_mean[:,0], color='blue', label='x')
        ax[0].hlines(spfilter.x_true,0, T, color='blue')
        ax[0].hlines(spfilter.y_true,0, T, color='orange')
        ax[0].plot(estimated_mean[:,1], color='orange', label='y')
        ax[0].set_title('Mean particle estimates')
        ax[0].grid()
        ax[0].legend()
        ax[1].plot([logmaxeigcov(estimated_cov[t,:,:]) for t in range(T)])
        ax[1].set_title('Log-max eigenvalue of particle covariance matrix')
        ax[1].grid()
        sys.stdout.flush()
        fig, ax = plt.subplots()
        ax.hlines(spfilter.x_true,0, T, color='blue')
        ax.hlines(spfilter.y_true,0, T, color='orange')
        ax.plot(estimated_mean[:,0], color='blue', label='x')
        ax.fill_between(np.linspace(0,T,T), estimated_mean[:,0]+estimated_cov[:,0,0], 
                        estimated_mean[:,0]-estimated_cov[:,0,0], 
                        color='blue', alpha=0.2)
        ax.plot(estimated_mean[:,1], color='orange', label='y')
        ax.fill_between(np.linspace(0,T,T), estimated_mean[:,1]+estimated_cov[:,1,1], 
                        estimated_mean[:,1]-estimated_cov[:,1,1], 
                        color='orange', alpha=0.4)
        ax.grid()
        ax.legend()
    
    mpfilter = MultiSenseParticle(d, T, M, sensor_pos)
    xyz, estimated_mean, estimated_cov = mpfilter.multi_sense()
    fig, ax = plt.subplots()
    ax.hlines(mpfilter.x_true,0, T, color='blue', label='x')
    ax.hlines(mpfilter.y_true,0, T, color='orange', label='y')
    for j in range(m):
        ax.plot(estimated_mean[j,:,0], color='blue')
        ax.fill_between(np.linspace(0,T,T), estimated_mean[j,:,0]+estimated_cov[j,:,0,0],\
                        estimated_mean[j,:,0]-estimated_cov[j,:,0,0],\
                        color='blue', alpha=0.2)
        ax.plot(estimated_mean[j,:,1], color='orange')
        ax.fill_between(np.linspace(0,T,T), estimated_mean[j,:,1]+estimated_cov[j,:,1,1], 
                        estimated_mean[j,:,1]-estimated_cov[j,:,1,1], 
                        color='orange', alpha=0.2)
    ax.grid()
    ax.legend()
    fig, ax = plt.subplots(1,2)
    ax[0].hlines(mpfilter.x_true,0, T, color='blue', label='x')
    ax[0].hlines(mpfilter.y_true,0, T, color='orange', label='y')
    ax[0].grid()
    ax[1].grid()
    ax[0].set_title('Mean particle estimates')
    ax[1].set_title('Log-max eigenvalue of particle covariance matrix')
    for j in range(m):
        ax[0].plot(estimated_mean[j,:,0], color='blue')
        ax[0].plot(estimated_mean[j,:,1], color='orange')
        ax[0].legend()
        ax[1].plot([logmaxeigcov(estimated_cov[j,t,:,:]) for t in range(T)])
        sys.stdout.flush()
    plt.show()