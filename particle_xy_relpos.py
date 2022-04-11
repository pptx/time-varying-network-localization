import numpy as np
import os
from pyDOE import lhs
from math import atan2
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
import sys


"""
Implement particle filter for estimating the target position.
Use observation model of full-dimensional Gaussian, so that each sensing agent 
is capable of observing and inferring the target.
"""
class SingleSenseParticle(object):

    def __init__(self, d, T, M, sense_pos):
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
        self.x_true = 13
        self.y_true = 10
        self.sensor_pos = sense_pos 
        
    def generate_particles(self):
        """
        Generate particles representing target position using Latin hypercube sampling
        """
        xyz_0 = 15*lhs(self.d, int(self.M))
        xyz_0[:,0] = xyz_0[:,0]-1 # Center x
        xyz_0[:,1] = xyz_0[:,1]-1 # Center y
        return xyz_0

    def generate_observations(self):
        """
        Generate noisy relative displacement observations between the sensor and the target
        """
        true_obs = np.zeros((self.T,self.d))

        for t in range(T):
            relative_pos = np.array([self.x_true, self.y_true]) - self.sensor_pos
            true_obs[t,:] = multivariate_normal.rvs(relative_pos, 0.3*np.eye(self.d))
        return true_obs

    def single_sense(self):
        # Generate particles representing target position
        xy = self.generate_particles()
        xy_0 = self.generate_particles()
        true_obs = self.generate_observations()
        
        estimated_mean = np.random.rand(self.T, self.d)
        estimated_cov = np.random.rand(self.T, self.d, self.d)
        
        # Define particle weights 
        alfa = [1./self.M for i in range(int(self.M))]
        new_alfa = [1./self.M for i in range(int(self.M))]
        weight_ratio = [0. for i in range(self.T)]
        cov = np.eye(self.d)
        
        for t in range(self.T):
            # Update particle weights and normalize
            for s_id in range(int(self.M)):
                new_alfa[s_id] = multivariate_normal.pdf(true_obs[t,:], 
                xy[s_id,:]-self.sensor_pos, cov)
            alfa = new_alfa/np.sum(new_alfa)
            # print(alfa, np.min(alfa), np.max(alfa), np.mean(alfa))
            weight_ratio[t] = np.max(alfa)/np.min(alfa)
            
            # Plotting particle distributions in target space
            if (t%200 == 0):
                print ('Iteration:', t, 'Weight ratio:', weight_ratio[t])
                fig, ax = plt.subplots()
                ax.scatter(xy[:,0], xy[:,1])
                ax.scatter(self.x_true, self.y_true, marker='^', label='Target location')
                ax.scatter(self.sensor_pos[0], self.sensor_pos[1], marker='s', label='Sensor location')
                ax.set_title('Estimate spread at time = %i'% int(t))
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.legend()
                            
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
                xy_0[k,:] = xy[j,:] + 0.01*np.random.normal(0,1,2)
            
            estimated_mean[t,:] = np.mean(xy_0, axis=0)
            estimated_cov[t,:,:] = np.cov(xy_0.T, bias=True)
            alfa = new_alfa
            xy = xy_0
            
        return xy, weight_ratio, estimated_mean, estimated_cov
        

        
class MultiSenseParticle(object):

    def __init__(self, d, T, M, sensor_pos):
        """
        :d: State dimensions
        :T: Number of algorithm iterations
        :M: Number of particles in the filter 
        :sense_pos: Position of multiple sensors
        :x_true: Define true x-coordinate of target
        :y_true: Define true y-coordinate of target
        """
        self.d = d
        self.T = T
        self.M = M
        self.dt = 0.5
        self.x_true = 13
        self.y_true = 10
        self.sensor_pos = sensor_pos 
        self.n = np.shape(self.sensor_pos)[0]
        
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
        
    def generate_particles(self):
        """
        Generate particles representing target position using Latin hypercube sampling
        """
        xyz_0 = 15*lhs(self.d, int(self.M))
        xyz_0[:,0] = xyz_0[:,0]-1 # Center x
        xyz_0[:,1] = xyz_0[:,1]-1 # Center y
        return xyz_0

    def generate_observations(self, sense_pos):
        """
        Generate noisy relative displacement observations between each sensor and the target
        """
        relative_pos = np.array([self.x_true, self.y_true]) - sense_pos
        true_obs = multivariate_normal.rvs(relative_pos, 0.3*np.eye(self.d))
        return true_obs

    def multi_sense(self):
        # Generate agent-specific particles representing target position
        n_xy = np.zeros((self.n, int(self.M), self.d))
        n_xy0 = np.zeros((self.n, int(self.M), self.d))
        for i in range(self.n):
            n_xy[i, :,:] = self.generate_particles()
            n_xy0[i, :,:] = self.generate_particles()
        
        estimated_mean = np.random.rand(self.n, self.T, self.d)
        estimated_cov = np.random.rand(self.n, self.T, self.d, self.d)
        
        # Define particle weights 
        alfa = [[1./self.M for i in range(int(self.M))] for i in range(self.n)]
        new_alfa = [[1./self.M for i in range(int(self.M))] for i in range(self.n)]
        
        cov = np.eye(self.d)
        
        for t in range(self.T):
            At = self.rand_network()
            for i in range(self.n):
                true_obs = self.generate_observations(self.sensor_pos[i,:])
                for s_id in range(int(self.M)):
                    new_alfa[i][s_id] = multivariate_normal.pdf(true_obs, n_xy[i,s_id,:]-self.sensor_pos[i,:], cov)
                alfa[i] = new_alfa[i]/np.sum(new_alfa[i])
                            
            
            for i in range(self.n):
                neighbor = np.where(At[i,:]>0)[0]
                # Create mixed alpha 
                for it_idx, jn in enumerate(neighbor):
                    if it_idx == 0:
                        weights = [At[i,jn]*elem for elem in alfa[jn]]
                        xyz_appended = n_xy[jn, :,:]
                    else:
                        weights = weights + [At[i,jn]*elem for elem in alfa[jn]]
                        xyz_appended = np.vstack((xyz_appended, n_xy[jn, :,:]))
                
                # Stratified mixed resampling with modified weights 
                j = 0
                c = weights[0]
                for k in range(int(self.M)):
                    u = 1./len(weights)*np.random.rand(1)
                    beta = u + np.float(k)/(len(weights))
                    while beta > c:
                        j = j+1
                        c = c+weights[j]
                    new_alfa[i][k] = 1./(self.M)
                    n_xy0[i,k,:] = xyz_appended[j,:] + 0.01*np.random.normal(0,1,2)
                estimated_mean[i,t,:] = np.mean(n_xy0[i,:,:], axis=0)
                estimated_cov[i,t,:,:] = np.cov(n_xy0[i,:,:].T, bias=True)
            alfa = new_alfa
            n_xy = n_xy0
            
        return n_xy, estimated_mean, estimated_cov

if __name__ == '__main__':
    d = 2
    M = 1e2
    T = 1001
    m = 4
    sensor_pos = np.array([[7, 5], [0, 0], [0, 5], [7, 0]])
    logmaxeigcov = lambda cov: np.log(np.abs(np.max(np.linalg.eig(cov)[0])))
    
    for j in range(1):
        spfilter = SingleSenseParticle(d, T, M, sensor_pos[j,:])
        xyz, weight_ratio, estimated_mean, estimated_cov = spfilter.single_sense()
        fig, ax = plt.subplots(1,2)
        ax[0].plot(estimated_mean[:,0], color='blue', label='x')
        ax[0].hlines(spfilter.x_true,0, T, color='blue')
        ax[0].hlines(spfilter.y_true,0, T, color='orange')
        ax[0].plot(estimated_mean[:,1], color='orange', label='y')
        ax[0].legend()
        ax[1].plot([logmaxeigcov(estimated_cov[t,:,:]) for t in range(T)])
        ax[1].set_title('Maximum eigenvalue of particle covariance matrix')
        sys.stdout.flush()
    
    mpfilter = MultiSenseParticle(d, T, M, sensor_pos)
    xyz, estimated_mean, estimated_cov = mpfilter.multi_sense()
    fig, ax = plt.subplots(1,2)
    for j in range(m):
        ax[0].plot(estimated_mean[j,:,0], color='blue', label='x')
        ax[0].hlines(mpfilter.x_true,0, T, color='blue')
        ax[0].hlines(mpfilter.y_true,0, T, color='orange')
        ax[0].plot(estimated_mean[j,:,1], color='orange', label='y')
        ax[0].legend()
        ax[1].plot([logmaxeigcov(estimated_cov[j,t,:,:]) for t in range(T)])
        ax[1].set_title('Maximum eigenvalue of particle covariance matrix')
    plt.show()