import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
import networkx as nx 
import os
from pyDOE import lhs
import pickle 

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rc('font', size=10) #controls default text size
plt.rc('axes', titlesize=16) #fontsize of the title
plt.rc('axes', labelsize=14) #fontsize of the x and y labels
# plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
# plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
plt.rc('legend', fontsize=12) #fontsize of the legend

def graph_diameter(A):
    """
    Input: Adjacency matrix
    """
    G = nx.from_numpy_matrix(A)
    return nx.diameter(G)

def convert_to_ds(A):
    """
    Convert adjacency matrix into a doubly stochastic form using repeated normalization 
    along rows and columns (Sinkhorn algorithm)
    """
    err = 0.1
    while err > 1e-3:
        A_prev = A
        # Make row stochastic
        A = A/np.sum(A, axis=1).reshape(-1,1)
        # Make column stochastic
        A = np.transpose(A.T/np.sum(A, axis=0).reshape(-1,1))
        err = np.linalg.norm(A-A_prev)
    return A

def generate_connected_graph_edges(n, n_edges):
    """
    Generate a random connected graph with specified number of edges
    """
    assert n >= 2
    E_MIN = n-1
    E_MAX = 0.5*(n**2 - n)
    assert n_edges >= E_MIN
    assert n_edges <= E_MAX
    A = np.zeros((n,n))
    in_line = [0, 1]
    A[0, 1] = 1
    out_of_line = [i for i in range(2, n)]
    # Connect isolated nodes to the graph
    for i in range(2, n):
        node_in_line_idx = np.random.randint(0, len(in_line), 1)
        node_in_line = in_line[node_in_line_idx[0]]
        A[i, node_in_line] = 1
        in_line.append(i)
        out_of_line.remove(i)
    
    A = A + np.transpose(A)
    A = A + np.eye(n)
    
    # Add remaining number of edges to the connected graph
    for i in range(n_edges - E_MIN):
        can_add = 0
        while can_add == 0:
            random_agent = np.random.randint(0, n, 1)[0]
            can_add = n - np.sum(A[:, random_agent])
        
        n_indices = np.where(A[:,random_agent]==0)[0].tolist()
        random_connection = np.random.randint(0, n, 1)[0]
        A[random_agent, random_connection] = 1
    A = A + np.transpose(A)
    A = np.array(A>0)*1
    A = A.astype(np.float)
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigs = list(np.linalg.eig(L)[0])
    # Remove one of the zeros
    eigs.remove(np.min(eigs))
    # Sparsest cut: minimize the ratio of the number of edges across the cut 
    # divided by the number of vertices in the smaller half of the partition
    fiedler_eig = np.min(eigs)
    A_base = A
    A = convert_to_ds(A)
    return A, D, A_base, fiedler_eig, graph_diameter(A)

def rand_network(n):
    """
    Generate n-node network with arbitrary number of edges
    Represent the network as a row stochastic matrix 
    """
    A = np.eye(n)
    n_edges = np.random.randint(1,2*n)
    for i in range(n_edges):
        i1 = np.random.randint(1,n+1)-1
        i2 = np.random.randint(1,n+1)-1
        while (i2 == i1):
            i2 = np.random.randint(1,n)
        A[i1, i2] = 1
    A = A/np.sum(A, axis=1).reshape(-1,1)
    return A

def broken_link_network(A):
    """
    Remove arbitrary number of edges from given adjacency matrix 
    Return new row stochastic version of the adjacency matrix
    """
    A = A - np.diag(A)
    # Number of edges without self-loops
    edge_indices = np.where(A>0) # edge_indices[0] = array([0, 1, 1, 1, 2, 3, 4], dtype=int64)
    n_edges = np.sum(A>0)
    nedges_remove = np.random.choice(n_edges)
    redge_idx = np.random.choice(n_edges, size=nedges_remove, replace=False)
    for e_idx in redge_idx:
        A[edge_indices[0][e_idx], edge_indices[0][e_idx]] = 0
    A = A + np.eye(np.shape(A)[0])
    At = A/np.sum(A, axis=1).reshape(-1,1)
    return At
    
def generate_networks(T, n):
    """
    Generate time varying 5-connected communication networks, and a connected observation network
    """
    At = np.zeros((T, n, n))

    A = generate_connected_graph_edges(n, n+1)[0]
    for t in range(T):
        # Generate row stochastic matrix
        if (t%5 == 0):
            At[t,:,:] = generate_connected_graph_edges(n, n+1)[0] #
        else:
            At[t,:,:] = rand_network(n) #generate_connected_graph_edges(n, n+1)[0] #
    
    return A, At

def generate_broken_link_networks(T, n):
    """
    Generate time varying communication networks simulating link failures, and a connected observation network
    """
    At = np.zeros((T, n, n))

    A = generate_connected_graph_edges(n, n+1)[0]
    for t in range(T):
        At[t,:,:] = broken_link_network(A)
    
    return A, At

def generate_observations(T, n, d, A, x_true):
    """
    Generate noisy relative displacement observations for each sensor node in the network 
    :T: Number of iterations 
    :n: Number of sensors
    :d: State dimensions
    :A: Adjacency matrix 
    :x_true: True sensor positions 
    """
    z = []
    for agent_i in range(n):
        neighbors = np.where(A[agent_i,:]>0)[0]
        n_i = len(neighbors)
        H_it = np.zeros((n_i*d, d*n))
        Omega_zit = np.zeros((n_i*d, n_i*d))
        z_i = np.zeros((T, n_i*d))
        
        # Observation matrix H and Omega_z depend on network configuration
        for idx, agent in enumerate(neighbors):
            H_it[idx*d:idx*d+d, agent*d:agent*d+d] += np.eye(d)
            H_it[idx*d:idx*d+d, agent_i*d:agent_i*d+d] += -np.eye(d)
            if agent != agent_i:
                Omega_zit[idx*d:idx*d+d, idx*d:idx*d+d] = 0.5*np.eye(d)

        # Generate observations
        for t in range(T):
            z_i[t,:] = np.random.multivariate_normal(H_it.dot(x_true), Omega_zit)
            
        z.append(z_i)
    return z

def distributed_Gaussian_filter(T,n,d,A,At,z, x_true, alfa):
    """
    Implement distributed Gaussian localization for fixed observation and time-varying communication network.
    :T: Number of iterations 
    :n: Number of sensors
    :d: State dimensions
    :A: Adjacency matrix 
    :At: Time-varying communication matrix 
    :z: Observations at each agent 
    :x_true: True sensor positions 
    :alfa: Positive weight term on agents' noisy likelihoods
    """
    mu = np.zeros((T+1, n, n*d))
    Omega = np.zeros((T+1, n, n*d, n*d))
    
    # Initialize Information matrix for each agent
    for i in range(n):
        Omega[0,i,:,:] = np.eye(n*d)
    
    for t in range(T):
        for agent_i in range(n):
            neighbors = np.where(A[agent_i,:]>0)[0]
            n_i = len(neighbors)
            H_it = np.zeros((n_i*d, d*n))
            Omega_zit = np.zeros((n_i*d, n_i*d))
            
            # Observation matrix H and Omega_z depend on network configuration
            for idx, agent in enumerate(neighbors):
                H_it[idx*d:idx*d+d, agent*d:agent*d+d] += np.eye(d)
                H_it[idx*d:idx*d+d, agent_i*d:agent_i*d+d] += -np.eye(d)
                if agent != agent_i:
                    Omega_zit[idx*d:idx*d+d, idx*d:idx*d+d] = 0.5*np.eye(d)

            # Generate observations
            z_it = z[agent_i][t,:]
            # print ('Observation for agent ', agent_i, 'is ', z_it, Omega_zit)
            
            Omega[t+1, agent_i, :, :] = alfa*np.dot(np.dot(H_it.T, Omega_zit), H_it)
            mu[t+1, agent_i, :] = alfa*np.dot(np.dot(H_it.T, Omega_zit), z_it.reshape(-1,1)).flatten()
            for j in range(n):
                Omega[t+1, agent_i, :, :] = Omega[t+1, agent_i, :, :] + At[t,agent_i,j]*Omega[t, j, :, :]
                mu[t+1, agent_i, :] = mu[t+1, agent_i, :] + At[t,agent_i,j]*(Omega[t, j, :, :].dot(mu[t, j, :].reshape(-1,1))).flatten()
            mu[t+1, agent_i, :] = np.dot(np.linalg.pinv(Omega[t+1, agent_i, :, :]), mu[t+1, agent_i, :])
            
    return mu, Omega, A, At


if __name__ == '__main__':
    n = 10
    T = 1200
    d = 2 # Dimensionality of state x, cooperative localization has same observation dimensionality
    alfa = 1.
    x_true = lhs(d, n, criterion='maximin')#np.random.rand(n,d)
    x_true = x_true - np.tile(x_true[0,:],(n,1))
    val1 = np.zeros((T,3))
    print ('True positions are', x_true)

    A, At = generate_networks(T, n)
    z = generate_observations(T, n, d, A, x_true.flatten())
    
    viz_agent_idx = 1
    # import pickle
    # out_data = {'who':'Distributed Gaussian for self localization', 'mean':mu, 'infoMat':Omega, 'truePos':x_true, 'commMat':Aobs}
    # pickling_on = open("coop_loc.pickle","wb")
    # pickle.dump(out_data, pickling_on)
    # pickling_on.close()
    
    """
    Plot the estimated position of agent 2 by all agents in the network for different likelihood weights
    """
    for alfa_iter, alfa in enumerate([0.5, 1, 1.5]):
        mu, Omega, Aobs, At = distributed_Gaussian_filter(T,n,d, A, At,z, x_true.flatten(), alfa)# Format [x1,x2,x3,...]^T

        fig, ax = plt.subplots()
        ax.hlines(x_true[1,0]-x_true[0,0], xmin = 0, xmax = T, color='red', linestyles='--', label='$x_2^{\star}$')
        ax.hlines(x_true[1,1]-x_true[0,1], xmin = 0, xmax = T, color='red' , label='$y_2^{\star}$')
        for i in range(n):
            x_est = mu[:, i, 2]-mu[:, i, 0]
            # 3-sigma error
            x_err = 3*np.reciprocal(Omega[:,i,2,2])
            ax.plot(x_est, '--', linewidth=1)
            ax.fill_between(np.arange(0,T+1), x_est - x_err, x_est + x_err, alpha=0.03)
            y_est = mu[:, i, 3]-mu[:, i, 1]
            y_err = 3*np.reciprocal(Omega[:,i,3,3])
            ax.plot(y_est, ':', linewidth=1)
            ax.fill_between(np.arange(0,T+1), y_est - y_err, y_est + y_err, alpha=0.03)
        ax.set_ylim([-1,1])
        ax.set_xlabel('Time steps')
        ax.set_ylabel('Estimated agent 2 coordinates')

    # Plot Observation network configuration
    fig, ax = plt.subplots()
    for i in range(n):
        ax.scatter(x_true[i,0]-x_true[0,0], x_true[i,1]-x_true[0,1],marker='^', s=50)
        A = Aobs
        for j in range(n):
            if A[i,j] > 0:
                ax.plot([x_true[i,0]-x_true[0,0], x_true[j,0]-x_true[0,0]]
                        , [x_true[i,1]-x_true[0,1], x_true[j,1]-x_true[0,1]], '--', color='red', linewidth=0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Observation network configuration')
    
    # Plot time varying communication network
    for t in range(5):
        fig, ax = plt.subplots()
        for i in range(n):
            ax.scatter(x_true[i,0]-x_true[0,0], x_true[i,1]-x_true[0,1],marker='^', s=50)
            A = At[t,:,:]
            for j in range(n):
                if Aobs[i,j] > 0:
                    ax.plot([x_true[i,0]-x_true[0,0], x_true[j,0]-x_true[0,0]]
                        , [x_true[i,1]-x_true[0,1], x_true[j,1]-x_true[0,1]], ':', color='red', linewidth=0.2)
                if A[i,j] > 0:
                    # ax.plot([x_true[i,0]-x_true[0,0], x_true[j,0]-x_true[0,0]]
                            # , [x_true[i,1]-x_true[0,1], x_true[j,1]-x_true[0,1]], '--', color='blue', linewidth=1.5)
                    ax.arrow(x_true[i,0]-x_true[0,0], x_true[i,1]-x_true[0,1], 
                            x_true[j,0]-x_true[i,0], x_true[j,1]-x_true[i,1], color='blue',
                            ls='-.', alpha=0.5, head_width=0.02, head_length=0.05, length_includes_head=True)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Network configuration at time = %d' %(t+1))
        filename = 'network'+str(t)+'.png'
        plt.savefig(filename,
                    format='png',
                    bbox_inches ="tight",
                    transparent = True,
                    orientation ='landscape')
       
    plt.show()
