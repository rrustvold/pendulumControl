import numpy as np
import robot_calc_functions as tf

def bracket_3vec(v):
    bracketV = np.array( [ [0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0] ] )
    return bracketV

def bracket_6vec(vec):
    # takes a 6x1 vector
    # returns the bracket of the vectos [V]
    
    omega = np.zeros(3)
    omega[0] = vec[0]
    omega[1] = vec[1]
    omega[2] = vec[2]
    bracketOmega = bracket_3vec(omega)

    v = np.zeros((3,1))
    v[0,0] = vec[3]
    v[1,0] = vec[4]
    v[2,0] = vec[5]

    bracket = np.append(bracketOmega, v, 1)
    bracket = np.append(bracket, np.zeros((1,4)), 0)

    return bracket

def bracket_adjoint(vec):
    # Takes a 6x1 vector
    # returns the bracket of the adjoint [Ad_V]
    omega = np.zeros(3)
    omega[0] = vec[0]
    omega[1] = vec[1]
    omega[2] = vec[2]

    v = np.zeros(3)
    v[0] = vec[3]
    v[1] = vec[4]
    v[2] = vec[5]

    bracketOmega = bracket_3vec(omega)
    
    bracketV = bracket_3vec(v)
    
    top = np.append(bracketOmega, np.zeros((3,3)),1)
    
    bottom = np.append(bracketV, bracketOmega, 1)

    bracket_adjoint = np.matrix(np.append(top,bottom,0))

    return bracket_adjoint

def trans_adjoint(T):
    # takes a 4x4 transformation matrix
    # returns a 6x6 adjoint
    
    R = np.matrix( T[0:3, 0:3] ) # The rotation part

    p = T[0:3, 3]  # The displacement part

    top = np.append(R, np.zeros((3,3)), 1)
    bottom = np.matrix(bracket_3vec(p)) * R
    bottom = np.append(bottom,R,1)

    adjoint = np.matrix(np.append(top,bottom,0))

    return adjoint

def lie_bracket(V1, V2):
    BV1 = np.matrix(bracket_6vec(V1))
    BV2 = np.matrix(bracket_6vec(V2))
    lie = bracket_adjoint(V1) * V2
    return lie


def inverse_dynamics(theta, dtheta, ddtheta, Mlist, Slist, Glist, g, F_tip):
    n = len(Glist) ## number of links
    N = n+1
##    g = np.array([0,-9.8,0])  ## Gravity

    M_0 = [None]*(N+1)  ## list of configurations from 0 to i

    M_0[0] = np.matrix(np.eye(4))

    theta = np.append(theta, 0)
    dtheta = np.append(dtheta, 0)
    ddtheta = np.append(ddtheta, 0)


    M_im1 = [None]*(N+1)
    T_im1 = [None]*(N+1)
    S = [None]*(N+1)
    A = [None]*(N+1)
    V = [None]*N
    dV = [None]*N
    G = np.multiply(-1.,g)

    V[0] = np.transpose(np.matrix(np.array([0.]*6)))
    dV[0] = np.transpose(np.matrix(np.hstack(([0.,0.,0.], G))))

    ## Forward
    for i in range(1,N+1):

        ## eqn (8.49)
        M_im1[i] = Mlist[i-1]  #M_0,1 = Mlist[0], M_1,2 = Mlist[1]

        M_0[i] = M_0[i-1] * M_im1[i]  #M_0,1, M_0,1

        S[i] = np.transpose(np.matrix(Slist[i-1]))  ## column mector, S_1

        A[i] = trans_adjoint(np.linalg.inv(M_0[i])) * S[i]

        exparg = np.squeeze(np.array(A[i]*theta[i-1]))
        
        T_im1[i] = M_im1[i] * np.matrix(tf.MatrixExp6(exparg))

        ## eqn (8.50)
        if i < N:
            V[i] = trans_adjoint(np.linalg.inv(T_im1[i])) * V[i-1] + A[i]*dtheta[i-1]
            dV[i] = trans_adjoint(np.linalg.inv(T_im1[i])) * np.matrix(dV[i-1]) + lie_bracket(V[i], A[i]) * dtheta[i-1] + A[i]*ddtheta[i-1]

    ## Backward

    G = [None]*N
    T_ip1 = [None]*N
    F = [None]*(N+1)
##    F_tip = np.array([0]*6)
    F[N] = np.transpose(np.matrix(F_tip))
    tau = [None]*N

    for i in range(N-1,0,-1):
        G[i] = np.matrix(Glist[i-1])
        T_ip1[i] = np.linalg.inv(T_im1[i+1])

        F[i] = np.transpose(trans_adjoint(T_ip1[i])) * F[i+1] + G[i]*dV[i] - np.transpose(bracket_adjoint(V[i])) * G[i] * V[i]

        tau[i] = np.transpose(F[i]) * A[i]

    taulist = [None]*n
    
    for i in range(n):
        taulist[i] = tau[i+1][0,0] 

    #taulist[1]=-taulist[1]
    return taulist

def mass_matrix(theta, Mlist, Glist, Slist):
    n = len(Glist)
    g = [0.,0.,0.]
    F_tip = [0.]*6
    dtheta = [0.]*n
    ddtheta = [0.]*n
    M = np.zeros((n,n))
    
    for i in range(n):
        ddtheta[i] = 1
        M[:,i] = inverse_dynamics(theta, dtheta, ddtheta, Mlist, Slist, Glist, g, F_tip)
        ddtheta[i] = 0

    return M

def quadratic_terms(theta, dtheta, Mlist, Glist, Slist):
    n = len(Glist)
    g = [0.,0.,0.]
    F_tip = [0.]*6
    ddtheta = [0.]*n
    c = inverse_dynamics(theta, dtheta, ddtheta, Mlist, Slist, Glist, g, F_tip)

    return c

def end_effector_forces(theta, F_tip, Mlist, Glist, Slist):
    n = len(Glist)
    g = [0.,0.,0.]
    dtheta = [0.]*n
    ddtheta = [0.]*n
    
    JTFtip = inverse_dynamics(theta, dtheta, ddtheta, Mlist, Slist, Glist, g, F_tip)

    return JTFtip

def gravity_forces(theta, Mlist, Glist, Slist, g):
    n=len(Glist)
    F_tip = [0.]*6
    dtheta = [0.]*n
    ddtheta = [0.]*n
    gravity = inverse_dynamics(theta, dtheta, ddtheta, Mlist, Slist, Glist, g, F_tip)

    return gravity


def forward_dynamics(theta, dtheta, tau, F_tip, g, Mlist, Glist, Slist):

    n = len(Glist)
    #print tau
    Qterms = quadratic_terms(theta, dtheta, Mlist, Glist, Slist)
    ##print Qterms
   # print Qterms
    Gterms = gravity_forces(theta, Mlist, Glist, Slist, g)
   ## print Gterms
    #print Gterms
    RHS = np.transpose(np.matrix(np.subtract(np.subtract(tau, Qterms), Gterms))) #- end_effector_forces(theta, F_tip, Mlist, Glist, Slist)

    ##print RHS
    M = np.matrix(mass_matrix(theta, Mlist, Glist, Slist))
    ##print M
    ddtheta = np.linalg.inv(M)*RHS

    return ddtheta












    

    
    
