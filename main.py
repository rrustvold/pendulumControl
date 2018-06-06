import numpy as np
import myFunctions as mf
import robot_calc_functions as tf
from math import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy import sin, cos, exp

##### Link 1 #####
m1 = 10
L1 = .01
Izz1 = 0.
Ixx1 = m1*L1**2 /12
Iyy1 = m1*L1**2 / 12

I1 = np.array([ [Ixx1, 0., 0.],
                [0., Iyy1, 0.],
                [0., 0., Izz1] ])

M01 = np.array([ [1., 0., 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., L1/2],
                 [0., 0., 0., 1.] ])

##### Link 2 #####
m2 = 2.
L2 = 0.618
Izz2 = m2*L2**2 /12
Ixx2 = 0.
Iyy2 = m2*L2**2 / 12
I2 = np.array([ [Ixx2, 0., 0.],
                [0., Iyy2, 0.],
                [0., 0., Izz2] ])

M12 = np.array([ [1., 0., 0., L2/2],
                 [0., 1., 0., 0.],
                 [0., 0., 1., L1/2],
                 [0., 0., 0., 1.] ])


##### Link 3 #####
m3 = 4.
L3 = 1.
Izz3 = m3*L3**2 /12
Ixx3 = 0.
Iyy3 = m3*L3**2 / 12
I3 = np.array([ [Ixx3, 0., 0.],
                [0., Iyy3, 0.],
                [0., 0., Izz3] ])

M23 = np.array([ [1., 0., 0., L2/2+L3/2],
                 [0., 1., 0., 0.],
                 [0., 0., 1., 0.],
                 [0., 0., 0., 1.] ])

##### End Effector Link #####
M34 = np.array([ [1., 0., 0., L3/2],
                 [0., 1., 0., 0.],
                 [0., 0., 1., 0.],
                 [0., 0., 0., 1.] ])

Mlist = np.array((M01, M12, M23, M34))

G1 = np.append(I1, np.zeros((3,3)), 1)
bottom1 = np.append( np.zeros((3,3)), m1*np.eye(3), 1 )
G1 = np.append(G1, bottom1, 0)

G2 = np.append(I2, np.zeros((3,3)), 1)
bottom2 = np.append( np.zeros((3,3)), m2*np.eye(3), 1 )
G2 = np.append(G2, bottom2, 0)

G3 = np.append(I3, np.zeros((3,3)), 1)
bottom3 = np.append( np.zeros((3,3)), m3*np.eye(3), 1 )
G3 = np.append(G3, bottom3, 0)

Glist = np.array((G1, G2, G3))

Slist = np.array([ [0., 0., 0., 1., 0., 0.],
                   [0., 0., 1., 0., 0., 0.],
                   [0., 0., 1., 0., -L2, 0.],
                   [0., 0., 1., 0., -(L2+L3), 0.],
                   [0, 0, 0., 0., 0., 0] ])


def control_force(position, velocity):
    ## Uses Octave to determine LQR gains
    
    p = position[0]
    t1 = pi/2-position[1]
    t2 = pi/2-(position[2]+position[1])
    dp = velocity[0]
    dt1 = -velocity[1]
    dt2 = -(velocity[1]+velocity[2])

    kp = -31.6
    kdp = -60.7
    kt1 = 2038.
    kdt1 = 122.
    kt2 = -2452.
    kdt2 = -451.
    
    F = kp*p + kdp*dp + kt1*t1 + kdt1*dt1 + kt2*t2 + kdt2*dt2

    #print F
    return F

g = [0., -9.8, 0.]
theta = [-1., pi/2+.1, 0.]  ## initial condition
#theta = [.5, pi/2+.2, -.1]
#dtheta = [1.5,0.5,-.05]          ## initial condition

#theta = [0, pi/2-.1, 0]
#theta = [0., -pi/2, 0.]
dtheta = [0,0,0]

dt = .0025
N = 2500
numLinks = 3

thetaRecord = np.zeros((numLinks,N))
dthetaRecord = np.zeros((numLinks,N))
tau = np.zeros((numLinks,N))
F_tip = np.zeros((6,N))

thetaRecord[:,0] = theta
dthetaRecord[:,0] = dtheta

qhat = np.zeros((numLinks*2,N))  ## state estimate

qhat[0:3,0] = theta# + np.random.normal(0,.1)  ## initial state estiamte is initial condition plus noise
qhat[3:6,0] = dtheta

## linearized point mass model (from octave)
A = np.matrix(np.array([[ 0.00000,    1.00000,    0.00000,    0.00000,    0.00000,    0.00000],
   [-0.00000,   -0.00000,   -5.88000,   -0.00000,   -0.00000,   -0.00000],
    [0.00000,    0.00000,    0.00000,    1.00000,    0.00000,    0.00000],
   [-0.00000,   -0.00000,   57.08738,   -0.00000,  -31.71521,   -0.00000],
    [0.00000,    0.00000,    0.00000,    0.00000,    0.00000,    1.00000],
   [-0.00000,   -0.00000,  -29.40000,   -0.00000,   29.40000,   -0.00000] ]))

B = np.matrix(np.array([ [0.00000],
   [0.10000],
   [0.00000],
  [-0.16181],
   [0.00000],
  [-0.00000] ]))

u = np.zeros((numLinks*2,1))

## Covariance matrices
P = np.matrix(np.eye(numLinks*2))  ## among states
H = np.matrix(np.eye(numLinks*2))  ## among sensors
Q = np.matrix(np.eye(numLinks*2))  ## among states noise
R = np.matrix(np.eye(numLinks*2))  ## among sensors noise

P = A*P*A.T + Q

K_p = P*H.T*np.linalg.inv(H*P*H.T + R)  # Kalman Gain

normRecord = np.zeros(N)
def norm(state):
    tstate = [None]*6
    N = 0
    tstate[1] = abs((pi/2 - state[1])%(2*pi))    
    tstate[2] = abs((pi/2 - (state[1]+state[2]))%(2*pi))
    for j in [1,2]:
        if abs(tstate[j]) < pi:
            tstate[j] = abs(tstate[j])
        else:
            tstate[j] = pi - (abs(tstate[j])%pi)

    tstate[3] = state[3]
    tstate[4] = -state[4]
    tstate[5] = -(state[4]+state[5])

    #N = np.sqrt(3000*tstate[1]**2 + 3000*tstate[2]**2 + 100*tstate[4]**2 + 100*tstate[5]**2)
    N = 2*abs(tstate[1]) + 3*abs(tstate[2]) + .2*abs(tstate[3]) + abs(tstate[4]) + .666*abs(tstate[5])
    
    return N


for i in range(1,N):
    tau[:,i] = -.375* dthetaRecord[:,i-1] #+ np.random.normal(0,.5) ## Drag on joints + noise
    
    qhat[0:3,i-1] = thetaRecord[0:3,i-1]
    qhat[3:6,i-1] = dthetaRecord[0:3,i-1]
    normRecord[i] = norm(qhat[:,i-1])

    tau[0,i] = control_force(qhat[0:3,i-1], qhat[3:6,i-1]) ## apply control input based on state estimate
    
    # Run the non-linear forward dynamics of the system
    ddtheta = mf.forward_dynamics(thetaRecord[:,i-1], dthetaRecord[:,i-1], tau[:,i], F_tip[:,i-1], g, Mlist, Glist, Slist)  
    thetaRecord[:,i] = thetaRecord[:,i-1] + dt*dthetaRecord[:,i-1]
    
    for j in range(numLinks):
        dthetaRecord[j,i] = dthetaRecord[j,i-1] + dt*ddtheta[j,0]  

    # Predict the next state using linear dynamics
    u[0] = tau[0,i]
    qhat[:,i] = np.transpose(A*np.matrix(qhat[:,i-1]).T + B*u[0])

    # Take sensor measurements
    z = np.matrix(np.append(thetaRecord[:,i], dthetaRecord[:,i]) + np.random.normal(0,.5)).T

    #update state estimate
    qhat[:,i] = np.transpose(np.matrix(qhat[:,i]).T + K_p*(z - H*np.matrix(qhat[:,i]).T))
 


theta1 = thetaRecord[0,:]
theta2 = thetaRecord[1,:]
theta3 = thetaRecord[2,:]
u = tau[0,:]

##plt.plot(theta1)
##plt.plot(theta2)
##plt.plot(theta3)
##plt.plot(u)
##plt.plot(normRecord)
##plt.show()

#create animation
    
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([],[])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [theta1[i], theta1[i]+L2*cos(theta2[i]),
             theta1[i]+L2*cos(theta2[i]), theta1[i]+L2*cos(theta2[i]) + L3*cos(theta2[i]+theta3[i]) ] #,
            ## L1*cos(theta1[i]) + L2*cos(theta1[i]+theta2[i]), L1*cos(theta1[i]) + L2*cos(theta1[i]+theta2[i]) + L3*cos(theta1[i]+theta2[i]+theta3[i])]
    thisy = [0, L2*sin(theta2[i]),
             L2*sin(theta2[i]), L2*sin(theta2[i]) + L3*sin(theta2[i]+theta3[i]) ]#,
           ##  L1*sin(theta1[i]) + L2*sin(theta1[i]+theta2[i]), L1*sin(theta1[i]) + L2*sin(theta1[i]+theta2[i]) + L3*sin(theta1[i]+theta2[i]+theta3[i])]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

##Writer = animation.writers['ffmpeg']
##writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani = animation.FuncAnimation(fig, animate, np.arange(1, N), interval=dt/2, \
                              blit=True, init_func=init)
##ani.save('pen2.mp4', writer='ffmpeg', fps=400, bitrate=900, dpi=120)
plt.show()



















