# pendulumControl
Simulating the dynamics and control of a double pendulum on a slider


This program simulates and controls the dynamics of a double pendulum on a cart. It demonstrates the following:

1. Accurately simulating the dynamics of the system. This is accomplished using the textbook "Modern Robotics -- Mechanics, Planning, and Control" by Kevin M Lynch and Frank C. Park as a guide. Essentially, the kinematics of the system are understood through Lie algebra. The vast majority of the code goes into this task.

2. Estimating the state of the system using a Kalman filter. To make the problem more realistic, I've introduced noise into the state estimation so the model is not operating with perfect information. However, it uses an optimal state estimator (the Kalman filter) to obtain good estimates.

3. Applying a control force to the cart to keep the pendulum in an upright position and to move the cart to the origin. The control force acts only on the cart. The joints of the pendulum are free to rotate without any control (though there is some friction in the joints). The controller is a linear quadratic regulator (optimal controller). The coefficients for the LQR were found using a seperate matlab script. The matlab script only needs to be run once to determine the LQR coefficients.

the main script carries out the simluation and creates an animation. Included in this repository are two animated gifs. One is for the system without any controller. It is let go from some initial condition and naturally it flops around chaotically. The second animation shows the successful estimator/controller keeping the pendulum upright and moving it carefully to the origin.
