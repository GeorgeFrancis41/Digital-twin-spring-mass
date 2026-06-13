import matplotlib.pyplot as plt
import numpy as np

#Intial conditions
m = 1#kg (Mass of the ball)
k = 5#N/m (Spring constant)
c = 0.3#Ns/m (Damping constant)
c_injection = 3 #Damping constant for fault injection
x0 = 1#m (Initial position displaced from equilibrium)
v0 = 0#m/s (Initial velocity)
state0 = [x0,v0] #State gives the position and velocity at the current moment [x,v]
dt = 0.01#s (Time step)
T0 = 0#s (Initial time)
T = 20#s (Total time)

#Define a function whcih computes the derivatives 
def derivative(state, t, m, c, k):
    x, v = state 
    dxdt = v
    dvdt = (-k * x - c * v)
    return [dxdt, dvdt]

#Create a range for all of time and spave set the intial time and space to be state 0 in the range
t = np.arange(T0, T, dt)
states = np.zeros((len(t),2))
states[0] = state0

#Evolve the simulation
for i in range(1,len(t)):
    
    #After 5 seconds we deliberatley corrupt the data by making the damping constant different
    #This lets us see how the digital twin adapts to this change
    if t[i] <= 5:
        derivs = derivative(states[i-1], t[i-1], m, c, k)
        states[i] = states[i-1] + dt * np.array(derivs) #Evolves the motion in time
    else:
        derivs = derivative(states[i-1], t[i-1], m, c_injection, k)
        states[i] = states[i-1] + dt * np.array(derivs) #Evolves the motion in time
    
x = states[:,0]

#Add noise to the system to make it more realistic
sigma = 0.02 #Sensor noise standard deviation
noise = np.random.normal(0, sigma, len(t))
sensor_reading = x + noise


#Plot displacement against time
plt.plot(t,x, c = 'k', label = "Displacement vs time")

#Compare with added noise plot
plt.plot(t, sensor_reading, alpha = 0.4, c = 'b', label = 'Noise')

plt.xlabel("Time (s)")
plt.ylabel("Displacement from equilibrum (m)")
plt.grid()
plt.legend()
plt.show()

np.save("sensor_data.npy", sensor_reading)
np.save("true_position.npy", x)
np.save("time.npy", t)