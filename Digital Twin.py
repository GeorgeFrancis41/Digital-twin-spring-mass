import matplotlib.pyplot as plt
import numpy as np

# Load physical system output
sensor_reading = np.load("sensor_data.npy") #Noisy measurments from sensor
true_position = np.load("true_position.npy") #Actual position of ball
t = np.load("time.npy") #Time array
dt = t[1] - t[0] #Time step

m, c, k = 1.0, 0.3, 5.0

# Using a Kalaman filter to estimate unknowns, begin by setting paramaters
Q = np.diag([0.0001, 0.0001]) #Process Noise
R = np.array([[0.05**2]]) #Measurment Nouse (sigma squared)
H = np.array([[1,0]]) #Observation - tells sensor to only measure position

#Initial estimates
x_est = np.array([1,0]) #Inital state estimate
P = np.eye(2) * 0.1 #Initial uncertainty

#Transition Matrix
def get_F(m, c, k, dt):
    return np.array([[1,dt],
                     [(-k/m)*dt, 1-(c/m)*dt]
                     ]) #Comes from discretizign the spring equation

#Then predict the values
F = get_F(m, c, k, dt)
x_pred = F @ x_est #Project state forward one timestep
P_pred = F @ P @ F.T +Q #Project uncertainty forward, adding noise

def Kalman_update(x_pred, P_pred, z, H, R):
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred
    return x_est, P

# Initialise
x_est = np.array([1.0, 0.0])
P = np.eye(2) * 0.1
F = get_F(m, c, k, dt)

estimated_states = np.zeros((len(t), 2))
residuals = np.zeros(len(t))

for i in range(len(t)):
    # Predict
    x_pred = F @ x_est
    P_pred = F @ P @ F.T + Q

    # Update
    z = np.array([sensor_reading[i]])
    x_est, P = Kalman_update(x_pred, P_pred, z, H, R)

    # Store
    estimated_states[i] = x_est
    residuals[i] = abs(sensor_reading[i] - x_est[0])
    
x_kalman = estimated_states[:, 0]

plt.plot(t, true_position, label="True position", linewidth=2, color="black")
plt.plot(t, sensor_reading, label="Sensor reading", alpha=0.4, linewidth=0.8, color="blue")
plt.plot(t, x_kalman, label="Kalman estimate", linewidth=2, color="red", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.legend()
plt.grid(True)
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Top plot - tracking
ax1.plot(t, true_position, label="True position", linewidth=2, color="black")
ax1.plot(t, sensor_reading, label="Sensor reading", alpha=0.4, linewidth=0.8, color="blue")
ax1.plot(t, x_kalman, label="Kalman estimate", linewidth=2, color="red", linestyle="--")
ax1.axvline(x=5, color="orange", linestyle="--", label="Fault injected")
ax1.set_ylabel("Displacement (m)")
ax1.legend()
ax1.grid(True)

# Bottom plot - residual monitor
window = 200
residuals_smooth = np.convolve(residuals, np.ones(window)/window, mode='same')
threshold = 0.018

ax2.plot(t, residuals_smooth, label="Residual", color="purple", linewidth=0.8)
ax2.axhline(y=threshold, color="red", linestyle="--", label="Alert threshold")
ax2.axvline(x=5, color="orange", linestyle="--", label="Fault injected")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Residual (m)")
ax2.set_title("Residual Monitor")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()