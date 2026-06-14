# Digital-twin-spring-mass
Virtual spring-mass system modelled in python with movements predicted by digital twin
Kalman filter used for state estimation and residual monitor for fault detection. Build as a short passion project to explore curiosity.

##What it does
This simulation creates a digital twin of a ball-spring oscillation to estimate movements by the system

## How to run
```bash
pip install -r requirements.txt
python physical_system.py
python digital_twin.py
```

## Key design descisions
- Linear Kalman filter used due to spring-mass system being linear
- Fault injection is software controlled allowing for precise detection.
