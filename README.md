# Driving-games


This repository has been created for the project of the master course "Planning and Decision Making for Autonomous Robots" with <a href="https://idsc.ethz.ch/research-frazzoli/people/person-detail.MjI0MDM0.TGlzdC8yNjg5LDQ4ODg4MTE2Mw==.html">Prof. Emilio Frazzoli</a>. The project's goal is to guide each agent (vehicle) to a designated area by first generating the path, then controlling the agent, and finally avoiding both static obstacles and other agents.

<p align="center">
    <img src="https://github.com/LilianLaporte/Driving-games/assets/93781819/de0686da-915a-4439-85b8-732a4fd30570" alt="GIF" width="60%"/>
</p>

## Car model: Bicycle
<p align="center">
    <img src="https://github.com/LilianLaporte/Driving-games/assets/93781819/61933982-3ac8-4afd-b2ba-f1b8f9738eff" alt="Bicycle" width="30%"/>
</p>

- **State**: $`x=\begin{bmatrix}
p_{1} & p_{2} & \theta
\end{bmatrix}`$
- **Inputs**: $`u=\begin{bmatrix}
v_{r} & \omega
\end{bmatrix}`$
- **Dynamics**: 
$`\dot{p_{1}}=v_{r}(t)cos(\theta(t))`$  
&emsp; &emsp; &emsp; &emsp; $`\dot{p_{2}}=v_{r}(t)sin(\theta(t))`$  
&emsp; &emsp; &emsp; &emsp; $`\dot{\theta }=\frac{v_{r}(t)}{b}tant(\omega (t))`$

## Pipeline
- **Path planning**: RRT*
- **Control**: PID
- **Other agents detection**: LIDAR
- **Dynamic avoidance**: New path generation around the other agent
