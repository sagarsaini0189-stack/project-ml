# project-ml
# AeroNav-RL: Deep Reinforcement Learning for Autonomous Drone Flight Navigation

## Project Overview
This repository contains an end-to-end autonomous drone navigation engine powered by a Deep Q-Network (DQN). The agent learns collision avoidance and optimal flight path trajectory planning inside a GPS-denied synthetic waypoint grid using temporal difference reinforcement learning.

## System Architecture
- **State Representation:** 6-dimensional normalized vector tracking drone spatial position, obstacle offset coordinates, and goal waypoint distance.
- **Action Space:** Discrete 4-directional spatial kinematic control (`Up`, `Down`, `Left`, `Right`).
- **Policy Network:** Multi-Layer Perceptron (MLP) with dense hidden layers and `ReLU` activations estimating optimal expected future Q-values.
- **Experience Replay Memory:** Decouples consecutive flight samples to break temporal correlation and guarantee stable Bellman gradient convergence.
