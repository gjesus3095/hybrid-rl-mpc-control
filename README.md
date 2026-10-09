# Hybrid RL-IK Control

A modular hybrid control framework combining **Reinforcement Learning (RL)** for base control with **optimization-based Inverse Kinematics (IK/QP)** for robotic manipulation.

The initial setup consists of a **3-DoF planar base (X, Y, yaw)** controlled exclusively by RL and a **7-DoF Franka Panda arm** controlled by IK/QP. The controllers operate independently but remain coupled through the robot's kinematics and dynamics: the arm compensates for base motion to maintain end-effector tracking, while the RL policy accounts for changes in the system's center of mass (CoM) caused by arm movements.

The long-term goal is to extend this architecture to **humanoid whole-body control**, combining RL-based balancing and locomotion with precise model-based manipulation.
