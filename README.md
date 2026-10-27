# Detecting and Predicting Earthquake Ground Motion Directionality Patterns Using ML

Mini-Project for **UE24CS352A - Machine Learning**.
Based on the research paper by Alan Poulos and Rodrigo Silva-Lopez (Blume Earthquake Engineering Center, Stanford University).

## Overview

Horizontal ground shaking intensity during an earthquake varies with orientation ($\text{RotD100}$). While the orientation at a single site is approximately uniform, nearby sites ($\le 5\text{ km}$) exhibit spatial correlation. This project predicts the absolute angular difference $\Delta\theta \in [0^\circ, 90^\circ]$ between the maximum ground shaking orientations of two recording sites $(s_i, s_j)$ during an earthquake $E_k$:

$$
\Delta\theta_{i,j}^k = f(s_i, s_j, E_k)
$$
