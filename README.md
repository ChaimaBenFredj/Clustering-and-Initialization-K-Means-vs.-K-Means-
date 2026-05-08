# K-Means vs K-Means++ — Clustering and Initialization

Experimental project for the *Statistical Methods for Machine Learning* course (A.Y. 2025/26).  
Instructor: Nicolò Cesa-Bianchi — University of Milan.

## Project Overview

This project investigates the impact of initialization on K-Means clustering performance 
by comparing standard K-Means (random initialization) with K-Means++ (smart initialization) 
across 50 independent trials on two datasets.

Both algorithms are implemented from scratch using only NumPy.

## Repository Structure

| File | Description |
|------|-------------|
| `datasets.py` | Loads and visualizes the two datasets: Synthetic blobs and the Digits dataset |
| `kmeans.py` | K-Means algorithm implemented from scratch |
| `kmeans_plus_plus.py` | K-Means++ algorithm implemented from scratch |
| `experiments.py` | Runs 50 trials for each algorithm and measures inertia and variability |
| `plots.py` | Generates all the report's plots |
| `ChaimaBenFredj_ML_Report.pdf` | Full written report |

## How to Run

Run the files in this order:

```bash
python datasets.py         # to verify datasets load correctly
python kmeans.py           # to implement K-Means 
python kmeans_plus_plus.py # to implement test K-Means++ 
python experiments.py      # to run full 50-trial comparison
python plots.py            # to generate our plots
```
