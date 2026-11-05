**Course:** Unsupervised Machine Learning (DLBDSMLUSL01)
**Author:** Muhammad Sunusi Abubakar
**Institution:** International University of Applied Sciences IU Hochschule
**Date:** November 2026

# Mental Health in Technology-related Jobs: Unsupervised Learning

## Project Overview
This repository contains the Python scripts and data processing pipeline for the DLBDSMLUSL01 Unsupervised Learning and Feature Engineering case study. The project utilizes the OSMI Mental Health in Tech 2016 dataset to identify latent clusters of employees based on their mental health awareness, workplace benefits, and sector environments. The insights derived from these clusters inform targeted HR mitigation strategies.

## Repository Structure
* `phase1_exploration.py`: Ingests the raw CSV, calculates dimensionality, and quantifies missing value percentages across all 63 features.
* `phase2_preprocessing.py`: Executes the data cleaning pipeline, including threshold-based dropping of highly sparse columns, mode imputation for missing categorical data, dynamic text standardization (specifically for the messy 'Gender' feature), and categorical numerical encoding via `LabelEncoder`.
* `phase3_clustering.py`: Applies the K-Means algorithm and calculates Within-Cluster Sum of Squares (WCSS/Inertia) to generate an Elbow Curve, determining the optimal number of clusters ($k=4$).
* `phase4_analysis.py`: Executes the final K-Means clustering model with $k=4$ and extracts cluster centroids (averages) for key HR-related features to profile the distinct employee segments.

## Setup and Execution
1. Ensure Python 3.x is installed.
2. Create and activate a virtual environment (`venv`).
3. Install required dependencies: `pip install pandas numpy scikit-learn matplotlib`
4. Place the `mental-health-in-tech-2016.csv` dataset in the root directory.
5. Execute the scripts in sequential order (Phase 1 through 4) using the Python interpreter.

## Dependencies
* pandas
* numpy
* scikit-learn
* matplotlib
