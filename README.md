# Numerical Solution of Two-Dimensional Pollutant Transport Problems in Soils

This repository contains a Python implementation for mathematical modeling and numerical simulation of 2D pollutant transport and convection-diffusion processes in porous soil structures. The project focuses on solving partial differential equations (PDEs) to predict mass transfer and contaminant distribution over time.

**Mathematical Modeling:** Implementation of 2D convection-diffusion equations with boundary conditions.
**Numerical Methods:** Utilization of finite difference / finite volume schemes to transform continuous models into stable discrete computational pipelines.
**Data Visualization:** High-quality 2D heatmaps and contour plots generated with `Matplotlib` to analyze pollutant concentration dynamics.
**Performance Analysis:** Optimized matrix operations and scientific computation using `NumPy` and `SciPy`.

## Tech Stack & Libraries
**Python 3.x** - Core programming language
**NumPy** - Multi-dimensional arrays and fast matrix computations
**SciPy** - Scientific computing and numerical integration
**Matplotlib** - Data visualization and plotting simulation grids

## Project Structure
* `main.py` — Entry point of the simulation.
* `solver.py` — Implementation of numerical methods and PDE solving algorithms.
* `config.py` — Boundary conditions, soil coefficients, and grid parameters.
* `plots/` — Directory containing exported visualization plots.

## Setup & Installation

1. Clone the repository:
```bash
git clone [https://github.com/turchakslavik228-hash/soil-pollutant-transport-simulation.git](https://github.com/turchakslavik228-hash/soil-pollutant-transport-simulation.git)
cd soil-pollutant-transport-simulation