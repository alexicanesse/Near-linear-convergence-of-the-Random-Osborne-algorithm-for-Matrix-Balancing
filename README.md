# Random Osborne Algorithm for Matrix Balancing

This repository implements a random variant of Osborne's algorithm for Matrix Balancing, as discussed in the article by Jason M. Altschuler and Pablo A. Parrilo titled "Near-linear convergence of the Random Osborne algorithm for Matrix Balancing" [1].

## Overview

Matrix Balancing is a widely used pre-conditioning task for computing eigenvalues and matrix exponentials. Osborne's algorithm, the practitioners' choice since 1960, is now implemented in most numerical software packages. However, its theoretical properties are not well understood. The article demonstrates that a simple random variant of Osborne's algorithm converges in near-linear time in input sparsity.

## Article Details

- **Title**: Near-linear convergence of the Random Osborne algorithm for Matrix Balancing
- **Authors**: Jason M. Altschuler, Pablo A. Parrilo
- **Published in**: *Mathematical Programming*
- **Volume**: 198, **Issue**: 1, **Pages**: 363-397, **Year**: 2023
- **DOI**: [10.1007/s10107-022-01825-4](https://doi.org/10.1007/s10107-022-01825-4)

## Usage

To apply the random Osborne algorithm to your matrix balancing task:

1. **Read the Article**: Familiarize yourself with the theoretical underpinnings discussed in the [research article](https://doi.org/10.1007/s10107-022-01825-4).
2. **Clone the Repository**: Clone this repository to your local machine.

   ```bash
   git clone git@github.com:alexicanesse/Near-linear-convergence-of-the-Random-Osborne-algorithm-for-Matrix-Balancing.git
   ```
3. **Install Dependencies** : Ensure you have the necessary dependencies installed. You can find them in the `requirements.txt` file.

   ```bash
   pip install -r requirements.txt
   ```
4. Utilize the provided notebook to implement the random Osborne algorithm on your matrix balancing task.

## Contributor

- Alexi Canesse (alexi.canesse@ens-lyon.fr)
