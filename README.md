# Monte Carlo Pricing for Exotic Options

This project implements a flexible and modular Monte Carlo simulation framework to price vanilla and exotic options. Monte Carlo pricing estimates the expected payoff of an option under risk-neutral dynamics by simulating a large number of asset paths.

This project includes variance reduction techniques, convergence analysis, and detailed visualizations to assess the efficiency and accuracy of different estimators.


---

## Features

- Pricing of vanilla and exotic options using Monte Carlo simulation
- Convergence and variance analysis with graphical outputs
- Variance reduction methods: Antithetic variates and control variates
- Modular and extensible code structure


##  Project Structure

monte-carlo-pricer/
├── models/ # Asset dynamics (e.g., GBM)
├── pricers/ # Monte Carlo pricing engine
├── products/ # Option products (European, Asian, etc.)
├── sensitivity/ # Sensitivity computation (e.g., price vs. sigma, K, T...)
├── notebook.ipynb # Demo and visualizations
├── README.md # Project documentation
├── requirements.txt # Dependencies
└── .gitignore # Files to exclude from Git