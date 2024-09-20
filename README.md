# Hybrid Project Optimization in Chatroud

University project for calculating the optimal design of a microgrid system for Chatroud and Bootia factory. The design and size
of the following parts are calculated and optimized:

- PV Arrays
- Wind Turbines
- Batteries

The optimization is done using Genetics Algorithm to minimize the cost of the plan.

## Home Demand

For home demand the optimized result is the following:

- 201 PV arrays (MAX POWER 30MW)
- 2 Wind Turbines (MAX POWER 5MW)
- 0 Batteries

The cost for this configuration for 12 years, with inflation rate 0.05 is **$76,447,349.72**.

## Bootia Demand

Bootia has two main differences with home demand

- Bootia has a constant demand
- Bootia has its own natural gas plant, which means the price for electricity is different
- Bootia has a much higher demand than home

The optimization results for Bootia demand:

- 6440 PV arrays (MAX POWER 966MW)
- No wind turbines
- No batteries

The cost for this configuration for 12 years, with inflation rate 0.05 is **$71,487,174,846.74**.

Further data and info can be found in `plots_and_data` directory

## Usage

**IMPORTANT NOTE**: Python version should be `>3.12` for this project to work

First install the package using the following command:

```bash
pip install -r requirements.txt
```

Then run the project using python:

```bash
python3 src/main.py
```
