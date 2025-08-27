# CMC — Critical Micelle Concentration Workflows & Analysis

This repository contains tools, data, and scripts to **plan**, **execute**, and **analyze** critical micelle concentration (CMC) experiments, including surfactant pairings and automated workflows.

---

## Repository Structure

- `All_data/` — experimental data (browse locally).
- `random_exploration/` — exploratory notebooks or scripts.
- `workflow_code/`
  - `analysis/`
    - `cmc_data_analysis.py` — fits I1/I3 vs. concentration, computes CMC.
    - `cmc_exp_new.py` — experiment planner.
    - `CMC_trial_assignment_with_runs.csv` — surfactant pairings & run mapping.
    - `greedy_grouped_trials.csv` — grouped trial designs.
  - `logs/` — example experiment logs.
  - `workflows/`
    - `CMC_rough_refined.py` — main workflow (rough → refined screens).
    - `cmc_shared.py` — shared helpers.
    - `cmc_repeats.py` — repeat handling utilities.

---
