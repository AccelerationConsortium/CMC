# CMC — Critical Micelle Concentration Workflow

This repository hosts data and code for planning, running, and analyzing **critical micelle concentration (CMC)** experiments, including both **single-surfactant** and **mixed-surfactant** studies.

---

## Repository Structure

### 1) `all_data/`
All CMC data collected using this workflow:
- **Contents:** raw data for single CMCs and mixed CMCs.
- **Analysis notebooks:** a Jupyter notebook that aggregates the data **using `CMC_helper_function.py`** for data visualization and summary plots.

### 2) `random_exploration/`
Early, exploratory work from the initial workflow development:
- Includes the **initial setup on an OTFlex system** and some preliminary data analysis.
- **Note:** this folder is legacy and **will be archived**.

---

## Notes
- The `random_exploration/` directory is kept for provenance and will be archived once its contents are fully superseded by the current workflow.
- For visualizing combined datasets, start with the notebook in `all_data/` which relies on `CMC_helper_function.py`.

---
