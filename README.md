# CMC Experimental Planning and Analysis Toolkit

This repository contains helper functions to design, execute, and analyze Critical Micelle Concentration (CMC) experiments involving mixtures of up to three surfactants. The code supports CMC estimation, solution preparation, and automated protocol generation for liquid handling robots like the OT-2 Flex.

## 📦 Files Overview

There are four main helper Python files:

### 1. `cmc_data_analysis.py`

- Fits the I3/I1 fluorescence intensity ratio as a function of surfactant concentration.
- Determines the CMC (critical micelle concentration) from the curve.
- Returns the fitting scores and calculated CMC.

---

### 2. `cmc_exp.py`

This is the core module for preparing and calculating experimental conditions for CMC measurements.

**Main Components:**

- `surfactant_library`:  
  A dictionary containing:
  - Surfactant name
  - CAS number
  - Reported single-surfactant CMC
  - Surfactant category

- `CMC_estimate`:  
  - Takes a list of surfactants and their ratios.
  - Uses reported single-surfactant CMC values to estimate the CMC of the mixture.
  - If only a single surfactant is used, returns its reported CMC directly.

- `generate_cmc_concentrations`:  
  - Generates 12 concentrations centered around the estimated CMC.
  - Uses fewer points far from the CMC and more points densely around it for better resolution.

- `surfactant_substock`:  
  - Prepares a substock surfactant mixture close to the CMC.
  - This helps minimize pipetting error in liquid handling.
  - Returns the substock concentration and volumes of water + surfactants required.

- `calculate_volumes`:  
  - Calculates the volumes of substock, water, and pyrene for each sample.
  - Based on concentrations generated from `generate_cmc_concentrations`.

- `generate_exp`:  
  - Combines all of the above functions.
  - Converts a list of surfactants and their ratios into the required components and volumes to:
    - Prepare the substock
    - Prepare each concentration sample
  - Returns two dictionaries:
    - `exp`: Full experiment data
    - `small_exp`: Only essential data needed to generate an OT-2 Flex protocol

---

### 3. `cmc_generate_protocol.py`

- Takes the `small_exp` dictionary from `cmc_exp.py`
- Automatically updates `cmc_OTFlex_protocol.py` with the correct volumes for a specific CMC experiment

---

### 4. `cmc_OTFlex_protocol.py`

- A template protocol for OT-2 Flex robots.
- Gets overwritten by `cmc_generate_protocol.py` to reflect your experiment setup.

---

## 📓 Example Usage

A user case is provided in the file:  
📄 **`20250417_example.ipynb`**

This notebook walks through:

1. Selecting surfactants and defining their mixing ratios
2. Estimating mixed CMC
3. Generating concentration series
4. Calculating all required volumes
5. Producing a ready-to-run OT-2 Flex protocol
