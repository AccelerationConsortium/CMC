# CMC Experimental Planning and Analysis Toolkit

This toolkit helps plan, execute, and analyze Critical Micelle Concentration (CMC) experiments involving up to three surfactants. It includes functions to estimate mixed CMCs, prepare sub-stocks, generate protocols for automated systems, and analyze experimental results.

## 📁 Repository Structure

- **`cmc_data_analysis.py`**  
  Fits the I1/I3 fluorescence intensity ratio as a function of surfactant concentration and determines the CMC along with goodness-of-fit scores.

- **`cmc_exp.py`**  
  Provides a comprehensive set of functions to convert a surfactant mixture (up to 3 components with user-defined ratios) into actionable volumes for experimental preparation.
  
  **Key functions:**
  - `surfactant_library`: Dictionary storing surfactant name, CAS number, reported CMC, and surfactant category.
  - `CMC_estimate`: Estimates the mixed CMC from a list of surfactants and ratios.
  - `generate_cmc_concentrations`: Spreads 12 concentrations with strategic density around the estimated CMC.
  - `surfactant_substock`: Prepares a substock mixture of surfactants close to the CMC to simplify liquid handling.
  - `calculate_volumes`: Calculates volumes of substock, water, and pyrene for target concentrations.
  - `generate_exp`: Integrates the above to generate a complete plan (output: `exp` and `small_exp` dictionaries).

- **`cmc_generate_protocol.py`**  
  Uses the `small_exp` dictionary to modify and generate the `cmc_OTFlex_protocol.py` with updated volume instructions for a specific experiment.

- **`cmc_OTFlex_protocol.py`**  
  Template protocol for use with an OT-2 Flex liquid handling robot, which gets customized by `cmc_generate_protocol.py`.

- **`20250417_example.ipynb`**  
  Example notebook demonstrating how to use the toolkit end-to-end, including planning and protocol generation.

