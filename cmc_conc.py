import numpy as np
import pandas as pd

# surfactant_library
surfactant_library = {
    "surfactant_1": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": 0.1,
        "SMILES": "tba"
    },
    "surfactant_2": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": 10,
        "SMILES": "tba"
    },
    "surfactant_3": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": 20,
        "SMILES": "tba"
    },
    "surfactant_4": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
    "surfactant_5": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
    "surfactant_6": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
    "surfactant_7": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
    "surfactant_8": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
    "surfactant_9": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
    "surfactant_10": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
    "surfactant_11": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
    "surfactant_12": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
}

def CMC_estimate(surfactants, ratios):
    cmc_total = 0.0
    for surfactant, ratio in zip(surfactants, ratios):
        if surfactant is not None:
            cmc = surfactant_library[surfactant]['CMC']
            cmc_total += cmc * ratio
    return cmc_total



def generate_cmc_concentrations(cmc):
    """
    Generate 12 concentration points from cmc/3 to cmc*3.
    """
    # Log-spaced: from cmc/10 to cmc/1.5 (3 points)
    below = np.logspace(np.log10(cmc / 3), np.log10(cmc / 1.5), 3)

    # Linearly spaced: ±25% around CMC (6 points)
    around = np.linspace(cmc * 0.75, cmc * 1.25, 6)

    # Log-spaced: from cmc*1.5 to cmc*10 (3 points)
    above = np.logspace(np.log10(cmc * 1.5), np.log10(cmc * 3), 3)

    return np.concatenate([below, around, above]).tolist()

    
def surfactant_mix(cmc_concs, s1, s2=None, s3=None, s1_ratio=None, s2_ratio=None,
                   s1_stock=50, s2_stock=50, s3_stock=50):
    """
    Calculate the volumes of up to three surfactants and water needed to prepare a surfactant mix.

    Parameters:
    - cmc_concs: list of CMC concentrations (in mM)
    - s1, s2, s3: surfactant names or labels (only s1 required)
    - s1_ratio, s2_ratio: component ratios (s3 ratio = 1 - s1 - s2)
    - s1_stock, s2_stock, s3_stock: stock concentrations in mM
    - mix_stock: desired stock concentration of the mix (default: 50 mM)

    Returns:
    - Dict of surfactant volumes and water volume (µL)
    """

    # Constants
    probe_volume = 3  # µL
    total_cmc_volume = 300  # µL
    final_volume = 1000  # µL

    # Calculate CoC (Critical overall concentration)
    max_cmc_conc = max(cmc_concs)
    mix_stock_conc = max_cmc_conc / ((total_cmc_volume - probe_volume) / total_cmc_volume)

    # Handle ratios
    if s2 is None:
        s1_ratio = 1.0
        s2_ratio = 0.0
        s3_ratio = 0.0
    elif s3 is None:
        if s1_ratio is None:
            raise ValueError("s1_ratio must be specified for two-component system.")
        s2_ratio = 1 - s1_ratio
        s3_ratio = 0.0
    else:
        if s1_ratio is None or s2_ratio is None:
            raise ValueError("s1_ratio and s2_ratio must be specified for three-component system.")
        s3_ratio = 1 - s1_ratio - s2_ratio
        if not (0 <= s3_ratio <= 1):
            raise ValueError("Invalid ratios: sum of s1_ratio and s2_ratio exceeds 1.")

    # Total moles needed
    total_mmol = mix_stock_conc * final_volume / 1000  # mmol

    # Calculate volumes
    result = {}

    v1 = (total_mmol * s1_ratio) / (s1_stock / 1000)
    result[s1] = v1

    v2 = 0
    v3 = 0

    if s2:
        v2 = (total_mmol * s2_ratio) / (s2_stock / 1000)
        result[s2] = v2

    if s3:
        v3 = (total_mmol * s3_ratio) / (s3_stock / 1000)
        result[s3] = v3

    # Water is just the remainder
    water = final_volume - v1 - v2 - v3
    result['Water'] = water

    return mix_stock_conc, result



def calculate_volumes(concentration_list, stock_concentration):
    total_volume = 300
    probe_volume = 3

    concentrations = []
    surfactant_volumes = []
    water_volumes = []
    probe_volumes = []
    total_volumes = []

    for conc in concentration_list:
        surfactant_volume = (conc * (total_volume - probe_volume)) / stock_concentration
        water_volume = total_volume - probe_volume - surfactant_volume

        # Round for consistency
        surfactant_volume = round(surfactant_volume, 2)
        water_volume = round(water_volume, 2)

        # Collect values
        concentrations.append(conc)
        surfactant_volumes.append(surfactant_volume)
        water_volumes.append(water_volume)
        probe_volumes.append(probe_volume)
        total_volumes.append(round(surfactant_volume + water_volume + probe_volume, 2))

    df = pd.DataFrame({
        "concentration": concentrations,
        "surfactant volume": surfactant_volumes,
        "water volume": water_volumes,
        "probe volume": probe_volumes,
        "total volume": total_volumes
    })

    return df



