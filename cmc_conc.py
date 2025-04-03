import numpy as np
# surfactant_library

surfactant_library = {
    "surfactant_1": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
    "surfactant_2": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
        "SMILES": "tba"
    },
    "surfactant_3": {
        "MW": "tba",
        "CAS": "tba",
        "CMC": "tba",
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

def CMC_estimate (s1,  s2=None, s3=None, s1_ratio=None, s2_ratio=None):

    if s2 == None and s3 == None:
        # Only one surfactant
        cmc1 = surfactant_library[s1]['CMC']
        return cmc1
    
    elif s2 != None and s3 == None:
        # Two surfactants
        cmc1 = surfactant_library[s1]['CMC']
        cmc2 = surfactant_library[s2]['CMC']
        return cmc1 * s1_ratio + cmc2 * (1-s1_ratio)
    
    elif s2 != None and s3 != None:
        # Three surfactants
        cmc1 = surfactant_library[s1]['CMC']
        cmc2 = surfactant_library[s2]['CMC']
        cmc3 = surfactant_library[s3]['CMC']
        return cmc1 * s1_ratio + cmc2 * s2_ratio + cmc3 * (1-s1_ratio-s2_ratio)


def generate_cmc_concentrations(cmc):
    """
    Generate 12 concentration points from cmc/10 to cmc*10.
    """
    # Log-spaced: from cmc/10 to cmc/1.5 (4 points)
    below = np.logspace(np.log10(cmc / 10), np.log10(cmc / 1.5), 4)

    # Linearly spaced: ±25% around CMC (4 points)
    around = np.linspace(cmc * 0.75, cmc * 1.25, 4)

    # Log-spaced: from cmc*1.5 to cmc*10 (4 points)
    above = np.logspace(np.log10(cmc * 1.5), np.log10(cmc * 10), 4)

    return np.concatenate([below, around, above]).tolist()

    
def surfactant_mix(cmc_concs, s1, s2=None, s3=None, s1_ratio=None, s2_ratio=None,
                   s1_stock=50, s2_stock=50, s3_stock=50, mix_stock=50):
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
    coc = max_cmc_conc / ((total_cmc_volume - probe_volume) / total_cmc_volume)

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
    total_mmol = coc * final_volume / 1000  # mmol

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

    return result

