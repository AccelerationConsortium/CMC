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

def CMC_estimate (s1,  s2 = 'None', s3 = 'None', s1_ratio = 'None', s2_ratio = 'None'):

    if s2 == 'None' and s3 == 'None':
        # Only one surfactant
        cmc1 = surfactant_library[s1]['CMC']
        return cmc1
    
    elif s2 != 'None' and s3 == 'None':
        # Two surfactants
        cmc1 = surfactant_library[s1]['CMC']
        cmc2 = surfactant_library[s2]['CMC']
        return cmc1 * s1_ratio + cmc2 * (1-s1_ratio)
    
    elif s2 != 'None' and s3 != 'None':
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

    
