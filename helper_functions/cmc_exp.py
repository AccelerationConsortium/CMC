import numpy as np
import pandas as pd



# surfactant_library
surfactant_library = {
    "SDS": {
        "full_name": "Sodium Dodecyl Sulfate",
        "CAS": "151-21-3",
        "CMC": 8.3,
        "Category": "anionic",
    },


    "SLS": {
        "full_name": "Sodium Lauryl Sulfate",
        "CAS": "151-21-3",
        "CMC": 7.7,
        "Category": "anionic",
    },


    "NaDC": {
        "full_name": "Sodium Docusate",
        "CAS": "577-11-7",
        "CMC": 8.2,
        "Category": "anionic",
    },

    
    "NaC": {
        "full_name": "Sodium Cholate",
        "CAS": "361-09-1",
        "CMC": 11,
        "Category": "anionic",
    },


    "CTAB": {
        "full_name": "Hexadecyltrimethylammonium Bromide",
        "CAS": "57-09-0",
        "CMC": 0.93,
        "Category": "cationic",
    },


    "DTAB": {
        "full_name": "Dodecyltrimethylammonium Bromide",
        "CAS": "1119-94-4",
        "CMC": 15.85,
        "Category": "cationic",
    },


    "TTAB": {
        "full_name": "Tetradecyltrimethylammonium Bromide",
        "CAS": "1119-97-7",
        "CMC": 3.77,
        "Category": "cationic",
    },


    "BAC": {
        "full_name": "Benzalkonium Chloride",
        "CAS": "63449-41-2",
        "CMC": 0.42,
        "Category": "cationic",
    },


    "T80": {
        "full_name": "Tween 80",
        "CAS": "9005-65-6",
        "CMC": 0.015,
        "Category": "nonionic",
    },

    
    "T20": {
        "full_name": "Tween 20",
        "CAS": "9005-64-5",
        "CMC": 0.0355,
        "Category": "nonionic",
    },


    "P188": {
        "full_name": "Kolliphor® P 188 Geismar",
        "CAS": "9003-11-6",
        "CMC": 0.325,
        "Category": "nonionic",
    },


    "P407": {
        "full_name": "Kolliphor® P 407 Geismar",
        "CAS": "9003-11-6",
        "CMC": 0.1,
        "Category": "nonionic",
    },

    "CAPB": {
        "full_name": "Cocamidopropyl Betaine",
        "CAS": "61789-40-0",
        "CMC": 0.627,
        "Category": "zwitterionic",
    },

    "SBS-12": {
        "full_name": "Sulfobetaine-12",
        "CAS": "14933-08-5",
        "CMC": 3,
        "Category": "zwitterionic",
    },

    "SBS-14": {
        "full_name": "Sulfobetaine-14",
        "CAS": "14933-09-6",
        "CMC": 0.16,
        "Category": "zwitterionic",
    },
    
    "CHAPS": {
        "full_name": "CHAPS",
        "CAS": "75621-03-3",
        "CMC": 8.5,
        "Category": "zwitterionic",
    }
}

def CMC_estimate(list_of_surfactants, list_of_ratios):

    cmc_total = 0.0
    for surfactant, ratio in zip(list_of_surfactants, list_of_ratios):
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

    
def surfactant_mix(cmc_concs, list_of_surfactants, list_of_ratios,
                   stock_concs=[50, 50, 50]):
    """
    Calculate the volumes of up to three surfactants and water needed to prepare a surfactant mix.

    Parameters:
    - cmc_concs: list of CMC concentrations (in mM)
    - list_of_surfactants: list of 3 surfactant names or None
    - list_of_ratios: list of 3 ratios (must sum <= 1)
    - stock_concs: list of 3 stock concentrations in mM

    Returns:
    - Tuple of (mix_stock_conc, dict with 3 surfactants (named or placeholders) and water, values in µL)
    """

    # Constants
    probe_volume = 3  # µL
    total_cmc_volume = 300  # µL
    final_volume = 1800  # µL


    # Calculate adjusted mix stock concentration
    max_cmc_conc = max(cmc_concs)
    mix_stock_conc = max_cmc_conc / ((total_cmc_volume - probe_volume) / total_cmc_volume)

    # Total moles needed
    total_mmol = mix_stock_conc * final_volume / 1000  # mmol

    result = {}
    total_surfactant_volume = 0

    for i in range(3):
        surf = list_of_surfactants[i]
        ratio = list_of_ratios[i]
        stock = stock_concs[i]

        # Assign placeholder if name is None
        surf_label = surf if surf is not None else f"None_{i+1}"

        # Calculate volume or assign 0
        volume = (total_mmol * ratio) / (stock / 1000) if ratio > 0 else 0

        result[surf_label] = volume
        total_surfactant_volume += volume

    result['water'] = final_volume - total_surfactant_volume

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

def generate_exp(list_of_surfactants, list_of_ratios, stock_concs=[50, 50, 50]):

        # Validations
    if len(list_of_surfactants) != 3 or len(list_of_ratios) != 3 or len(stock_concs) != 3:
        raise ValueError("Inputs must all have 3 elements.")
    if sum(list_of_ratios) != 1:
        raise ValueError("Sum of surfactant ratios must be == 1")

    estimated_CMC = CMC_estimate(list_of_surfactants, list_of_ratios)
    cmc_concentrations = generate_cmc_concentrations(estimated_CMC)
    mix_stock_conc, mix_stock_vol = surfactant_mix(cmc_concentrations, list_of_surfactants, list_of_ratios, stock_concs=stock_concs)
    df = calculate_volumes(cmc_concentrations, mix_stock_conc)

    exp = {
        "list_of_surfactants": list_of_surfactants,
        "list_of_ratios": list_of_ratios,
        "original_surfactant_stock_concs": stock_concs,
        "surfactant_mix_stock_conc": mix_stock_conc,
        "surfactant_mix_stock_vols": mix_stock_vol,
        "estimated_CMC": estimated_CMC,
        "df": df,
    }

    small_exp = {

        "surfactant_mix_stock_vols": mix_stock_vol,
        "solvent_mix_vol": df["surfactant volume"].tolist(),
        "water_vol": df["water volume"].tolist(),
        "pyrene_vol": df["probe volume"].tolist(),
    }

    return exp, small_exp


