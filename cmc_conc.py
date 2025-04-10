import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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

def CMC_estimate(list_of_surfactants, list_of_ratios):

        # Sanity checks
    if len(list_of_surfactants) != 3 or len(list_of_ratios) != 3:
        raise ValueError("Both 'list_of_surfactants' and 'list_of_ratios' must have exactly 3 elements.")
    if sum(list_of_ratios) > 1:
        raise ValueError("Sum of surfactant ratios must be <= 1.")
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
    final_volume = 1000  # µL

    # Validations
    if len(list_of_surfactants) != 3 or len(list_of_ratios) != 3 or len(stock_concs) != 3:
        raise ValueError("Inputs must all have 3 elements.")
    if sum(list_of_ratios) > 1:
        raise ValueError("Sum of surfactant ratios must be <= 1.")

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

    result['Water'] = final_volume - total_surfactant_volume

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

    estimated_CMC = CMC_estimate(list_of_surfactants, list_of_ratios)
    cmc_concentrations = generate_cmc_concentrations(estimated_CMC)
    mix_stock_conc, mix_stock_vol = surfactant_mix(cmc_concentrations, list_of_surfactants, list_of_ratios, stock_concs=stock_concs)
    df = calculate_volumes(cmc_concentrations, mix_stock_conc)

    exp = {
        "list_of_surfactants": list_of_surfactants,
        "list_of_ratios": list_of_ratios,
        "stock_concs": stock_concs,
        "mix_stock_conc": mix_stock_conc,
        "mix_stock_vol": mix_stock_vol,
        "estimated_CMC": estimated_CMC,
        "cmc_concentrations": cmc_concentrations,
        "mix_stock_conc": mix_stock_conc,
        "mix_stock_vol": mix_stock_vol,
        "df": df,
    }

    return exp




# Define the Boltzmann sigmoid function
def boltzmann(x, A1, A2, x0, dx):
    return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))

def CMC_plot(i1_i3_ratio, conc):
    # Initial guess for parameters [A1, A2, x0, dx]
    p0 = [max(i1_i3_ratio), min(i1_i3_ratio), (max(conc) - min(conc))/2, (max(conc) - min(conc)) / 5]

    # Fit the data to the Boltzmann sigmoid
    popt, pcov = curve_fit(boltzmann, conc, i1_i3_ratio, p0, maxfev=5000)
    A1, A2, x0, dx = popt

    # Compute the second CMC (xCMC2) using the derived formula
    xCMC2 = x0 + dx * np.log((A1 - A2) / (0.5 * (A1 - A2)))

    # Compute R-squared value for goodness of fit
    residuals = i1_i3_ratio - boltzmann(conc, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((i1_i3_ratio - np.mean(i1_i3_ratio))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Plot the data and fitted curve
    x_fit = np.linspace(min(conc), max(conc), 100)
    y_fit = boltzmann(x_fit, *popt)

    plt.figure(figsize=(8, 6))
    plt.scatter(conc, i1_i3_ratio, label='Experimental Data', color='blue')
    plt.plot(x_fit, y_fit, label='Boltzmann Fit', color='red')
    plt.axvline(x0, linestyle='--', color='green', label=f'(xCMC)1 = {x0:.2f} mM')
    plt.axvline(xCMC2, linestyle='--', color='purple', label=f'(xCMC)2 = {xCMC2:.2f} mM')
    plt.xlabel('SDS Concentration (mM)')
    plt.ylabel('I₁/I₃ Ratio')
    plt.title('CMC Determination using Boltzmann Fit')
    plt.legend()
    plt.grid()
    plt.show()

    # Output the computed CMC values and R-squared
    print(f'Estimated (xCMC)1: {x0:.2f} mM')
    print(f'Estimated (xCMC)2: {xCMC2:.2f} mM')
    print(f'Average (xCMC)1/2: {(x0 + xCMC2)/2:.2f} mM')
    print(f'Fit Accuracy (R²): {r_squared:.4f}')