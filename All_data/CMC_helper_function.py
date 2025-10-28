import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from scipy.optimize import curve_fit
import math
import os


surfactant_library = {
    "SDS": {
        "full_name": "Sodium Dodecyl Sulfate",
        "CAS": "151-21-3",
        "CMC": 8.5,
        "Category": "anionic",
        "MW": 289.39,
        "stock_conc": 50,  # mM
        "low": 7,
        "high": 10,
        "ref_1": 8.3,
        "ref_2": 8.25,
        "ref_3": 7
    },


    "DSS": {
        "full_name": "Sodium Docusate",
        "CAS": "577-11-7",
        "CMC": 5.3375,
        "Category": "anionic",
        "MW": 445.57,
        "stock_conc": 25,  # mM
        "low":2.48,
        "high": 8.2,
        "ref_1": 8.2,
        "ref_2": 2.475,
        "ref_3": 2.48
    },

    
    "NaC": {
        "full_name": "Sodium Cholate",
        "CAS": "361-09-1",
        "CMC": 14,
        "Category": "anionic",
        "MW": 431.56,
        "stock_conc": 50,  # mM
        "low": 13,
        "high":15,
        "ref_1": 14,
        "ref_2": 14,
        "ref_3": 12,
    },


    "CTAB": {
        "full_name": "Hexadecyltrimethylammonium Bromide",
        "CAS": "57-09-0",
        "CMC": 1.07,
        "Category": "cationic",
        "MW": 364.45,
        "stock_conc": 5, # mM
        "low": 0.9,
        "high": 1.24,
        "ref_1": 0.92,
        "ref_2": 1.24,
        "ref_3": 0.9,
    },


    "DTAB": {
        "full_name": "Dodecyltrimethylammonium Bromide",
        "CAS": "1119-94-4",
        "CMC": 15.85,
        "Category": "cationic",
        "MW": 308.34,
        "stock_conc": 50,  # mM
        "low": 15.7,
        "high": 16,
        "ref_1": 15.7,
        "ref_2": 16,
        "ref_3": 16,
    },


    "TTAB": {
        "full_name": "Tetradecyltrimethylammonium Bromide",
        "CAS": "1119-97-7",
        "CMC": 3.985,
        "Category": "cationic",
        "MW": 336.39,
        "stock_conc": 50,  # mM
        "low": 3.77,
        "high": 4.2,
        "ref_1": 4.2,
        "ref_2": 3.8,
        "ref_3": 3.77,
    },

    "CAPB": {
        "full_name": "Cocamidopropyl Betaine",
        "CAS": "61789-40-0",
        "CMC": 0.627,
        "Category": "zwitterionic",
        "MW": 342.52,
        "stock_conc": 50,  # mM
        "low": 0.28,
        "high": 0.974,
        "ref_1": 0.28,
        "ref_2": 0.974,
        "ref_3": 0.881,
    },
    
    "CHAPS": {
        "full_name": "CHAPS",
        "CAS": "75621-03-3",
        "CMC": 8,
        "Category": "zwitterionic",
        "MW": 614.88,
        "stock_conc": 30,  # mM
        "low": 6,
        "high": 10,
        "ref_1": 6.41,
        "ref_2": 8,
        "ref_3": 8,
    }
}

SURFACTANT_ORDER = ['SDS', 'DSS', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS']

red = '#e64b35'
blue='#4dbbd5'


# Boltzmann sigmoidal function
def boltzmann(x, A1, A2, x0, dx):
    return A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx))

# CMC_plot function as provided
def CMC_plot(ax, ratio, conc ,log=1, plot=1):

    if log:
        conc = np.log(conc)

    p0 = [
        max(ratio),
        min(ratio),
        (max(conc) + min(conc)) / 2,
        (max(conc) - min(conc)) / 5
    ]
    popt, _ = curve_fit(boltzmann, conc, ratio, p0, maxfev=5000)
    A1, A2, x0, dx = popt

    # compute R²
    residuals = ratio - boltzmann(conc, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ratio - np.mean(ratio))**2)
    r2 = 1 - ss_res / ss_tot


    if log:
        x0 = math.exp(x0)

    if plot:
        # generate fit curve
        x0 = np.log(x0)
        x_fit = np.linspace(min(conc), max(conc), 200)

        # scatter with blue face and thin black edge
        ax.scatter(
            conc, ratio,
            s=25,
            facecolors=blue,
            edgecolors='black',
            linewidth=0.5,
            label='Data'  # optional
        )

        # fitted Boltzmann curve in blue
        ax.plot(
            x_fit,
            boltzmann(x_fit, *popt),
            color=blue,
            lw=1,
            label='Fit'   # optional
        )

        # vertical CMC line in blue
        ax.axvline(
            x0,
            linestyle='--',
            color=blue,
            lw=1,
            label='CMC'   # optional
        )

        x0 = math.exp(x0)
        # annotate stats
        ax.text(
            0.95, 0.95,
            f"CMC = {x0:.2f}\n$R^2$ = {r2:.3f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                    ec="black", alpha=0.7)
        )
    return x0, r2


def reliability_analysis(df,
                         surf_col='surfactant',
                         rep_col='replicate',
                         assay_col='assay',
                         time_col='time',
                         conc_col='concentration',
                         ratio_col='ratio'):

    # enforce surfactant as an ordered categorical
    df[surf_col] = pd.Categorical(df[surf_col],
                                  categories=SURFACTANT_ORDER,
                                  ordered=True)

    # — Step A: get one CMC per surf×rep×assay×time —
    records = []
    for surf in SURFACTANT_ORDER:
        sub_s = df[df[surf_col] == surf]
        if sub_s.empty:
            continue
        for rep in sub_s[rep_col].unique():
            for assay in sub_s[assay_col].unique():
                for t in sub_s[time_col].unique():
                    sub = sub_s[(sub_s[rep_col]==rep) &
                                (sub_s[assay_col]==assay) &
                                (sub_s[time_col]==t)]
                    if sub.empty:
                        continue
                    cmc, _ = CMC_plot(None,
                                      sub[ratio_col].values,
                                      sub[conc_col].values,
                                      log=1, plot=0)
                    records.append({
                        surf_col:  surf,
                        rep_col:   rep,
                        assay_col: assay,
                        time_col:  t,
                        'CMC':     cmc
                    })
    summary_df = pd.DataFrame.from_records(records)

    # helper for CV
    cv = lambda x: x.std()/x.mean() if len(x)>1 and x.mean()!=0 else np.nan

    # — Step B: compute CVs + mean/std per surfactant —
    out = []
    for surf, grp in summary_df.groupby(surf_col):
        # CV by replicate
        cv_rep = (grp
                  .groupby([assay_col, time_col])['CMC']
                  .apply(cv)
                  .dropna().mean())

        # CV by assay
        cv_assay = (grp
                    .groupby([rep_col, time_col])['CMC']
                    .apply(cv)
                    .dropna().mean())

        # CV by time
        cv_time = (grp
                   .groupby([rep_col, assay_col])['CMC']
                   .apply(cv)
                   .dropna().mean())

        # overall mean & std of all repeats
        mean_all = grp['CMC'].mean()
        mean_10min = grp[grp[time_col]==10]['CMC'].mean()
        std_all  = grp['CMC'].std()
        std_10min  = grp[grp[time_col]==10]['CMC'].std()

        # overall CV
        cv_overall = std_all/mean_all if len(grp)>1 and mean_all!=0 else np.nan

        out.append({
            surf_col:         surf,
            'replicate CV': cv_rep,
            'assay CV':     cv_assay,
            'time CV':      cv_time,
            'overall CV':      cv_overall,
            'measured CMC':        mean_all,
            'measured CMC STD':         std_all,
            'measured CMC 10min': mean_10min,
            'measured CMC STD 10min':  std_10min
        })

    result = pd.DataFrame(out)

    # enforce row order
    result[surf_col] = pd.Categorical(
        result[surf_col],
        categories=SURFACTANT_ORDER,
        ordered=True
    )
    result = result.sort_values(by=surf_col).reset_index(drop=True)

    result['literature low']  = result[surf_col].map(lambda s: surfactant_library[s]['low'])
    result['literature high'] = result[surf_col].map(lambda s: surfactant_library[s]['high'])
    
    return result


def plot_cmc_comparison(
    df,
    surfactant_col='surfactant',
    surfactant_order=SURFACTANT_ORDER,
    offset=0.1,
    measured_color = blue,
    literature_color= red,
    marker_size=8,
    line_width=4
):

    # Clean and filter names
    df_plot = df.copy()
    df_plot[surfactant_col] = df_plot[surfactant_col].astype(str).str.strip()
    df_plot = df_plot[df_plot[surfactant_col].isin(surfactant_order)]

    fig, ax = plt.subplots(figsize=(8, len(surfactant_order)*0.45))

    for _, row in df_plot.iterrows():
        surf = row[surfactant_col]
        idx = surfactant_order.index(surf)
        mean, std = row['measured CMC 10min'], row['measured CMC STD 10min']
        lit_low, lit_high = row['literature low'], row['literature high']

        # Measured CMC range: fill with group color, border in black
        ax.plot(
            [mean - std, mean + std],
            [idx + offset, idx + offset],
            marker='o',
            markersize=marker_size,
            linewidth=line_width*0.6,
            color=measured_color,
            markerfacecolor=measured_color,
            markeredgecolor='black',
            label='Measured' if idx == 0 else ""
        )
        # Literature CMC range: fill with group color, border in black
        ax.plot(
            [lit_low, lit_high],
            [idx - offset, idx - offset],
            marker='o',
            markersize=marker_size,
            linewidth=line_width*0.6,
            color=literature_color,
            markerfacecolor=literature_color,
            markeredgecolor='black',
            label='Literature' if idx == 0 else ""
        )

    # Set y-ticks and labels, then invert y-axis so first item is at top
    ax.set_yticks(range(len(surfactant_order)))
    ax.set_yticklabels(surfactant_order)
    ax.invert_yaxis()

    ax.set_xlabel('CMC Value (mM)')

    # reverse the two labels
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])

    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    return fig, ax


def plot_cmc_rmse(table1, surfactant_library):

    rmse_results = []

    for _, row in table1.iterrows():
        surf_name = row['surfactant_1']
        measured = row['measured CMC 10min']
        
        if surf_name in surfactant_library:
            refs = surfactant_library[surf_name]
            ref_values = [refs[k] for k in ['ref_1', 'ref_2', 'ref_3'] if k in refs and refs[k] is not None]
            if len(ref_values) > 0:
                rmse = np.sqrt(np.mean((np.array(ref_values) - measured) ** 2))
            else:
                rmse = np.nan
        else:
            rmse = np.nan
        
        rmse_results.append({'surfactant': surf_name, 'RMSE': rmse})
    
    rmse_df = pd.DataFrame(rmse_results)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rmse_df['surfactant'], rmse_df['RMSE'])
    ax.set_xlabel('Surfactant')
    ax.set_ylabel('RMSE (Measured vs. Ref CMC)')
    ax.set_title('RMSE Comparison of Measured and Reference CMC Values')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()

    return fig, ax

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# solve_rubingh_beta is not needed here but assume it's imported or defined elsewhere

def plot_cmc_vs_surf1_ratio(results_df, single_cmc_df, key, beta_df, ncols=5, Clint=False, filter_pairs=None):
    """
    Plots measured CMC vs. Surfactant 1 ratio. 
    Can filter the plots to only include specific surfactant pairs.
    """

    # Define colors locally to prevent NameError, using standard Matplotlib defaults/hex.

    s1 = 'surfactant_1'
    s2 = 'surfactant_2'
    rat = 'surfactant_1_ratio'
    cmc = f'CMC_{key}'
    surf = 'surfactant_1'
    meas = 'measured CMC 10min'

    # build lookup for pure compounds
    cmc_lookup = single_cmc_df.set_index(surf)[meas].to_dict()

    # --- Create beta lookup ---
    beta_lookup = {}
    for _, row in beta_df.iterrows():
        pair_key = (row['Surfactant 1'], row['Surfactant 2'])
        beta_lookup[pair_key] = {
            'avg': row['Average Beta'],
            'std': row['Beta Std Dev']
        }
    # --------------------------

    # find all unique combos
    combos = results_df[[s1, s2]].drop_duplicates().reset_index(drop=True)
    
    # --- NEW: Filter combos if filter_pairs is provided ---
    if filter_pairs is not None:
        # Create a list of 'S1/S2' strings for easy lookup
        combo_names = [f"{row[s1]}/{row[s2]}" for _, row in combos.iterrows()]
        
        # Determine which combos to keep
        filter_indices = [i for i, name in enumerate(combo_names) if name in filter_pairs]
        combos = combos.iloc[filter_indices].reset_index(drop=True)
    # -----------------------------------------------------

    nplots = len(combos)
    nrows = int(np.ceil(nplots / ncols)) if nplots > 0 else 1

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*4, nrows*3),
                             squeeze=False)
    
    # Keep track of the *actual* plot count, as the 'idx' from combos.iterrows() will be continuous
    # but we need to map it to the correct row/col index (plot_index)
    plot_index = 0

    #for idx, combo in combos.iterrows(): # Change was needed here: use plot_index instead of idx
    for _, combo in combos.iterrows(): 
        surf1 = combo[s1]
        surf2 = combo[s2]
        
        # Map the current sequential plot to its grid coordinates
        ax = axes[plot_index // ncols, plot_index % ncols]

        # mixture data
        sub = results_df[(results_df[s1]==surf1) & (results_df[s2]==surf2)]
        x_mix = sub[rat].values
        y_mix = sub[cmc].values

        # pure endpoints
        y0 = cmc_lookup.get(surf2, np.nan)  # at ratio 0 (Surfactant 2 pure)
        y1 = cmc_lookup.get(surf1, np.nan)  # at ratio 1 (Surfactant 1 pure)

        # combine and sort (measured)
        x_all = np.concatenate(([0], x_mix, [1]))
        y_all = np.concatenate(([y0], y_mix, [y1]))
        order = np.argsort(x_all)
        x_all, y_all = x_all[order], y_all[order]

        # measured
        ax.plot(
            x_all, y_all,
            linestyle='-',
            color=blue,              # line color
            marker='o',
            markersize=8,
            markerfacecolor=blue,    # fill color of the markers
            markeredgecolor='black', # border color of the markers
            markeredgewidth=0.5,     # border width of the markers
            label='Measured'
        )

        # optional Clint prediction
        if Clint and np.isfinite(y0) and np.isfinite(y1):
            x_grid = np.linspace(0, 1, 101)
            # Clint rule: 1/CMC_mix = x/CMC1 + (1-x)/CMC2
            y_clint = 1.0 / (x_grid / y1 + (1.0 - x_grid) / y0)
            ax.plot(
                x_grid, y_clint,
                linestyle='--',
                color=red,
                label='Clint'
            )
        
        beta_info = beta_lookup.get((surf1, surf2))
        
        if beta_info and np.isfinite(beta_info['avg']):
            avg = beta_info['avg']
            std = beta_info['std'] if np.isfinite(beta_info['std']) else 0.0
            
            # Create the LaTeX label (using the updated format)
            beta_label = f"$\\beta = {avg:.1f} \\pm {std:.1f}$"
            
            # Plot a dummy line with zero size/width to carry the legend label
            ax.plot([], [], ' ', label=beta_label)
        
        # Show the legend with the new beta entry
        ax.legend(fontsize=8, loc='best')
        # ------------------------------------

        # --- REVERTED TITLE ---
        ax.set_title(f"{surf1}/{surf2}", fontsize=10)
        # ----------------------
        
        ax.set_xlabel(f"{surf1} Ratio", fontsize=10)
        ax.set_ylabel('CMC (mM)')
        
        # Increment plot index
        plot_index += 1

    # turn off any empty subplots
    # Start checking from the last plotted index up to the total number of axes
    for j in range(plot_index, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout()
    
    if filter_pairs is None:
        plot_type = 'all_pairs'
    else:
        plot_type = 'filtered_pairs'

    # The figure saving logic is preserved from your original code
    if Clint:
        fig.savefig(f'figures/mixed_CMC_conc_clint_{key}_{plot_type}.png', dpi=300)
    else:
        fig.savefig(f'figures/mixed_CMC_conc_{key}_{plot_type}.png', dpi=300)
        
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq 

def solve_rubingh_beta(alpha, C1, C2, Cm, tol=1e-8, maxiter=200):
    """
    Given bulk fraction alpha, pure CMCs C1/C2, and mixed Cm,
    return (beta, x1_micelle). np.nan on failure.
    (The corrected Rubingh solver)
    """
    # ... (solve_rubingh_beta function remains unchanged, as it's correct) ...
    # sanity checks
    if not (np.isfinite(alpha) and np.isfinite(C1) and np.isfinite(C2) and np.isfinite(Cm)):
        return np.nan, np.nan
    if C1 <= 0 or C2 <= 0 or Cm <= 0 or alpha <= 0 or alpha >= 1:
        return np.nan, np.nan

    eps = 1e-8

    def phi(x):
        x = np.clip(x, eps, 1.0 - eps)
        F1 = (alpha * Cm) / (x * C1)
        F2 = ((1.0 - alpha) * Cm) / ((1.0 - x) * C2)
        if F1 <= 0 or F2 <= 0: return np.nan
        logF1 = np.log(F1)
        logF2 = np.log(F2)
        g1 = logF1 / (1.0 - x)**2
        g2 = logF2 / x**2
        return g1 - g2

    try:
        x_micelle = brentq(phi, eps, 1.0 - eps)
    except (ValueError, RuntimeError):
        xs = np.linspace(eps, 1.0 - eps, 400)
        vals = np.array([phi(xx) for xx in xs])
        if np.all(~np.isfinite(vals)): return np.nan, np.nan
        i = int(np.nanargmin(np.abs(vals)))
        x_micelle = xs[i]

    x = np.clip(x_micelle, eps, 1.0 - eps)
    F1 = (alpha * Cm) / (x * C1)
    F2 = ((1.0 - alpha) * Cm) / ((1.0 - x) * C2)
    
    if F1 <= 0 or F2 <= 0: return np.nan, np.nan
    
    g1 = np.log(F1) / (1.0 - x)**2
    g2 = np.log(F2) / x**2
    beta = 0.5 * (g1 + g2)
    
    return beta, x_micelle

# ----------------------------------------------------------------------
## Final Updated Plotting and Analysis Function
# ----------------------------------------------------------------------

def plot_rubingh_beta_vs_ratio(results_df, single_cmc_df, key, ncols=5, Clint=False,
                               exclude_extremes=True, extreme_threshold=0.05):

    # column helpers
    s1 = 'surfactant_1'
    s2 = 'surfactant_2'
    rat = 'surfactant_1_ratio'
    cmc_col = f'CMC_{key}'
    surf = 'surfactant_1'
    meas = 'measured CMC 10min'

    try:
        line_color = blue
    except NameError:
        line_color = '#1f77b4'

    # pure CMC lookup
    cmc_lookup = single_cmc_df.set_index(surf)[meas].to_dict()

    # unique ordered pairs
    combos = results_df[[s1, s2]].drop_duplicates().reset_index(drop=True)
    nplots = len(combos)
    nrows = int(np.ceil(nplots / ncols)) if nplots > 0 else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), squeeze=False)
    
    # List to store beta statistics for the final output DataFrame
    beta_stats = []

    # --- plotting loop ---
    for idx, combo in combos.iterrows():
        surf1 = combo[s1]
        surf2 = combo[s2]
        ax = axes[idx // ncols, idx % ncols]

        # subset for this ordered pair
        sub_all = results_df[(results_df[s1] == surf1) & (results_df[s2] == surf2)].copy()
        
        # Apply exclusion filter
        if exclude_extremes:
            sub = sub_all[
                (sub_all[rat] > extreme_threshold) & 
                (sub_all[rat] < (1.0 - extreme_threshold))
            ].copy()
        else:
            sub = sub_all.copy() 

        # Handle case where the filtered data is empty
        if sub.empty:
            ax.set_title(f"{surf1}/{surf2}", fontsize=10)
            ax.set_xlabel(f"{surf1} Ratio ($\\alpha_1$)", fontsize=10)
            ax.set_ylabel(r'$\beta$')
            ax.axhline(0.0, linewidth=0.8, color='k', alpha=0.4)
            ax.text(0.5, 0.5, f"No data: $\\alpha \in ({extreme_threshold}, {1-extreme_threshold})$", 
                    ha='center', va='center', transform=ax.transAxes, color='red', fontsize=9)
            
            # Record NaN stats for this pair
            beta_stats.append({
                'Surfactant 1': surf1,
                'Surfactant 2': surf2,
                'Average Beta': np.nan,
                'Beta Std Dev': np.nan,
                'N Points': 0
            })
            continue

        # get pure endpoints
        C1 = cmc_lookup.get(surf1, np.nan)
        C2 = cmc_lookup.get(surf2, np.nan)

        # compute beta for each mixture row in the filtered 'sub' dataframe
        alphas = sub[rat].to_numpy()
        Cms = sub[cmc_col].to_numpy()

        betas = np.full_like(alphas, np.nan, dtype=float)

        for i, (a, cm) in enumerate(zip(alphas, Cms)):
            b, xmic = solve_rubingh_beta(a, C1, C2, cm)
            betas[i] = b

        # --- CALCULATE AND RECORD STATS ---
        finite_betas = betas[np.isfinite(betas)]
        avg_beta = np.mean(finite_betas) if len(finite_betas) > 0 else np.nan
        std_beta = np.std(finite_betas) if len(finite_betas) > 1 else np.nan
        
        beta_stats.append({
            'Surfactant 1': surf1,
            'Surfactant 2': surf2,
            'Average Beta': avg_beta,
            'Beta Std Dev': std_beta,
            'N Points': len(finite_betas)
        })

        # --- PLOTTING (ONLY INCLUDED POINTS) ---
        # sort by alpha for a nicer line
        order = np.argsort(alphas)
        x_plot = alphas[order]
        y_plot = betas[order]

        ax.plot(
            x_plot, y_plot,
            linestyle='-',
            color=line_color,
            marker='o',
            markersize=8,
            markerfacecolor=line_color,
            markeredgecolor='black',
            markeredgewidth=0.5,
            label=r'$\beta$ (Rubingh)'
        )

        ax.set_title(f"{surf1}/{surf2}", fontsize=10)
        ax.set_xlabel(f"{surf1} Ratio ($\\alpha_1$)", fontsize=10)
        ax.set_ylabel(r'$\beta$')
        
        # Set y-limit based on calculated finite betas
        if len(finite_betas) > 0:
            y_min = np.nanmin(finite_betas)
            y_max = np.nanmax(finite_betas)
            # Use a slightly wider range than the data
            ax.set_ylim(y_min - 0.5, y_max + 0.5)
        
        # Ensure x-axis shows the full 0 to 1 range
        ax.set_xlim(0.0, 1.0)


        # zero line for reference
        ax.axhline(0.0, linewidth=0.8, color='k', alpha=0.4)
        
        # NOTE: Removed the axvspan calls for light red shading.


    # turn off unused axes
    for j in range(nplots, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout()
    fig.savefig(f'figures/beta_analysis_{key}.png', dpi=300)
    plt.show()
    
    # --- RETURN DATAFRAME ---
    return pd.DataFrame(beta_stats)





def plot_beta_heatmap(beta_values):

    from matplotlib.colors import LinearSegmentedColormap

    # Copy and mirror data so both (A,B) and (B,A) pairs exist
    df = beta_values.copy()
    mirrored = df.rename(columns={"Surfactant 1": "Surfactant 2", "Surfactant 2": "Surfactant 1"})
    df_full = pd.concat([df, mirrored], ignore_index=True).drop_duplicates(subset=["Surfactant 1", "Surfactant 2"])

    # Pivot table
    pivot = df_full.pivot(index="Surfactant 1", columns="Surfactant 2", values="Average Beta")

    # Reindex using fixed order
    pivot = pivot.reindex(index=SURFACTANT_ORDER[::-1], columns=SURFACTANT_ORDER)

    # Fill diagonal with 0.00
    for surf in SURFACTANT_ORDER:
        if surf in pivot.index and surf in pivot.columns:
            pivot.loc[surf, surf] = 0.00

    # Create mask of missing values
    mask = pivot.isna()

    # Define colormap
    cmap_continuous = LinearSegmentedColormap.from_list("custom_rwb", [blue, "white", red])

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot main heatmap
    sns.heatmap(
        pivot,
        cmap=cmap_continuous,
        vmin=-6,
        vmax=6,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor='grey',
        cbar_kws={'label': 'Average Beta'},
        ax=ax
    )

    # --- Overlay a light grey patch for N/A cells ---
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if mask.iloc[i, j]:
                # Draw light grey background rectangle
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1,
                    fill=True,
                    color='lightgrey',
                    ec='grey',
                    lw=0.5,
                    zorder=2
                ))
                # Add 'N/A' text
                ax.text(j + 0.5, i + 0.5, 'N/A',
                        ha='center', va='center',
                        color='black', fontsize=9, zorder=3)

    # Formatting
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()

    return fig, ax




def mixed_CMC_curve(data, data_type, font_size, show_plot=False, pair_filters=None):

    df = data[data_type].copy()

    # Column definitions
    I1_COL = '334_373'
    I3_COL = '334_384'
    RATIO_COL = 'ratio'
    SURF_1_COL = 'surfactant_1'
    SURF_1_RATIO_COL = 'surfactant_1_ratio'
    SURF_2_COL = 'surfactant_2'
    SURF_2_RATIO_COL = 'surfactant_2_ratio'
    ASSAY_COL = 'assay'
    CONC_COL = 'concentration'

    # Ensure ratio columns are numeric
    df[SURF_1_RATIO_COL] = pd.to_numeric(df[SURF_1_RATIO_COL], errors='coerce')
    df[SURF_2_RATIO_COL] = pd.to_numeric(df[SURF_2_RATIO_COL], errors='coerce')

    # Surfactant ordering
    SURFACTANT_ORDER = ['SDS', 'DSS', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS']
    df[SURF_1_COL] = pd.Categorical(df[SURF_1_COL], categories=SURFACTANT_ORDER, ordered=True)
    df[SURF_2_COL] = pd.Categorical(df[SURF_2_COL], categories=SURFACTANT_ORDER, ordered=True)

    # Identify unique surfactant combinations (including ratios)
    combos = (
        df[[SURF_1_COL, SURF_2_COL, SURF_1_RATIO_COL, SURF_2_RATIO_COL]]
        .drop_duplicates()
        .sort_values([SURF_1_COL, SURF_2_COL, SURF_1_RATIO_COL])
        .reset_index(drop=True)
    )

    results = []

    # --- NEW: Calculate ALL results first, regardless of plotting ---
    for idx, combo in combos.iterrows():
        surf1 = combo[SURF_1_COL]
        surf2 = combo[SURF_2_COL]
        r1 = float(combo[SURF_1_RATIO_COL])
        r2 = float(combo[SURF_2_RATIO_COL])

        sub = df[
            (df[SURF_1_COL] == surf1) &
            (df[SURF_2_COL] == surf2) &
            (df[SURF_1_RATIO_COL] == r1)
        ]
        conc = sub[CONC_COL].values
        ratio_vals = sub[RATIO_COL].values

        # Perform the fit to get CMC and R2
        cmc_values, r2_fit = CMC_plot(None, ratio_vals, conc, log=1, plot=0) # Pass None for ax and plot=0 since we only want the values

        results.append({
            SURF_1_COL:       surf1,
            SURF_1_RATIO_COL: r1,
            SURF_2_COL:       surf2,
            SURF_2_RATIO_COL: r2,
            f'CMC_{data_type}': cmc_values,
            f'R2_{data_type}':  r2_fit
        })
    # --- END NEW RESULT CALCULATION ---


    # --- PLOTTING LOGIC (mostly from previous answer) ---
    def plot_combos(part, name_suffix):
        if part.empty:
            return # Skip if no data

        nplots = len(part)
        ncols = 5
        nrows = int(np.ceil(nplots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

        for plot_idx, combo in part.iterrows(): # Use plot_idx for subplot position
            surf1 = combo[SURF_1_COL]
            surf2 = combo[SURF_2_COL]
            r1 = float(combo[SURF_1_RATIO_COL])
            r2 = float(combo[SURF_2_RATIO_COL])

            # ax position is based on the index within the PART, not the global index
            row_idx = plot_idx // ncols
            col_idx = plot_idx % ncols
            ax = axes[row_idx, col_idx]

            ax.tick_params(labelsize=font_size-2)


            sub = df[
                (df[SURF_1_COL] == surf1) &
                (df[SURF_2_COL] == surf2) &
                (df[SURF_1_RATIO_COL] == r1)
            ]
            conc = sub[CONC_COL].values
            ratio_vals = sub[RATIO_COL].values

            # This call is ONLY for plotting now
            CMC_plot(ax, ratio_vals, conc, log=1, plot=1)
            ax.set_title(f"{surf1}/{surf2} ({r1:.2f}/{r2:.2f})", fontsize=font_size)

            leg = ax.get_legend()
            if leg:
                # 1. Set font size for the legend entries (labels)
                for text in leg.get_texts():
                    text.set_fontsize(font_size) # Use the input font_size

                # 2. Set font size for the legend title
                # You were already doing this part correctly!
                leg.set_title(leg.get_title().get_text(), prop={'size': font_size + 2})

            # 1. Y-axis label for all figures on the left
            if col_idx == 0:
                ax.set_ylabel('I1/I3 ratio', fontsize=font_size)
            else:
                ax.set_ylabel('')

            # 2. X-axis label for all figures on the bottom
            if row_idx == nrows - 1:
                ax.set_xlabel('Concentration (log scale)', fontsize=font_size)
            else:
                ax.set_xlabel('')



        # Disable unused axes
        for j in range(nplots, nrows * ncols):
            axes[j // ncols, j % ncols].axis('off')

        plt.tight_layout()
        if show_plot:
            plt.show()

        fig.savefig(f'figures/mixed_CMC_curve_{data_type}_{name_suffix}.png', dpi=300)

    # --- Start of Plotting Selection Logic ---
    if pair_filters is not None:
        # Filter combinations based on the list of surfactant pairs
        filtered_combos = []
        for pair in pair_filters:
            surf1, surf2 = pair.split('/')

            # Match combinations where (surf1, surf2) or (surf2, surf1)
            match = combos[
                ((combos[SURF_1_COL] == surf1) & (combos[SURF_2_COL] == surf2)) |
                ((combos[SURF_1_COL] == surf2) & (combos[SURF_2_COL] == surf1))
            ]
            filtered_combos.append(match)

        # Plot all selected pairs in one figure
        part_to_plot = pd.concat(filtered_combos).drop_duplicates().reset_index(drop=True)
        if not part_to_plot.empty:
            plot_combos(part_to_plot, "filtered")
    else:
        # Original logic: Split ALL calculated combos into two parts for plotting
        part1 = combos.iloc[:45].reset_index(drop=True)
        part2 = combos.iloc[45:].reset_index(drop=True)

        # Plot both parts
        plot_combos(part1, "part_I")
        plot_combos(part2, "part_II")
    # --- End of Plotting Selection Logic ---

    results_df = pd.DataFrame(results)
    return results_df