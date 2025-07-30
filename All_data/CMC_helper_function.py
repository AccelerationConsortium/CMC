import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from scipy.optimize import curve_fit
import math


surfactant_library = {
    "SDS": {
        "full_name": "Sodium Dodecyl Sulfate",
        "CAS": "151-21-3",
        "CMC": 8.5,
        "Category": "anionic",
        "MW": 289.39,
        "stock_conc": 50,  # mM
        "low": 7,
        "high": 10
    },


    "NaDC": {
        "full_name": "Sodium Docusate",
        "CAS": "577-11-7",
        "CMC": 5.3375,
        "Category": "anionic",
        "MW": 445.57,
        "stock_conc": 25,  # mM
        "low":2.48,
        "high": 8.2
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
    },


    "DTAB": {
        "full_name": "Dodecyltrimethylammonium Bromide",
        "CAS": "1119-94-4",
        "CMC": 15.85,
        "Category": "cationic",
        "MW": 308.34,
        "stock_conc": 50,  # mM
        "low": 15.7,
        "high": 16
    },


    "TTAB": {
        "full_name": "Tetradecyltrimethylammonium Bromide",
        "CAS": "1119-97-7",
        "CMC": 3.985,
        "Category": "cationic",
        "MW": 336.39,
        "stock_conc": 50,  # mM
        "low": 3.77,
        "high": 4.2
    },

    "CAPB": {
        "full_name": "Cocamidopropyl Betaine",
        "CAS": "61789-40-0",
        "CMC": 0.627,
        "Category": "zwitterionic",
        "MW": 342.52,
        "stock_conc": 50,  # mM
        "low": 0.28,
        "high": 0.974
    },
    
    "CHAPS": {
        "full_name": "CHAPS",
        "CAS": "75621-03-3",
        "CMC": 8,
        "Category": "zwitterionic",
        "MW": 614.88,
        "stock_conc": 30,  # mM
        "low": 6,
        "high": 10
    }
}

SURFACTANT_ORDER = ['SDS', 'NaDC', 'NaC', 'CTAB', 'DTAB', 'TTAB', 'CAPB', 'CHAPS']

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
            fontsize=8,
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
        std_all  = grp['CMC'].std()

        # overall CV
        cv_overall = std_all/mean_all if len(grp)>1 and mean_all!=0 else np.nan

        out.append({
            surf_col:         surf,
            'replicate CV': cv_rep,
            'assay CV':     cv_assay,
            'time CV':      cv_time,
            'overall CV':      cv_overall,
            'measured CMC':        mean_all,
            'measured CMC STD':         std_all
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
    measured_color = red,
    literature_color= blue,
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
        mean, std = row['measured CMC'], row['measured CMC STD']
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


def plot_cmc_vs_surf1_ratio(results_df, single_cmc_df, ncols=5):


    s1 = 'surfactant_1'
    s2 = 'surfactant_2'
    rat = 'surfactant_1_ratio'
    cmc = 'CMC'
    surf = 'surfactant_1'
    meas = 'measured CMC'

    # build lookup for pure compounds
    cmc_lookup = single_cmc_df.set_index(surf)[meas].to_dict()

    # find all unique combos
    combos = results_df[[s1, s2]].drop_duplicates().reset_index(drop=True)
    nplots = len(combos)
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*4, nrows*3),
                             squeeze=False)

    for idx, combo in combos.iterrows():
        surf1 = combo[s1]
        surf2 = combo[s2]
        ax = axes[idx // ncols, idx % ncols]

        # mixture data
        sub = results_df[(results_df[s1]==surf1)&(results_df[s2]==surf2)]
        x_mix = sub[rat].values
        y_mix = sub[cmc].values

        # pure endpoints
        y0 = cmc_lookup.get(surf2, np.nan)  # at ratio 0
        y1 = cmc_lookup.get(surf1, np.nan)  # at ratio 1

        # combine and sort
        x_all = np.concatenate(([0], x_mix, [1]))
        y_all = np.concatenate(([y0], y_mix, [y1]))
        order = np.argsort(x_all)
        x_all, y_all = x_all[order], y_all[order]

        # single line+marker plot
        ax.plot(
            x_all, y_all,
            linestyle='-',
            color=blue,              # line color
            marker='o',
            markersize=8,
            markerfacecolor=blue,    # fill color of the markers
            markeredgecolor='black',   # border color of the markers
            markeredgewidth=0.5,       # border width of the markers
            label=f"{surf1}/{surf2}"
        )

        ax.set_title(f"{surf1}/{surf2}", fontsize=10)
        ax.set_xlabel(f"{surf1} Ratio", fontsize=10)
        ax.set_ylabel('CMC (mM)')
#        ax.grid(True, linestyle='--', alpha=0.5)
#        ax.legend(fontsize=6, loc='best')

    # turn off any empty subplots
    for j in range(nplots, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    plt.tight_layout()
    fig.savefig('figures/mixed_CMC_conc.png', dpi=300)
    plt.show()