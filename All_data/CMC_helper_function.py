import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def plot_cmc_vs_surf1_ratio(results_df, single_cmc_df, ncols=5, Clint=False):

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
        sub = results_df[(results_df[s1]==surf1) & (results_df[s2]==surf2)]
        x_mix = sub[rat].values
        y_mix = sub[cmc].values

        # pure endpoints
        y0 = cmc_lookup.get(surf2, np.nan)  # at ratio 0
        y1 = cmc_lookup.get(surf1, np.nan)  # at ratio 1

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
            ax.legend(fontsize=6, loc='best')

        ax.set_title(f"{surf1}/{surf2}", fontsize=10)
        ax.set_xlabel(f"{surf1} Ratio", fontsize=10)
        ax.set_ylabel('CMC (mM)')

    # turn off any empty subplots
    for j in range(nplots, nrows * ncols):
        axes[j // ncols, j % ncols].axis('off')

    if Clint == 0:
        plt.tight_layout()
        fig.savefig('figures/mixed_CMC_conc.png', dpi=300)
        plt.show()

    if Clint == 1:
        plt.tight_layout()
        fig.savefig('figures/mixed_CMC_conc_clint.png', dpi=300)
        plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # --------------------------
# # 数值工具
# # --------------------------

# _EPS = 1e-9

# def _safe_log(x):
#     return np.log(np.clip(x, 1e-300, None))

# def _bracket_root(f, a=1e-6, b=1-1e-6, n_try=80):
#     """
#     在(0,1)内尝试为 f(x)=0 找到变号区间；失败返回None
#     """
#     xs = np.linspace(a, b, n_try)
#     fs = [f(x) for x in xs]
#     for i in range(len(xs)-1):
#         if np.isfinite(fs[i]) and np.isfinite(fs[i+1]) and fs[i]*fs[i+1] < 0:
#             return xs[i], xs[i+1]
#     return None

# def _bisect_newton(f, df, a, b, maxit=80, tol=1e-10):
#     """
#     先二分逼近到较小区间，再混合牛顿迭代；要求 f(a)*f(b)<0
#     """
#     fa, fb = f(a), f(b)
#     if not (np.isfinite(fa) and np.isfinite(fb)):
#         raise RuntimeError("Function not finite at bracket ends.")
#     if fa*fb > 0:
#         raise RuntimeError("Root not bracketed.")

#     x = 0.5*(a+b)
#     for k in range(maxit):
#         # 尝试牛顿
#         fx = f(x)
#         dfx = df(x) if np.isfinite(fx) else np.nan
#         step_ok = False
#         if np.isfinite(fx) and np.isfinite(dfx) and abs(dfx) > 1e-14:
#             xn = x - fx/dfx
#             if (a < xn < b):
#                 x = xn
#                 step_ok = True

#         if not step_ok:
#             # 二分
#             if fa*fx <= 0:
#                 b, fb = x, fx
#             else:
#                 a, fa = x, fx
#             x = 0.5*(a+b)

#         if abs(fx) < tol or (b-a) < tol:
#             return np.clip(x, _EPS, 1-_EPS)
#     return np.clip(x, _EPS, 1-_EPS)

# # --------------------------
# # Rubingh 单点反演：已知 alpha1/CMCmix/CMC1/CMC2 -> 解 X1、beta_i
# # --------------------------

# def invert_X1_beta_from_point(alpha1, cmc_mix, cmc1, cmc2):
#     """
#     用“左右两条 beta 式相等”构造方程，先解 X1，再给出 beta_i
#     方程：
#     ln(α1*CMCmix/(X1*CMC1))/(1-X1)^2 = ln((1-α1)*CMCmix/((1-X1)*CMC2))/X1^2
#     """
#     a1 = np.clip(alpha1, _EPS, 1-_EPS)
#     cmc_mix = float(cmc_mix)
#     cmc1 = float(cmc1)
#     cmc2 = float(cmc2)

#     def left(X):
#         return _safe_log(a1*cmc_mix/(X*cmc1))/((1.0 - X)**2)

#     def right(X):
#         return _safe_log((1.0 - a1)*cmc_mix/((1.0 - X)*cmc2))/(X**2)

#     def f(X):
#         return left(X) - right(X)

#     def df(X):
#         # 对 left-right 的解析导数（稳定起见，分步写）
#         # left = ln(A/(X*CMC1)) / (1-X)^2 = [lnA - lnX - lnCMC1] * (1-X)^(-2)
#         L_num = _safe_log(a1*cmc_mix) - _safe_log(X) - _safe_log(cmc1)
#         L_den = (1.0 - X)**2
#         dL_num = -1.0 / X
#         dL_den = -2.0 * (1.0 - X)
#         dleft = (dL_num * L_den - L_num * dL_den) / (L_den**2)

#         # right = ln(B/((1-X)*CMC2)) / X^2 = [lnB - ln(1-X) - lnCMC2] * X^(-2)
#         R_num = _safe_log((1.0 - a1)*cmc_mix) - _safe_log(1.0 - X) - _safe_log(cmc2)
#         R_den = X**2
#         dR_num = 1.0 / (1.0 - X)
#         dR_den = 2.0 * X
#         dright = (dR_num * R_den - R_num * dR_den) / (R_den**2)

#         return dleft - dright

#     br = _bracket_root(f)
#     if br is None:
#         # 退化情况下，尝试从中心用牛顿
#         x0 = 0.5
#         # 简单回退：强制在(ε,1-ε)内
#         x = np.clip(x0, _EPS, 1 - _EPS)
#         # 粗糙牛顿几步
#         for _ in range(50):
#             fx = f(x)
#             dfx = df(x)
#             if not np.isfinite(fx) or not np.isfinite(dfx) or abs(dfx) < 1e-12:
#                 break
#             x_new = x - fx/dfx
#             if not (0 < x_new < 1):
#                 break
#             if abs(x_new - x) < 1e-10:
#                 x = x_new
#                 break
#             x = x_new
#         X1 = np.clip(x, _EPS, 1-_EPS)
#     else:
#         X1 = _bisect_newton(f, df, br[0], br[1])

#     # 计算 beta_i（两条式应近似一致）
#     beta_L = left(X1)
#     beta_R = right(X1)
#     beta_i = 0.5*(beta_L + beta_R)
#     return X1, beta_i

# # --------------------------
# # 给定 beta，解 X1（预测用）：ln(X1/(1-X1)) - ln r + beta*(1-2X1)=0
# # --------------------------

# def solve_X1_from_alpha_beta(alpha1, cmc1, cmc2, beta):
#     alpha1 = np.clip(alpha1, _EPS, 1-_EPS)
#     r = (alpha1/(1.0 - alpha1)) * (cmc2/cmc1)

#     def f(X):
#         return _safe_log(X) - _safe_log(1.0 - X) - _safe_log(r) + beta*(1.0 - 2.0*X)

#     def df(X):
#         return 1.0/X + 1.0/(1.0 - X) - 2.0*beta

#     br = _bracket_root(f)
#     if br is None:
#         # 容错：从中点用牛顿
#         x = 0.5
#         for _ in range(60):
#             fx = f(x)
#             dfx = df(x)
#             if not np.isfinite(fx) or not np.isfinite(dfx) or abs(dfx) < 1e-12:
#                 break
#             x_new = x - fx/dfx
#             if not (0 < x_new < 1):
#                 break
#             if abs(x_new - x) < 1e-12:
#                 x = x_new
#                 break
#             x = x_new
#         return np.clip(x, _EPS, 1.0-_EPS)
#     return _bisect_newton(f, df, br[0], br[1])

# def predict_cmc_from_beta(alpha1, cmc1, cmc2, beta):
#     X1 = solve_X1_from_alpha_beta(alpha1, cmc1, cmc2, beta)
#     # 两个等价表达式：
#     cmc_mix1 = (X1*cmc1/alpha1) * np.exp((1.0 - X1)**2 * beta)
#     cmc_mix2 = ((1.0 - X1)*cmc2/(1.0 - alpha1)) * np.exp((X1**2) * beta)
#     # 数值平均（抑制极小误差）
#     return 0.5*(cmc_mix1 + cmc_mix2), X1

# # --------------------------
# # 全局 beta 拟合：最小化 log-误差平方和
# # --------------------------

# def fit_global_beta(alphas, cmc_meas, cmc1, cmc2):
#     alphas = np.asarray(alphas, float)
#     cmc_meas = np.asarray(cmc_meas, float)

#     def objective(beta):
#         preds = []
#         for a, y in zip(alphas, cmc_meas):
#             yhat, _ = predict_cmc_from_beta(a, cmc1, cmc2, beta)
#             preds.append(yhat)
#         preds = np.asarray(preds)
#         res = _safe_log(preds) - _safe_log(cmc_meas)
#         return float(np.sum(res*res))

#     # 先粗网格扫一遍
#     grid = np.linspace(-20.0, 20.0, 801)  # 步长~0.05
#     vals = np.array([objective(b) for b in grid])
#     i0 = int(np.argmin(vals))
#     b_left = grid[max(0, i0-1)]
#     b_right = grid[min(len(grid)-1, i0+1)]

#     # 局部黄金分割搜索（单峰近似）
#     phi = 0.61803398875
#     a, c = b_left, b_right
#     # 如果左右相等，扩展一点区间
#     if a == c:
#         a, c = grid[max(0, i0-5)], grid[min(len(grid)-1, i0+5)]
#     b = c - phi*(c - a)
#     d = a + phi*(c - a)
#     fb = objective(b)
#     fd = objective(d)
#     for _ in range(80):
#         if fb < fd:
#             c, d, fd = d, b, fb
#             b = c - phi*(c - a)
#             fb = objective(b)
#         else:
#             a, b, fb = b, d, fd
#             d = a + phi*(c - a)
#             fd = objective(d)
#         if abs(c - a) < 1e-6:
#             break
#     beta_star = 0.5*(a + c)
#     return beta_star, objective(beta_star)

# # --------------------------
# # Rubingh plot：y = ln(α1*CMCmix/(X1*CMC1)) vs x = (1-X1)^2，斜率≈beta
# # --------------------------

# def rubingh_plot(ax, alphas, cmc_mix, cmc1, cmc2, title=None, save_path=None):
#     Xs, betas = [], []
#     for a, y in zip(alphas, cmc_mix):
#         X1, beta_i = invert_X1_beta_from_point(a, y, cmc1, cmc2)
#         Xs.append(X1)
#         betas.append(beta_i)
#     Xs = np.asarray(Xs)
#     betas = np.asarray(betas)

#     x = (1.0 - Xs)**2
#     y = _safe_log(alphas*cmc_mix/(Xs*cmc1))
#     # 过原点的最小二乘斜率：beta_plot
#     beta_plot = float(np.sum(x*y) / max(np.sum(x*x), 1e-30))

#     ax.scatter(x, y, label='data')
#     xx = np.linspace(0, max(1e-6, x.max()*1.05), 100)
#     ax.plot(xx, beta_plot*xx, label=f'fit (slope = {beta_plot:.3f})')
#     ax.set_xlabel(r'$(1 - X_1)^2$')
#     ax.set_ylabel(r'$\ln\left(\alpha_1\,CMC_{mix}/(X_1\,CMC_1)\right)$')
#     if title:
#         ax.set_title(title)
#     ax.legend()

#     if save_path:
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300)
#     return beta_plot, Xs, betas

# # --------------------------
# # 你的主函数：加入 RST 预测、Rubingh plot 与 beta 计算
# # --------------------------

# def plot_cmc_vs_surf1_ratio(results_df, single_cmc_df, ncols=5, Clint=False, do_RST=True, outdir='figures'):
#     """
#     results_df: 包含列 ['surfactant_1','surfactant_2','surfactant_1_ratio','CMC']
#     single_cmc_df: 包含列 ['surfactant_1','measured CMC']  (你的原命名)
#     """

#     os.makedirs(outdir, exist_ok=True)

#     s1 = 'surfactant_1'
#     s2 = 'surfactant_2'
#     rat = 'surfactant_1_ratio'
#     cmc = 'CMC'
#     surf = 'surfactant_1'
#     meas = 'measured CMC'

#     # 纯组分 CMC 查表
#     cmc_lookup = single_cmc_df.set_index(surf)[meas].to_dict()

#     # unique 组合
#     combos = results_df[[s1, s2]].drop_duplicates().reset_index(drop=True)
#     nplots = len(combos)
#     nrows = int(np.ceil(nplots / ncols))

#     fig, axes = plt.subplots(nrows, ncols,
#                              figsize=(ncols*4, nrows*3),
#                              squeeze=False)

#     for idx, combo in combos.iterrows():
#         surf1 = combo[s1]
#         surf2 = combo[s2]
#         ax = axes[idx // ncols, idx % ncols]

#         # 混合数据
#         sub = results_df[(results_df[s1]==surf1) & (results_df[s2]==surf2)].copy()
#         x_mix = sub[rat].values.astype(float)  # α1
#         y_mix = sub[cmc].values.astype(float)  # CMC_mix

#         # 纯端点
#         y0 = cmc_lookup.get(surf2, np.nan)  # α1=0
#         y1 = cmc_lookup.get(surf1, np.nan)  # α1=1

#         # 合并与排序（测量）
#         x_all = np.concatenate(([0], x_mix, [1]))
#         y_all = np.concatenate(([y0], y_mix, [y1]))
#         order = np.argsort(x_all)
#         x_all, y_all = x_all[order], y_all[order]

#         # 实测曲线
#         ax.plot(x_all, y_all, linestyle='-', marker='o', markersize=5, label='Measured')

#         # 可选 Clint 预测
#         if Clint and np.isfinite(y0) and np.isfinite(y1):
#             x_grid = np.linspace(0, 1, 201)
#             y_clint = 1.0 / (x_grid / y1 + (1.0 - x_grid) / y0)
#             ax.plot(x_grid, y_clint, linestyle='--', label='Clint')

#         title_bits = [f"{surf1}/{surf2}"]

#         if do_RST and np.isfinite(y0) and np.isfinite(y1) and len(x_mix) >= 2:
#             # 单点反演 beta_i
#             beta_is = []
#             X1_points = []
#             for a, y in zip(x_mix, y_mix):
#                 try:
#                     X1_i, beta_i = invert_X1_beta_from_point(a, y, y1, y0)  # 注意：cmc1=y1, cmc2=y0
#                     beta_is.append(beta_i)
#                     X1_points.append(X1_i)
#                 except Exception:
#                     pass

#             beta_is = np.array(beta_is, float)
#             if beta_is.size > 0:
#                 beta_mean = float(np.mean(beta_is))
#                 beta_std = float(np.std(beta_is, ddof=1)) if beta_is.size >= 2 else np.nan
#                 title_bits.append(f"β̄={beta_mean:.3f}±{(beta_std if np.isfinite(beta_std) else 0):.3f}")

#             # 全局 beta 拟合
#             try:
#                 beta_star, obj = fit_global_beta(x_mix, y_mix, y1, y0)
#                 title_bits.append(f"β*={beta_star:.3f}")
#                 # 用 β* 画 RST 预测曲线
#                 x_grid = np.linspace(0, 1, 301)
#                 y_rst = []
#                 for a in x_grid:
#                     yhat, _ = predict_cmc_from_beta(a, y1, y0, beta_star)
#                     y_rst.append(yhat)
#                 y_rst = np.asarray(y_rst)
#                 ax.plot(
#                     x_grid, y_rst,
#                     linestyle='--',
#                     label=f'Rubingh (β={beta_star:.3f})'
#                 )

#             except Exception:
#                 beta_star = None

#             # 生成 Rubingh plot 并保存
#             try:
#                 fig_r, ax_r = plt.subplots(figsize=(4,3))
#                 save_path = os.path.join(outdir, f"rubingh_plot_{surf1}__{surf2}.png")
#                 beta_plot, Xs, betas_single = rubingh_plot(
#                     ax_r, x_mix, y_mix, y1, y0,
#                     title=f"Rubingh plot: {surf1}/{surf2}",
#                     save_path=save_path
#                 )
#                 plt.close(fig_r)
#                 title_bits.append(f"β(plot)={beta_plot:.3f}")
#             except Exception:
#                 pass

#         ax.set_title(" | ".join(title_bits), fontsize=9)
#         ax.set_xlabel(f"{surf1} Ratio (α1)", fontsize=9)
#         ax.set_ylabel('CMC (mM)', fontsize=9)
#         ax.legend(fontsize=7, loc='best')

#     # 关掉空子图
#     for j in range(nplots, nrows * ncols):
#         axes[j // ncols, j % ncols].axis('off')

#     plt.tight_layout()
#     outfile = os.path.join(outdir, f"mixed_CMC_conc_{'clint_' if Clint else ''}rst.png")
#     plt.savefig(outfile, dpi=300)
#     plt.show()
