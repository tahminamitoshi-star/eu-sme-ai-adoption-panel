"""
03_inference_checks.py
Additional inference and sensitivity checks requested in review (October 2026).
Not part of the original protocol; reported as post-review checks.

1. Estimation sample of every model (countries, years, observations).
2. 95% confidence intervals for every AI coefficient (cluster-robust).
3. Wild cluster bootstrap p-values (restricted, Webb six-point weights,
   9,999 replications), suited to a small number of clusters.
4. R3: model without the digital intensity control; R5: without GDP growth;
   R6: no control variables (country and year fixed effects only).
5. Leave-one-country-out: range of the AI coefficient when each country is dropped.

Input : ../04_data_clean/panel_clean.csv
Output: output/table7_inference.csv, table8_sensitivity.csv, inference_report.txt
Run   : python 03_inference_checks.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "05_analysis" / "output"
OUT.mkdir(parents=True, exist_ok=True)
SIZES = ["10-49", "50-249", "GE250"]
LAB = {"10-49": "Small", "50-249": "Medium", "GE250": "Large"}
OUTC = {"ln_lp": "ln real labour productivity", "ln_emp": "ln persons employed"}
B = 9999
rng = np.random.default_rng(20261010)
report = []


def log(t=""):
    report.append(str(t))
    print(t)


p = pd.read_csv(ROOT / "04_data_clean" / "panel_clean.csv")
p["ln_lp"] = np.log(p["lp_real_teur"])
p["ln_emp"] = np.log(p["emp"])


def design(d, y, controls):
    d = d.dropna(subset=[y, "ai_use"] + controls).copy()
    d = d[d.groupby("geo_code")["year"].transform("count") >= 2]
    X = pd.concat([d[["ai_use"] + controls],
                   pd.get_dummies(d["geo_code"], drop_first=False, dtype=float),
                   pd.get_dummies(d["year"], drop_first=True, prefix="y", dtype=float)], axis=1)
    return d, X.to_numpy(float), d[y].to_numpy(float), pd.factorize(d["geo_code"])[0]


def ols_cluster(X, y, g):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    G = g.max() + 1
    n, k = X.shape
    meat = np.zeros((k, k))
    for c in range(G):
        s = X[g == c].T @ e[g == c]
        meat += np.outer(s, s)
    V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V)), e


WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])


def wild_p(X, y, g):
    b, se, _ = ols_cluster(X, y, g)
    t0 = b[0] / se[0]
    Xr = X[:, 1:]                                   # impose beta_AI = 0
    br = np.linalg.pinv(Xr.T @ Xr) @ Xr.T @ y
    fit_r, e_r = Xr @ br, y - Xr @ br
    G = g.max() + 1
    tb = np.empty(B)
    for i in range(B):
        w = rng.choice(WEBB, size=G)[g]
        bb, sb, _ = ols_cluster(X, fit_r + e_r * w, g)
        tb[i] = bb[0] / sb[0]
    return float(np.mean(np.abs(tb) >= abs(t0)))


rows, sens = [], []
specs = {
    "Main": (lambda d: d, ["dii_basic", "gdp_growth"]),
    "R1: 2021 and 2023 only": (lambda d: d[d["year"].isin([2021, 2023])], ["dii_basic", "gdp_growth"]),
    "R2: excluding France and Sweden": (lambda d: d[~d["geo_code"].isin(["FR", "SE"])], ["dii_basic", "gdp_growth"]),
    "R3: without digital intensity": (lambda d: d, ["gdp_growth"]),
    "R5: without GDP growth": (lambda d: d, ["dii_basic"]),
    "R6: no control variables": (lambda d: d, []),
}
log("INFERENCE AND SENSITIVITY CHECKS (03_inference_checks.py)")
log("=" * 70)
for spec, (f, ctrl) in specs.items():
    log(f"\n--- {spec} ---")
    for y, ylab in OUTC.items():
        for s in SIZES:
            d, X, yy, g = design(f(p[p["size"] == s]), y, ctrl)
            b, se, _ = ols_cluster(X, yy, g)
            G = g.max() + 1
            from scipy import stats
            tcrit = stats.t.ppf(0.975, G - 1)
            p_cl = 2 * (1 - stats.t.cdf(abs(b[0] / se[0]), G - 1))
            p_w = wild_p(X, yy, g) if spec in ("Main", "R3: without digital intensity", "R5: without GDP growth", "R6: no control variables") else np.nan
            yrs = ", ".join(str(v) for v in sorted(d["year"].unique()))
            rows.append([spec, ylab, LAB[s], b[0], se[0], b[0] - tcrit * se[0], b[0] + tcrit * se[0],
                         p_cl, p_w, len(d), G, yrs])
            log(f"{ylab:<28} {LAB[s]:<7} b={b[0]:8.4f} SE={se[0]:.4f} 95% CI [{b[0]-tcrit*se[0]:.4f}, {b[0]+tcrit*se[0]:.4f}] "
                f"p(t, G-1)={p_cl:.3f} p(wild)={p_w:.3f} N={len(d)} countries={G} years={yrs}")

t7 = pd.DataFrame(rows, columns=["Specification", "Outcome", "Size class", "AI coefficient", "Clustered SE",
                                 "CI 95% low", "CI 95% high", "p-value (t, G-1 df)", "p-value (wild bootstrap)",
                                 "N", "Countries", "Years"])
t7.to_csv(OUT / "table7_inference.csv", index=False)

log("\n--- R4: leave one country out (main specification) ---")
for y, ylab in OUTC.items():
    for s in SIZES:
        sub = p[p["size"] == s]
        coefs, sig = [], 0
        for c in sorted(sub["geo_code"].unique()):
            d, X, yy, g = design(sub[sub["geo_code"] != c], y, ["dii_basic", "gdp_growth"])
            b, se, _ = ols_cluster(X, yy, g)
            coefs.append(b[0])
            sig += abs(b[0] / se[0]) > 2.06
        sens.append([ylab, LAB[s], min(coefs), max(coefs), sig, len(coefs)])
        log(f"{ylab:<28} {LAB[s]:<7} AI coefficient range [{min(coefs):.4f}, {max(coefs):.4f}]; "
            f"|t|>2.06 in {sig} of {len(coefs)} runs")
pd.DataFrame(sens, columns=["Outcome", "Size class", "Min AI coefficient", "Max AI coefficient",
                            "Runs with |t| > 2.06", "Runs"]).to_csv(OUT / "table8_sensitivity.csv", index=False)

log("\nNotes: p(t, G-1) uses the t distribution with G-1 degrees of freedom; "
    "wild bootstrap uses restricted residuals, Webb weights, 9,999 replications, seed 20261010.")
(OUT / "inference_report.txt").write_text("\n".join(report), encoding="utf-8")
