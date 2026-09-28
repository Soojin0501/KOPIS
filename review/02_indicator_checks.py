"""확인 4·5·6 — 성과 비교 단계의 수치를 저장된 결과 파일로 다시 계산한다.

  4. 보고서의 성공률이 결과 파일과 맞는가
  5. 보고서 안의 표들이 서로 맞는가
  6. 회귀 모델의 설명력은 무엇을 뜻하는가

실행 (저장소 루트에서)
  python review/02_indicator_checks.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "review" / "out"
OUT.mkdir(parents=True, exist_ok=True)
res = {}


def log(*a):
    print(*a, flush=True)


def rd(p):
    return pd.read_csv(ROOT / p, encoding="utf-8-sig")


# ───────── 확인 4. 성공률 ─────────
log("확인 4. 성공률")
REPORT = {"paid": 37.7, "free": 40.4}
res["check4"] = {}
for seg in ("paid", "free"):
    d = rd(f"results/comparison/composite_{seg}_pair_comparison.csv")
    mean_ab = (d["DPI_A"] + d["DPI_B"]) / 2
    rule = np.where(d["DPI_A+B"] >= mean_ab, "성공", "실패")
    n = len(d)
    ok = int((d["판정_mean"] == "성공").sum())
    fail = int((d["판정_mean"] == "실패").sum())
    out = {
        "rows": n,
        "unit": "공연장규모 × 장르조합 (공연 한 건이 아님)",
        "file_success": ok, "file_fail": fail,
        "success_rate": round(100 * ok / n, 1), "fail_rate": round(100 * fail / n, 1),
        "rule_matches_file": round(float((rule == d["판정_mean"]).mean()), 4),
        "report_value": REPORT[seg],
        "report_equals": "실패 비율" if abs(REPORT[seg] - 100 * fail / n) < 0.06 else
                         ("성공 비율" if abs(REPORT[seg] - 100 * ok / n) < 0.06 else "둘 다 아님"),
        "distinct_pairs": int(d["장르조합"].nunique()),
        "venue_sizes": d["공연장규모"].value_counts().to_dict(),
    }
    res["check4"][seg] = out
    log(f"  {seg}: {out}")

# ───────── 확인 5. 표 사이의 일치 ─────────
log("\n확인 5. 복합과 단일의 평균")
res["check5"] = {}
for folder in ("with_seats", "without_seats"):
    base = f"results/dpi/{folder}"
    files = {
        ("composite", "paid"): "DPI_ranked_Paid.csv", ("composite", "free"): "DPI_ranked_Free.csv",
        ("single", "paid"): "DPI_ranked_Paid_noncomplex.csv", ("single", "free"): "DPI_ranked_Free_noncomplex.csv",
    }
    data = {k: rd(f"{base}/{v}") for k, v in files.items()}
    block = {}
    for k, d in data.items():
        block[f"{k[0]}_{k[1]}"] = {
            "n": len(d),
            **{c: round(float(d[c].mean()), 4) for c in ("DPI_plus", "DPI_region_adj_seg", "U_participation", "S_scale", "A_afford")},
        }
    for kind in ("composite", "single"):
        d = pd.concat([data[(kind, "paid")], data[(kind, "free")]])
        block[f"{kind}_all"] = {
            "n": len(d),
            **{c: round(float(d[c].mean()), 4) for c in ("DPI_plus", "DPI_region_adj_seg", "U_participation", "S_scale", "A_afford")},
        }
    for seg in ("paid", "free"):
        a, b = data[("composite", seg)], data[("single", seg)]
        t = {}
        for c in ("DPI_plus", "U_participation", "S_scale", "A_afford"):
            tt = stats.ttest_ind(a[c].dropna(), b[c].dropna(), equal_var=False)
            t[c] = {"composite": round(float(a[c].mean()), 4), "single": round(float(b[c].mean()), 4),
                    "t": round(float(tt.statistic), 3), "p": float(f"{tt.pvalue:.3g}")}
        block[f"welch_{seg}"] = t
    res["check5"][folder] = block
    log(f"  [{folder}]")
    for k in ("composite_all", "single_all", "composite_paid", "single_paid", "composite_free", "single_free"):
        log(f"    {k:16s} {block[k]}")
    log(f"    유료 검정 {block['welch_paid']['DPI_plus']}")
    log(f"    무료 검정 {block['welch_free']['DPI_plus']}")

res["check5"]["report"] = {
    "table7_all": {"composite": 51.98, "single": 45.06, "U": [0.500714, 0.500013], "S": [0.500714, 0.500013], "A": [0.427556, 0.390496]},
    "table8_paid": {"composite": 40.6553, "single": 44.6132, "n": [477, 35980]},
    "table9_free": {"composite": 51.3927, "single": 41.3985, "n": [228, 4091]},
}
# 표 7 의 수치가 무엇에서 나왔는지 찾는다
cand = {}
for seg in ("paid", "free"):
    d = rd(f"results/comparison/composite_{seg}_pair_comparison.csv")
    cand[f"pair_{seg}"] = {"DPI_AB": round(float(d["DPI_A+B"].mean()), 4),
                           "mean_of_A_B": round(float(((d["DPI_A"] + d["DPI_B"]) / 2).mean()), 4),
                           "U_AB": round(float(d["U_AB"].mean()), 6), "S_AB": round(float(d["S_AB"].mean()), 6),
                           "A_AB": round(float(d["A_AB"].mean()), 6)}
both = pd.concat([rd("results/comparison/composite_paid_pair_comparison.csv"),
                  rd("results/comparison/composite_free_pair_comparison.csv")])
cand["pair_both"] = {"DPI_AB": round(float(both["DPI_A+B"].mean()), 4),
                     "mean_of_A_B": round(float(((both["DPI_A"] + both["DPI_B"]) / 2).mean()), 4),
                     "U_AB": round(float(both["U_AB"].mean()), 6), "S_AB": round(float(both["S_AB"].mean()), 6),
                     "A_AB": round(float(both["A_AB"].mean()), 6),
                     "U_single": round(float(pd.concat([both["U_A"], both["U_B"]]).mean()), 6),
                     "S_single": round(float(pd.concat([both["S_A"], both["S_B"]]).mean()), 6),
                     "A_single": round(float(pd.concat([both["A_A"], both["A_B"]]).mean()), 6)}
res["check5"]["table7_candidates"] = cand
log("  표 7 후보:", cand)

# ───────── 확인 6. 회귀 모델 ─────────
log("\n확인 6. 회귀 모델의 설명력")
res["check6"] = {}
F = ["U_participation", "S_scale", "A_afford"]
for seg, w in (("paid", (0.45, 0.35, 0.20)), ("free", (0.60, 0.40, 0.0))):
    d = rd(f"results/comparison/composite_{seg}_pair_comparison.csv").dropna(subset=F + ["DPI_plus"])
    X, y = d[F], d["DPI_plus"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    lin = LinearRegression().fit(Xtr, ytr)
    cv = cross_val_score(RandomForestRegressor(n_estimators=200, random_state=42), X, y,
                         cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    # 지표 공식 그대로: 가중 기하평균
    u, s, a = (d[c].clip(1e-9, 1) for c in F)
    wu, ws, wa = w
    formula = 100 * np.exp((wu * np.log(u) + ws * np.log(s) + wa * np.log(a)) / (wu + ws + wa))
    out = {
        "rows": len(d), "train_rows": len(Xtr), "test_rows": len(Xte),
        "rf_r2_on_test": round(float(r2_score(yte, pred)), 4),
        "rf_mse_on_test": round(float(mean_squared_error(yte, pred)), 2),
        "rf_r2_5fold": [round(float(x), 3) for x in cv],
        "rf_r2_5fold_mean": round(float(cv.mean()), 4),
        "linear_r2_on_test": round(float(r2_score(yte, lin.predict(Xte))), 4),
        "formula_r2_all": round(float(r2_score(y, formula)), 4),
        "formula_corr_all": round(float(np.corrcoef(y, formula)[0, 1]), 4),
        "feature_importance": dict(zip(F, [round(float(x), 3) for x in rf.feature_importances_])),
        "A_distinct_values": int(d["A_afford"].round(6).nunique()),
        "report": {"paid": {"r2": 0.918, "mse": 18.8}, "free": {"r2": 0.884, "mse": 19.9}}[seg],
        "notebook": {"paid": {"r2": 0.9210, "mse": 18.12}, "free": {"r2": 0.8785, "mse": 20.81}}[seg],
    }
    res["check6"][seg] = out
    log(f"  {seg}: {out}")

# 공연 단위에서도 공식만으로 지표가 재현되는지
for folder in ("with_seats",):
    for seg, fn, w in (("paid", "DPI_ranked_Paid.csv", (0.45, 0.35, 0.20)), ("free", "DPI_ranked_Free.csv", (0.60, 0.40, 0.0))):
        d = rd(f"results/dpi/{folder}/{fn}").dropna(subset=F + ["DPI_plus"])
        u, s, a = (d[c].clip(1e-9, 1) for c in F)
        wu, ws, wa = w
        formula = 100 * np.exp((wu * np.log(u) + ws * np.log(s) + wa * np.log(a)) / (wu + ws + wa))
        ratio = d["DPI_plus"] / formula
        res["check6"][f"performance_level_{seg}"] = {
            "rows": len(d), "formula_corr": round(float(np.corrcoef(d["DPI_plus"], formula)[0, 1]), 4),
            "ratio_min": round(float(ratio.min()), 3), "ratio_max": round(float(ratio.max()), 3),
        }
        log(f"  공연 단위 {seg}: {res['check6'][f'performance_level_{seg}']}")

# 성과 분석 표본에서 예측 장르가 빈 공연
d = rd("data/processed/complex2324_ffinal.csv")
res["empty_prediction_in_analysis_sample"] = {"rows": len(d), "empty": int(d["예측복합장르"].isna().sum())}
log("\n성과 분석 표본:", res["empty_prediction_in_analysis_sample"])

(OUT / "02_indicator_checks.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
log("저장:", OUT / "02_indicator_checks.json")
