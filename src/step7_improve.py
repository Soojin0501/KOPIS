"""보고서 결과물 ⑥ D-PI 를 통한 성과 분석 — 원인과 개선

  1. 실패한 조합의 U, S, A 분포를 본다 (원인)
  2. U, S, A 로 점수를 예측하는 회귀 모델을 만든다
  3. 실패한 조합이 기준선을 넘는 데 필요한 최소 개선량을 찾는다 (개선)

1 과 2 는 2025년 노트북에 코드가 있다. 3 은 보고서에 결과만 있고 코드가 남아 있지 않아
보고서의 설명대로 다시 만들었다. 개선 폭의 상한과 간격은 보고서에 없어 이 코드에서 정했다.

실행: python src/step7_improve.py  (step6 을 먼저 실행)
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from common import OUT, SEED, Check, log, read_csv, write_csv

F = ["U_AB", "S_AB", "A_AB"]
STEP, MAX_DELTA = 0.01, 0.20        # 이 코드에서 정한 값
REPORT = {
    "Paid": {"r2": 0.918, "mse": 18.8, "conversion": 80.0, "dU": 0.105, "dS": 0.054, "dA": 0.040,
             "gap_before": 7.71, "margin_after": 0.27, "fail_U": 0.28, "fail_S": 0.45, "fail_A": 0.45},
    "Free": {"r2": 0.884, "mse": 19.9, "conversion": 63.2, "dU": 0.121, "dS": 0.075, "dA": 0.0,
             "gap_before": 12.14, "margin_after": -1.46, "fail_U": 0.31, "fail_S": 0.48, "fail_A": 0.386},
}


def minimal_improvement(model, row, use_a):
    """기준선을 넘는 개선 중 합이 가장 작은 것을 찾는다. 없으면 상한까지 올린 결과를 돌려준다."""
    grid = np.round(np.arange(0, MAX_DELTA + 1e-9, STEP), 4)
    base = row[F].to_numpy(dtype=float)
    combos = np.array(list(itertools.product(grid, grid, grid if use_a else [0.0])))
    X = np.clip(base + combos, 0, 1)
    pred = model.predict(pd.DataFrame(X, columns=F))
    ok = pred >= row["기준선"]
    if ok.any():
        cost = combos.sum(axis=1)
        cost[~ok] = np.inf
        i = int(np.argmin(cost))
        return combos[i], float(pred[i]), True
    i = int(np.argmax(pred))
    return combos[i], float(pred[i]), False


def run(source, seg, name, c):
    d = read_csv(OUT / "step6" / source / f"pair_comparison_{name}.csv")
    d = d[d["판정"].isin(["성공", "실패"])].reset_index(drop=True)
    fails = d[d["판정"] == "실패"].copy()
    rep = REPORT[seg]

    dist = fails[F].agg(["mean", "min", "max"]).round(3)
    log(f"    실패 {len(fails)}행 · U 평균 {dist.loc['mean', 'U_AB']} ({dist.loc['min', 'U_AB']}~{dist.loc['max', 'U_AB']}) · "
        f"S 평균 {dist.loc['mean', 'S_AB']} · A 평균 {dist.loc['mean', 'A_AB']}")

    X, y = d[F], d["DPI_AB"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=200, random_state=SEED).fit(Xtr, ytr)
    lin = LinearRegression().fit(Xtr, ytr)
    r2, mse = r2_score(yte, rf.predict(Xte)), mean_squared_error(yte, rf.predict(Xte))
    imp = dict(zip(F, rf.feature_importances_.round(3)))
    log(f"    회귀: 학습 {len(Xtr)}행 · 평가 {len(Xte)}행 · 설명력 {r2:.3f} · 오차 {mse:.1f} · 선형회귀 설명력 {r2_score(yte, lin.predict(Xte)):.3f}")
    log(f"    중요도 {imp}")

    use_a = fails["A_AB"].nunique() > 1
    rows = []
    for _, r in fails.iterrows():
        delta, pred, ok = minimal_improvement(rf, r, use_a)
        rows.append({"공연장규모": r["공연장규모"], "장르조합": r["장르조합"], "현재 점수": r["DPI_AB"], "기준선": r["기준선"],
                     "격차": r["기준선"] - r["DPI_AB"], "dU": delta[0], "dS": delta[1], "dA": delta[2],
                     "개선 후 예측": pred, "여유": pred - r["기준선"], "전환": ok})
    res = pd.DataFrame(rows)
    write_csv(res, OUT / "step7" / source / f"minimal_improvement_{name}.csv")
    conv = 100 * res["전환"].mean() if len(res) else float("nan")
    log(f"    최소 개선: 전환 {int(res['전환'].sum())}/{len(res)} ({conv:.1f}%) · 평균 dU {res['dU'].mean():.3f} dS {res['dS'].mean():.3f} "
        f"dA {res['dA'].mean():.3f} · 격차 {res['격차'].mean():.2f} · 여유 {res['여유'].mean():+.2f}")

    if source == "saved_2025":
        c.add(f"{seg} 실패 행 수", len(fails), {"Paid": 20, "Free": 19}[seg], "2025 저장본")
        c.add(f"{seg} 실패 표본 U 평균", round(fails["U_AB"].mean(), 2), rep["fail_U"], "보고서", tol=0.01)
        c.add(f"{seg} 실패 표본 S 평균", round(fails["S_AB"].mean(), 2), rep["fail_S"], "보고서", tol=0.01)
        c.add(f"{seg} 회귀 설명력", round(r2, 3), rep["r2"], "보고서", tol=0.005)
        c.add(f"{seg} 회귀 오차", round(mse, 1), rep["mse"], "보고서", tol=0.1)
        c.add(f"{seg} 평균 격차", round(res["격차"].mean(), 2), rep["gap_before"], "보고서", tol=0.01)
        c.add(f"{seg} 전환 비율", round(conv, 1), rep["conversion"], "보고서", tol=0.1)
        c.add(f"{seg} 평균 dU", round(res["dU"].mean(), 3), rep["dU"], "보고서", tol=0.005)
        c.add(f"{seg} 평균 dS", round(res["dS"].mean(), 3), rep["dS"], "보고서", tol=0.005)
    return {"벌": source, "유무료": seg, "전체 행": len(d), "평가 행": len(Xte), "실패": len(fails),
            "설명력": round(r2, 3), "선형회귀 설명력": round(r2_score(yte, lin.predict(Xte)), 3),
            "전환 비율": round(conv, 1), "dU": round(res["dU"].mean(), 3), "dS": round(res["dS"].mean(), 3),
            "dA": round(res["dA"].mean(), 3), "격차": round(res["격차"].mean(), 2), "여유": round(res["여유"].mean(), 2)}


def main():
    log("⑥ 성과 분석 — 원인과 개선")
    c = Check("step7")
    out = []
    for source in ("saved_2025", "as_run", "as_reported"):
        for seg, name in (("Paid", "paid"), ("Free", "free")):
            log(f"\n  [{source} · {seg}]")
            out.append(run(source, seg, name, c))
    s = pd.DataFrame(out)
    write_csv(s, OUT / "step7" / "summary.csv")
    log("\n" + s.to_string(index=False))
    c.save()


if __name__ == "__main__":
    main()
