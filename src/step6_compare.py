"""보고서 결과물 ⑥ D-PI 를 통한 성과 분석 — 판정

  a. 복합과 단일의 평균 비교 (보고서 표 7)
  b. 공연장 규모 분류
  c. 장르 조합 비교: 복합 A+B 의 점수가 단일 A, B 평균 이상이면 성공
  d. 평균 차이 검정 (보고서 표 8, 표 9)

세 벌을 계산한다.

  saved_2025   2025년에 저장한 점수 파일을 그대로 읽는다. 보고서의 수치가 여기서 나왔다.
               유료 복합 공연만 다른 공식의 파일(DPI_ranked_Paid1)을 썼다.
  as_run       step5 의 as_run 점수. 유료 복합도 나머지와 같은 공식이다.
  as_reported  step5 의 as_reported 점수. 보고서가 설명한 공식이다.

실행: python src/step6_compare.py  (step5 를 먼저 실행)
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import OUT, SAVED, Check, log, norm_genre, pair_key, read_csv, venue_size, write_csv

METRICS = ["DPI_plus", "U_participation", "S_scale", "A_afford"]


def load(source):
    if source == "saved_2025":
        d = SAVED / "dpi" / "with_seats"
        comp = pd.concat([read_csv(d / "DPI_ranked_Paid1.csv"), read_csv(d / "DPI_ranked_Free.csv")])
        single = pd.concat([read_csv(d / "DPI_ranked_Paid_noncomplex.csv"), read_csv(d / "DPI_ranked_Free_noncomplex.csv")])
    else:
        comp = read_csv(OUT / "step5" / source / "dpi_composite.csv")
        single = read_csv(OUT / "step5" / source / "dpi_single.csv")
    for d in (comp, single):
        d["segment"] = np.where(d["is_free"].astype(str).str.lower() == "true", "Free", "Paid")
        d["size"] = venue_size(d["좌석수"])
    single["genre"] = single["장르명"].map(norm_genre)
    return comp.reset_index(drop=True), single.reset_index(drop=True)


def mean_table(comp, single):
    rows = []
    for m in METRICS:
        rows.append({"지표": m, "복합 평균": comp[m].mean(), "비복합 평균": single[m].mean(),
                     "차이": comp[m].mean() - single[m].mean()})
    return pd.DataFrame(rows)


def welch(comp, single, seg):
    a, b = comp[comp["segment"] == seg], single[single["segment"] == seg]
    rows = []
    for m in METRICS:
        x, y = a[m].dropna(), b[m].dropna()
        if x.nunique() <= 1 and y.nunique() <= 1:
            t, p = np.nan, np.nan        # 두 집단 모두 값이 하나뿐이면 검정할 수 없다
        else:
            r = stats.ttest_ind(x, y, equal_var=False)
            t, p = r.statistic, r.pvalue
        rows.append({"유무료": seg, "지표": m, "복합 n": len(x), "비복합 n": len(y),
                     "복합 평균": x.mean(), "비복합 평균": y.mean(), "t": t, "p": p})
    return pd.DataFrame(rows)


def pair_table(comp, single, seg):
    c = comp[(comp["segment"] == seg)].dropna(subset=["장르1", "장르2"]).copy()
    c["pair"] = [pair_key(a, b) for a, b in zip(c["장르1"], c["장르2"])]
    s = single[single["segment"] == seg]
    cg = c.groupby(["size", "pair"]).agg(n=("DPI_plus", "size"), DPI_AB=("DPI_plus", "mean"),
                                          U_AB=("U_participation", "mean"), S_AB=("S_scale", "mean"),
                                          A_AB=("A_afford", "mean")).reset_index()
    sg = s.groupby(["size", "genre"]).agg(DPI=("DPI_plus", "mean"), U=("U_participation", "mean"),
                                           S=("S_scale", "mean"), A=("A_afford", "mean"))
    cg["장르_A"] = cg["pair"].str.split("+").str[0]
    cg["장르_B"] = cg["pair"].str.split("+").str[1]
    for side in ("A", "B"):
        key = list(zip(cg["size"], cg[f"장르_{side}"]))
        for col in ("DPI", "U", "S", "A"):
            cg[f"{col}_{side}"] = [sg[col].get(k, np.nan) for k in key]
    cg["기준선"] = (cg["DPI_A"] + cg["DPI_B"]) / 2
    cg["판정"] = np.where(cg[["DPI_A", "DPI_B"]].isna().any(axis=1), "판정 불가",
                        np.where(cg["DPI_AB"] >= cg["기준선"], "성공", "실패"))
    return cg.rename(columns={"size": "공연장규모", "pair": "장르조합"})


def main():
    log("⑥ 성과 분석 — 판정")
    c = Check("step6")
    summary = []
    for source in ("saved_2025", "as_run", "as_reported"):
        log(f"\n  [{source}]")
        comp, single = load(source)
        out = OUT / "step6" / source

        t7 = mean_table(comp, single)
        write_csv(t7, out / "table7_mean_comparison.csv")
        dpi = t7.set_index("지표").loc["DPI_plus"]
        log(f"    표 7: 복합 {dpi['복합 평균']:.2f} · 단일 {dpi['비복합 평균']:.2f} · 차이 {dpi['차이']:+.2f}")

        tests = pd.concat([welch(comp, single, "Paid"), welch(comp, single, "Free")])
        write_csv(tests, out / "table8_9_welch.csv")

        rates = {}
        for seg, name in (("Paid", "paid"), ("Free", "free")):
            pt = pair_table(comp, single, seg)
            write_csv(pt, out / f"pair_comparison_{name}.csv")
            ok = pt[pt["판정"].isin(["성공", "실패"])]
            rate = 100 * (ok["판정"] == "성공").mean()
            rates[seg] = rate
            row = tests[(tests["유무료"] == seg) & (tests["지표"] == "DPI_plus")].iloc[0]
            log(f"    {seg}: 조합 {len(ok)}행 중 성공 {(ok['판정'] == '성공').sum()} → 성공률 {rate:.1f}% · "
                f"복합 {row['복합 평균']:.2f} (n={row['복합 n']}) vs 단일 {row['비복합 평균']:.2f} (n={row['비복합 n']}), p={row['p']:.3g}")
            summary.append({"벌": source, "유무료": seg, "조합 행": len(ok), "성공": int((ok["판정"] == "성공").sum()),
                            "성공률": round(rate, 1), "복합 n": int(row["복합 n"]), "단일 n": int(row["비복합 n"]),
                            "복합 평균": round(row["복합 평균"], 4), "단일 평균": round(row["비복합 평균"], 4),
                            "t": round(row["t"], 4), "p": row["p"]})

        if source == "saved_2025":
            c.add("표 7 복합 평균", round(dpi["복합 평균"], 2), 51.98, "보고서", tol=0.005)
            c.add("표 7 단일 평균", round(dpi["비복합 평균"], 2), 45.06, "보고서", tol=0.005)
            a = t7.set_index("지표").loc["A_afford"]
            c.add("표 7 가격 접근성 복합", round(a["복합 평균"], 6), 0.427556, "보고서", tol=1e-6)
            c.add("표 7 가격 접근성 단일", round(a["비복합 평균"], 6), 0.390496, "보고서", tol=1e-6)
            for seg, name, rep in (("Paid", "paid", 37.7), ("Free", "free", 40.4)):
                old = read_csv(SAVED / "comparison" / f"composite_{name}_pair_comparison.csv")
                new = read_csv(out / f"pair_comparison_{name}.csv")
                m = old.merge(new, on=["공연장규모", "장르조합"], how="outer", indicator=True)
                both = m[m["_merge"] == "both"]
                c.add(f"{seg} 조합 비교표 행 수", len(new[new["판정"] != "판정 불가"]), len(old), "2025 저장본")
                c.add(f"{seg} 조합 비교표 복합 점수 최대 차이", round(float((both["DPI_A+B"] - both["DPI_AB"]).abs().max()), 6), 0.0, "2025 저장본", tol=1e-6)
                c.add(f"{seg} 조합 비교표 판정이 같은 비율", round(float((both["판정_mean"] == both["판정"]).mean()), 4), 1.0, "2025 저장본")
                c.add(f"{seg} 성공률", round(rates[seg], 1), rep, "보고서", tol=0.05)
                c.add(f"{seg} 실패 비율", round(100 - rates[seg], 1), rep, "보고서", tol=0.05)
        if source == "as_reported":
            rep = {"Paid": (477, 35980, 40.6553, 44.6132), "Free": (228, 4091, 51.3927, 41.3985)}
            for seg, (n1, n2, m1, m2) in rep.items():
                row = tests[(tests["유무료"] == seg) & (tests["지표"] == "DPI_plus")].iloc[0]
                c.add(f"표 8·9 {seg} 복합 n", int(row["복합 n"]), n1, "보고서")
                c.add(f"표 8·9 {seg} 단일 n", int(row["비복합 n"]), n2, "보고서")
                c.add(f"표 8·9 {seg} 복합 평균", round(row["복합 평균"], 4), m1, "보고서", tol=0.005)
                c.add(f"표 8·9 {seg} 단일 평균", round(row["비복합 평균"], 4), m2, "보고서", tol=0.005)

    s = pd.DataFrame(summary)
    write_csv(s, OUT / "step6" / "summary.csv")
    log("\n" + s.to_string(index=False))
    c.save()


if __name__ == "__main__":
    main()
