"""둘째 장르에 기준을 두고 다시 비교한다

2025년 코드는 확률이 아무리 낮아도 상위 두 장르를 조합으로 냈다. 씨앗을 바꾸면 조합이 절반 넘게 달라졌다.
여기서는 두 조건을 모두 채운 조합만 "확정" 으로 본다.

  안정  씨앗 셋으로 학습한 예측이 모두 같은 조합을 낸다
  확신  셋 모두에서 둘째 장르의 확률이 기준 이상이다

조건을 채우지 못한 공연은 조합을 정하지 않는다. 첫째 장르가 셋 모두 같으면 "주 장르만 확정" 으로 남긴다.

실행: python src/step9_confident_pairs.py  (step4 improved, step5, step6 을 먼저 실행)
"""
import numpy as np
import pandas as pd

from common import OUT, SAVED, Check, log, norm_genre, pair_key, read_csv, venue_size, write_csv
from step4_genre_gcn import SEEDS
from step6_compare import pair_table

THRESHOLDS = [0.0, 0.1, 0.3, 0.5]
CHOSEN = 0.3


def load_predictions():
    ps = [read_csv(OUT / "step4" / "improved" / f"predictions_seed{s}.csv") for s in SEEDS]
    d = pd.DataFrame({"공연명": ps[0]["공연명"]})
    for i, p in enumerate(ps):
        d[f"g1_{i}"], d[f"g2_{i}"] = p["장르1"], p["장르2"]
        d[f"p1_{i}"], d[f"p2_{i}"] = p["확률1"], p["확률2"]
        d[f"pair_{i}"] = [pair_key(a, b) for a, b in zip(p["장르1"], p["장르2"])]
    n = len(ps)
    d["주 장르 안정"] = (d[[f"g1_{i}" for i in range(n)]].nunique(axis=1) == 1)
    d["조합 안정"] = (d[[f"pair_{i}" for i in range(n)]].nunique(axis=1) == 1)
    d["둘째 확률 최솟값"] = d[[f"p2_{i}" for i in range(n)]].min(axis=1)
    d["둘째 확률 평균"] = d[[f"p2_{i}" for i in range(n)]].mean(axis=1)
    d["첫째 확률 평균"] = d[[f"p1_{i}" for i in range(n)]].mean(axis=1)
    return d


def classify(d, tau):
    ok = d["조합 안정"] & (d["둘째 확률 최솟값"] >= tau)
    return np.where(ok, "조합 확정", np.where(d["주 장르 안정"], "주 장르만 확정", "판단 불가"))


def success(comp, single):
    out = {}
    for seg, name in (("Paid", "유료"), ("Free", "무료")):
        part = comp[comp["segment"] == seg].dropna(subset=["장르1", "장르2"])
        if part.empty:
            out[name] = (0, 0, 0, pd.DataFrame(columns=["n", "판정"]))
            continue
        t = pair_table(comp, single, seg)
        t = t[t["판정"].isin(["성공", "실패"])]
        out[name] = (len(t), int((t["판정"] == "성공").sum()), int(t["n"].sum()), t)
    return out


def main():
    log("⑧ 둘째 장르에 기준 두기")
    d = load_predictions()
    log(f"  복합 공연 {len(d):,}건 · 주 장르가 씨앗 셋에서 같음 {int(d['주 장르 안정'].sum()):,} · 조합이 셋에서 같음 {int(d['조합 안정'].sum()):,}")

    rows = []
    for tau in THRESHOLDS:
        k = pd.Series(classify(d, tau)).value_counts()
        rows.append({"기준": tau, "조합 확정": int(k.get("조합 확정", 0)), "주 장르만 확정": int(k.get("주 장르만 확정", 0)),
                     "판단 불가": int(k.get("판단 불가", 0)), "조합 확정 비율": round(k.get("조합 확정", 0) / len(d), 4)})
    sweep = pd.DataFrame(rows)
    log("\n" + sweep.to_string(index=False))
    write_csv(sweep, OUT / "step9" / "threshold_sweep.csv")

    d["분류"] = classify(d, CHOSEN)
    d["확정 조합"] = np.where(d["분류"] == "조합 확정", d["pair_0"], "")
    d["주 장르"] = np.where(d["주 장르 안정"], d["g1_0"], "")
    keep = ["공연명", "분류", "주 장르", "확정 조합", "첫째 확률 평균", "둘째 확률 평균", "둘째 확률 최솟값"]
    write_csv(d[keep], OUT / "step9" / "composite_genre_final.csv")
    log(f"\n  기준 {CHOSEN} 에서 조합 확정 {int((d['분류'] == '조합 확정').sum())}건")
    log("  확정 조합 상위: " + str(d[d["분류"] == "조합 확정"]["확정 조합"].value_counts().head(8).to_dict()))

    # 2025년 예측과 비교
    old = read_csv(SAVED / "classification" / "complex_predictions_final.csv")
    old_pair = np.array([pair_key(a, b) for a, b in zip(old["장르1"], old["장르2"])])
    conf = (d["분류"] == "조합 확정").to_numpy()
    agree = float((old_pair[conf] == d["pair_0"].to_numpy()[conf]).mean())
    agree_rest = float((old_pair[~conf] == d["pair_0"].to_numpy()[~conf]).mean())
    log(f"  2025년 예측과 조합이 같은 비율: 확정한 공연 {agree:.1%} · 나머지 {agree_rest:.1%}")

    # 성과 비교를 다시 한다
    comp = read_csv(OUT / "step5" / "as_run" / "dpi_composite.csv")
    single = read_csv(OUT / "step5" / "as_run" / "dpi_single.csv")
    for t in (comp, single):
        t["segment"] = np.where(t["is_free"].astype(str).str.lower() == "true", "Free", "Paid")
        t["size"] = venue_size(t["좌석수"])
    single["genre"] = single["장르명"].map(norm_genre)

    uniq = d.drop_duplicates("공연명", keep=False)
    m = comp.merge(uniq[["공연명", "분류", "확정 조합", "주 장르"]], on="공연명", how="left")
    log(f"\n  성과 점수가 있는 복합 공연 {len(comp)}건 중 예측과 이름으로 이어진 것 {int(m['분류'].notna().sum())}건")

    base = comp.dropna(subset=["장르1", "장르2"])
    res, tables = [], {}
    groups = [("2025년 조합 전부", base)]
    for tau in THRESHOLDS:
        k = d.copy()
        k["분류"] = classify(k, tau)
        u = k.drop_duplicates("공연명", keep=False)
        mm = comp.drop(columns=["장르1", "장르2"]).merge(u[u["분류"] == "조합 확정"][["공연명", "pair_0"]], on="공연명")
        mm["장르1"] = mm["pair_0"].str.split("+").str[0]
        mm["장르2"] = mm["pair_0"].str.split("+").str[1]
        groups.append((f"안정하고 둘째 확률 {tau} 이상", mm))
        if tau == CHOSEN:
            new = mm
    for label, c in groups:
        sres = success(c, single)
        for seg in ("유료", "무료"):
            n_rows, ok, n_perf, t = sres[seg]
            tables[(label, seg)] = t
            res.append({"조합": label, "유무료": seg, "쓴 공연": n_perf, "비교 행": n_rows, "성공": ok,
                        "성공률": round(100 * ok / n_rows, 1) if n_rows else None,
                        "공연 5건 이상인 행": int((t["n"] >= 5).sum()) if len(t) else 0})
    out = pd.DataFrame(res)
    log("\n" + out.to_string(index=False))
    write_csv(out, OUT / "step9" / "success_rate.csv")

    # 복합과 단일의 평균도 다시
    means = []
    for seg, name in (("Paid", "유료"), ("Free", "무료")):
        s_mean = single[single["segment"] == seg]["DPI_plus"].mean()
        for label, c in (("복합 전체", comp), ("조합 확정", new), ("주 장르만 확정", m[m["분류"] == "주 장르만 확정"])):
            x = c[c["segment"] == seg]["DPI_plus"]
            means.append({"유무료": name, "묶음": label, "공연": len(x), "복합 평균": round(x.mean(), 2),
                          "단일 평균": round(s_mean, 2), "차이": round(x.mean() - s_mean, 2)})
    mt = pd.DataFrame(means)
    log("\n" + mt.to_string(index=False))
    write_csv(mt, OUT / "step9" / "mean_by_group.csv")

    c = Check("step9")
    c.add("기준을 두기 전 조합을 붙인 공연", len(d), 1099, "2025 저장본")
    c.save()


if __name__ == "__main__":
    main()
