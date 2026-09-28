"""발표 자료의 결론 — 지역별, 공연장 규모별 복합 장르 성과 해석

발표 자료가 든 수치를 다시 계산하고, 그 수치가 해석을 받쳐 주는지 본다.

발표 자료의 해석
  서울      규모와 매출이 주도. 공연 수와 매출 최대, 좌석점유율은 중상위
  경인      참여도가 주도. 좌석점유율 최고
  충청·영남  참여도는 양호, 규모와 매출은 약함
  호남      참여도와 규모가 주도, 매출은 최저
  제주      표본 부족
  대극장    판매 좌석과 예매 금액 최대 (2,396석, 3,440만 원). 좌석점유율 0.740
  중극장    좌석점유율 최고 (0.789)
  소극장    평균 공연 회차 최장 (5회). 좌석점유율 0.727

2025년 노트북은 300석 미만을 소극장, 1000석 미만을 중극장으로 나눴다. 보고서 본문의 기준(300석 이하)과 다르다.

실행: python src/step8_region_venue.py  (step5 를 먼저 실행)
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import OUT, Check, log, read_csv, write_csv

METRICS = {"공연 수": ("공연명", "count"), "좌석점유율": ("최종_좌석점유율", "mean"), "공연 회차": ("총_공연회차_계산", "mean"),
           "판매 좌석": ("판매좌석수_합계", "mean"), "예매 금액": ("예매금액_합계", "mean"),
           "예매 금액 중앙값": ("예매금액_합계", "median"), "무료 비율": ("is_free", "mean"),
           "U": ("U_participation", "mean"), "S": ("S_scale", "mean"), "A": ("A_afford", "mean"), "점수": ("DPI_plus", "mean")}


def venue_2025(seats):
    return np.where(seats < 300, "소극장", np.where(seats < 1000, "중극장", "대극장"))


def summarize(df, by):
    g = df.groupby(by).agg(**{k: v for k, v in METRICS.items()}).reset_index()
    return g.sort_values("공연 수", ascending=False)


def anova(df, by, col, min_n=10):
    groups = [g[col].dropna().to_numpy() for _, g in df.groupby(by) if len(g) >= min_n]
    if len(groups) < 2:
        return None
    f = stats.f_oneway(*groups)
    k = stats.kruskal(*groups)
    all_ = np.concatenate(groups)
    between = sum(len(x) * (x.mean() - all_.mean()) ** 2 for x in groups)
    total = ((all_ - all_.mean()) ** 2).sum()
    return {"집단 수": len(groups), "F": round(float(f.statistic), 3), "p": float(f"{f.pvalue:.3g}"),
            "순위 검정 p": float(f"{k.pvalue:.3g}"), "설명되는 비율": round(float(between / total), 4)}


def main():
    log("⑦ 지역과 공연장 규모 해석")
    df = read_csv(OUT / "step5" / "as_run" / "dpi_composite.csv")
    df["is_free"] = df["is_free"].astype(str).str.lower() == "true"
    df["지역"] = df["지역"].replace("기타", "강원")          # 2025년 노트북과 같게
    df["규모"] = venue_2025(pd.to_numeric(df["좌석수"], errors="coerce"))
    log(f"  복합 공연 {len(df)}건")

    region = summarize(df, "지역")
    venue = summarize(df, "규모")
    write_csv(region.round(4), OUT / "step8" / "region_summary.csv")
    write_csv(venue.round(4), OUT / "step8" / "venue_summary.csv")
    pd.set_option("display.width", 250)
    log("\n" + region.round(3).to_string(index=False))
    log("\n" + venue.round(3).to_string(index=False))

    c = Check("step8")
    v = venue.set_index("규모")
    c.add("대극장 평균 판매 좌석", round(v.loc["대극장", "판매 좌석"]), 2396, "발표 자료", tol=1)
    c.add("대극장 평균 예매 금액 (만 원)", round(v.loc["대극장", "예매 금액"] / 10000), 3440, "발표 자료", tol=1)
    c.add("대극장 좌석점유율", round(v.loc["대극장", "좌석점유율"], 3), 0.740, "발표 자료", tol=0.001)
    c.add("중극장 좌석점유율", round(v.loc["중극장", "좌석점유율"], 3), 0.789, "발표 자료", tol=0.001)
    c.add("소극장 좌석점유율", round(v.loc["소극장", "좌석점유율"], 3), 0.727, "발표 자료", tol=0.001)
    c.add("소극장 평균 공연 회차", round(v.loc["소극장", "공연 회차"]), 5, "발표 자료", tol=0.5)

    r = region.set_index("지역")
    top = lambda col: r[col].idxmax()
    low = lambda col: r[r["공연 수"] >= 10][col].idxmin()
    c.add("공연 수가 가장 많은 지역", top("공연 수"), "서울", "발표 자료")
    c.add("평균 예매 금액이 가장 큰 지역", top("예매 금액"), "서울", "발표 자료")
    c.add("좌석점유율이 가장 높은 지역", top("좌석점유율"), "경인", "발표 자료")
    c.add("평균 예매 금액이 가장 작은 지역 (10건 이상)", low("예매 금액"), "호남", "발표 자료")

    log("\n  집단 사이의 차이가 뚜렷한가")
    tests = []
    for by in ("지역", "규모"):
        for col, label in (("최종_좌석점유율", "좌석점유율"), ("예매금액_합계", "예매 금액"), ("총_공연회차_계산", "공연 회차"),
                           ("판매좌석수_합계", "판매 좌석"), ("DPI_plus", "점수")):
            a = anova(df, by, col)
            if a:
                tests.append({"나눈 기준": by, "지표": label, **a})
                log(f"    {by} · {label}: p={a['p']} (순위 검정 p={a['순위 검정 p']}) · 설명되는 비율 {a['설명되는 비율']}")
    write_csv(pd.DataFrame(tests), OUT / "step8" / "tests.csv")

    # 서울의 평균 매출을 끌어올린 것이 몇 건인가
    seoul = df[df["지역"] == "서울"].sort_values("예매금액_합계", ascending=False)
    share = seoul["예매금액_합계"].head(5).sum() / seoul["예매금액_합계"].sum()
    log(f"\n  서울 {len(seoul)}건의 예매 금액 중 상위 5건이 차지하는 비율 {share:.1%}")
    log(f"  서울 평균 {seoul['예매금액_합계'].mean():,.0f} · 중앙값 {seoul['예매금액_합계'].median():,.0f}")
    big = df[df["규모"] == "대극장"].sort_values("예매금액_합계", ascending=False)
    share_b = big["예매금액_합계"].head(5).sum() / big["예매금액_합계"].sum()
    log(f"  대극장 {len(big)}건의 예매 금액 중 상위 5건이 차지하는 비율 {share_b:.1%}")
    write_csv(pd.DataFrame([{"묶음": "서울", "공연": len(seoul), "상위 5건의 매출 비율": round(float(share), 3),
                             "평균": round(seoul["예매금액_합계"].mean()), "중앙값": round(seoul["예매금액_합계"].median())},
                            {"묶음": "대극장", "공연": len(big), "상위 5건의 매출 비율": round(float(share_b), 3),
                             "평균": round(big["예매금액_합계"].mean()), "중앙값": round(big["예매금액_합계"].median())}]),
              OUT / "step8" / "revenue_concentration.csv")

    # 지역과 규모는 서로 얽혀 있나
    cross = pd.crosstab(df["지역"], df["규모"])
    write_csv(cross.reset_index(), OUT / "step8" / "region_by_venue.csv")
    log("\n" + cross.to_string())
    chi = stats.chi2_contingency(cross.loc[cross.sum(axis=1) >= 10])
    log(f"  지역과 규모의 독립성 검정 p={chi.pvalue:.3g}")
    c.save()


if __name__ == "__main__":
    main()
