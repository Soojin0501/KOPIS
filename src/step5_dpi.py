"""보고서 결과물 ⑤ 복합 장르 공연 성과 분석 (D-PI)

공연마다 참여도 U, 규모 S, 가격 접근성 A 를 0~1 로 만들고 하나의 점수로 합친다.
단일 공연과 복합 공연에 같은 함수를 쓴다.

두 가지 방식으로 계산한다.

  as_run       2025년 최종 노트북이 실제로 계산한 방식. 저장된 결과 파일과 같은 값이 나온다.
  as_reported  결과보고서가 설명한 방식.

둘의 차이
                          as_run                      as_reported
  유료 가중치 (U/S/A)      0.50 / 0.30 / 0.20          0.45 / 0.35 / 0.20
  가격 접근성 A            회차당매출과 단가를 섞음      단가의 역순 백분위
  좌석점유율 이상치         사분위 범위 밖 제거           제거하지 않음
  접근성 가산점            계산만 하고 점수에 넣지 않음   점수에 곱함
  회차당매출 하위 감점      계산만 하고 점수에 넣지 않음   없음 (보고서에 없음)

실행: python src/step5_dpi.py
"""
import numpy as np
import pandas as pd

from common import OUT, PROCESSED, SAVED, Check, log, read_csv, write_csv

NUM = ["예매금액_합계", "판매좌석수_합계", "총_공연회차_계산", "최종_좌석점유율",
       "평균_판매좌석수", "평균판매단가", "회차당매출", "접근성_점수"]
W_FREE = (0.60, 0.40)
W_PAID = {"as_run": (0.50, 0.30, 0.20), "as_reported": (0.45, 0.35, 0.20)}
BONUS_BETA, BONUS_CAP = 0.10, (0.90, 1.10)
K_GRID = [30, 60, 80, 100, 150]
CAP_GRID = [2.0, 3.0, 4.0, 5.0, 6.0]
W_ETA, W_RMS, W_DIST, W_RANK = 0.40, 0.30, 0.15, 0.15
RANK_FLOOR, RANK_PENALTY = 0.90, 0.20


def winsor(s, a=0.05, b=0.95):
    return s.clip(s.quantile(a), s.quantile(b))


def pct(s):
    return s.rank(pct=True, method="average")


def geo(parts, weights):
    g = np.ones(len(parts[0]))
    for p, w in zip(parts, weights):
        g = g * (np.clip(p, 1e-9, 1) ** w)
    return 100 * g ** (1.0 / sum(weights))


def usa(df, variant):
    occ = df["최종_좌석점유율"].astype(float)
    if variant == "as_run":
        occ = occ.clip(upper=1.0)
    U = pct(winsor(occ)).clip(0, 1)
    S = (0.5 * pct(winsor(np.log1p(df["총_공연회차_계산"].fillna(0).clip(lower=0))))
         + 0.3 * pct(winsor(np.log1p(df["판매좌석수_합계"].fillna(0).clip(lower=0))))
         + 0.2 * pct(winsor(df["평균_판매좌석수"].astype(float)))).clip(0, 1)
    price, rev = df["평균판매단가"], df["회차당매출"]
    if variant == "as_run":
        p, r = price.rank(pct=True, method="max"), rev.rank(pct=True, method="max")
        a_price = np.where(p < 0.5, 0.5, 1 - p)
        A = pd.Series((np.clip(r, 1e-9, 1) ** 0.6) * (np.clip(a_price, 1e-9, 1) ** 0.4), index=df.index)
    else:
        A = (1 - pct(winsor(price.astype(float)))).clip(0, 1)
    return U, S, A


def eta2(y, by):
    m = y.mean()
    g = y.groupby(by, dropna=False)
    between = (g.size() * (g.mean() - m) ** 2).sum()
    total = ((y - m) ** 2).sum()
    return float(between / total) if total else 0.0


def region_rms(y, by):
    m = y.mean()
    g = y.groupby(by, dropna=False)
    w = g.size() / g.size().sum()
    return float(np.sqrt(((g.mean() - m) ** 2 * w).sum()))


def region_adjust(y, by, k, cap):
    g = y.groupby(by, dropna=False)
    alpha = (g.transform("size") / (g.transform("size") + k)).clip(0, 1)
    delta = np.clip(alpha * (g.transform("mean") - y.mean()), -cap, cap)
    return y - delta


def best_region_params(y, by):
    rows = []
    for k in K_GRID:
        for cap in CAP_GRID:
            adj = region_adjust(y, by, k, cap)
            d = (adj - y).abs()
            rows.append({"K": k, "CAP": cap, "eta_post": eta2(adj, by), "rms_post": region_rms(adj, by),
                         "dist_p95": float(np.percentile(d, 95)), "spearman": float(y.corr(adj, method="spearman"))})
    r = pd.DataFrame(rows)
    mm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
    r["score"] = (W_ETA * mm(r["eta_post"]) + W_RMS * mm(r["rms_post"]) + W_DIST * mm(r["dist_p95"])
                  + W_RANK * mm(1 - r["spearman"]) + np.where(r["spearman"] < RANK_FLOOR, RANK_PENALTY, 0.0))
    r = r.sort_values(["score", "eta_post", "rms_post", "dist_p95"]).reset_index(drop=True)
    r["eta_pre"], r["rms_pre"] = eta2(y, by), region_rms(y, by)
    return r


def compute(df, variant):
    df = df.copy()
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["접근성_점수"] = df["접근성_점수"].clip(0, 11)
    if variant == "as_run":
        q1, q3 = df["최종_좌석점유율"].quantile(0.25), df["최종_좌석점유율"].quantile(0.75)
        iqr = q3 - q1
        df = df[(df["최종_좌석점유율"] >= q1 - 1.5 * iqr) & (df["최종_좌석점유율"] <= q3 + 1.5 * iqr)].copy()

    U, S, A = usa(df, variant)
    free = (df["평균판매단가"].fillna(0) <= 0) | (df["회차당매출"].fillna(0) <= 0)
    core_paid = geo([U, S, A], W_PAID[variant])
    core_free = geo([U, S], W_FREE) if variant == "as_reported" else 100 * (np.clip(U, 1e-9, 1) ** 0.6) * (np.clip(S, 1e-9, 1) ** 0.4)
    core = pd.Series(np.where(free, core_free, core_paid), index=df.index)

    inc = np.sqrt(pct(winsor((df["접근성_점수"].fillna(0) / 11.0).clip(0, 1))))
    mult = (1 + BONUS_BETA * (inc - inc.mean())).clip(*BONUS_CAP)

    df["U_participation"], df["S_scale"], df["A_afford"] = U, S, A
    df["is_free"] = free
    df["DPI_core"] = core
    df["access_multiplier"] = mult
    df["DPI_plus"] = core * mult if variant == "as_reported" else core   # as_run 은 가산점을 넣지 않았다
    df["segment"] = np.where(free, "Free", "Paid")

    params = {}
    df["DPI_region_adj_seg"] = np.nan
    for seg, part in df.groupby("segment"):
        grid = best_region_params(part["DPI_plus"], part["지역"])
        k, cap = float(grid.loc[0, "K"]), float(grid.loc[0, "CAP"])
        params[seg] = grid.loc[0].to_dict()
        df.loc[part.index, "DPI_region_adj_seg"] = region_adjust(part["DPI_plus"], part["지역"], k, cap)
    return df, params


def main():
    log("⑤ 성과지표 D-PI")
    c = Check("step5")
    inputs = {"composite": PROCESSED / "complex2324_ffinal.csv", "single": PROCESSED / "noncomplex2324_ffinal.csv"}
    saved = {("composite", "Paid"): "DPI_ranked_Paid.csv", ("composite", "Free"): "DPI_ranked_Free.csv",
             ("single", "Paid"): "DPI_ranked_Paid_noncomplex.csv", ("single", "Free"): "DPI_ranked_Free_noncomplex.csv"}
    summary = []
    for variant in ("as_run", "as_reported"):
        log(f"\n  [{variant}]")
        for kind, path in inputs.items():
            df, params = compute(read_csv(path), variant)
            write_csv(df, OUT / "step5" / variant / f"dpi_{kind}.csv")
            for seg in ("Paid", "Free"):
                part = df[df["segment"] == seg]
                p = params[seg]
                log(f"    {kind} {seg}: {len(part):,}건 · 평균 {part['DPI_plus'].mean():.4f} · "
                    f"지역 보정 K={p['K']:.0f} CAP={p['CAP']:.1f} · 지역 효과 {p['eta_pre']:.4f} → {p['eta_post']:.4f}")
                summary.append({"방식": variant, "구분": kind, "유무료": seg, "건수": len(part),
                                "DPI_plus": round(part["DPI_plus"].mean(), 4),
                                "DPI_region_adj": round(part["DPI_region_adj_seg"].mean(), 4),
                                "U": round(part["U_participation"].mean(), 6), "S": round(part["S_scale"].mean(), 6),
                                "A": round(part["A_afford"].mean(), 6), "K": p["K"], "CAP": p["CAP"]})
                if variant == "as_run":
                    old = read_csv(SAVED / "dpi" / "with_seats" / saved[(kind, seg)])
                    c.add(f"{kind} {seg} 건수", len(part), len(old), "2025 저장본")
                    c.add(f"{kind} {seg} 점수 평균", round(part["DPI_plus"].mean(), 4), round(old["DPI_plus"].mean(), 4), "2025 저장본", tol=1e-4)
                    c.add(f"{kind} {seg} 지역 보정 점수 평균", round(part["DPI_region_adj_seg"].mean(), 4),
                          round(old["DPI_region_adj_seg"].mean(), 4), "2025 저장본", tol=1e-3)
    write_csv(pd.DataFrame(summary), OUT / "step5" / "summary.csv")
    c.add("유료 가중치 (U/S/A)", "0.50/0.30/0.20", "0.45/0.35/0.20", "보고서")
    c.add("접근성 가산점이 점수에 들어갔나", "아니오", "예", "보고서")
    c.save()


if __name__ == "__main__":
    main()
