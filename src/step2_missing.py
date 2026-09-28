"""보고서 결과물 ② 복합 장르를 구별하기 위한 컬럼 선택

장르 구별에 쓸 열 8개의 결측을 연도별로 센다 (보고서 표 1, 표 2).
원작자명과 극작가명은 결측이 85% 를 넘어 분류 변수에서 뺐다.

실행: python src/step2_missing.py
"""
import pandas as pd

from common import EXCLUDED_GENRE, EXCLUDED_SUBGENRE, OUT, PROCESSED, Check, log, read_csv, write_csv

COLUMNS = ["공연명", "장르명", "세부장르명", "출연진내용", "제작진내용", "기획제작사명", "원작자명", "극작가명"]
YEARS = [2020, 2021, 2022, 2023, 2024]

# 보고서 표 1·2 의 값 (공연 수, 출연진 결측 수, 극작가 결측률)
REPORT = {
    "single": {2020: (5399, 1182, 96.70), 2021: (11872, 2555, 97.00), 2022: (16787, 5057, 97.24),
               2023: (18780, 5208, 99.05), 2024: (19774, 4455, 99.46)},
    "composite": {2020: (85, 31, 100.0), 2021: (133, 37, 100.0), 2022: (213, 65, 100.0),
                  2023: (325, 77, 100.0), 2024: (360, 79, 100.0)},
}


def missing_table(kind):
    rows, counts = [], {}
    for y in YEARS:
        name = f"{y}_complex.csv" if kind == "composite" else f"{y}_except_complex.csv"
        df = read_csv(PROCESSED / "by_year" / kind / name)
        df = df[df["장르명"] != EXCLUDED_GENRE]
        if kind == "composite":
            df = df[df["세부장르명"] != EXCLUDED_SUBGENRE]
        counts[y] = len(df)
        for col in COLUMNS:
            n = int(df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum())
            rows.append({"구분": kind, "연도": y, "공연수": len(df), "컬럼": col, "결측수": n,
                         "결측률": round(100 * n / len(df), 2) if len(df) else None})
    return pd.DataFrame(rows), counts


def main():
    log("② 열 선택과 결측")
    c = Check("step2")
    tables = []
    for kind in ("single", "composite"):
        t, counts = missing_table(kind)
        tables.append(t)
        for y in YEARS:
            rep_n, rep_cast, rep_writer = REPORT[kind][y]
            c.add(f"{kind} {y} 공연 수", counts[y], rep_n, "보고서 표")
            cast = int(t[(t["연도"] == y) & (t["컬럼"] == "출연진내용")]["결측수"].iloc[0])
            c.add(f"{kind} {y} 출연진 결측 수", cast, rep_cast, "보고서 표")
    out = pd.concat(tables)
    write_csv(out, OUT / "step2" / "missing_by_year.csv")

    pivot = out.pivot_table(index=["구분", "컬럼"], columns="연도", values="결측률").round(2)
    log(pivot.to_string())
    high = out.groupby("컬럼")["결측률"].min()
    dropped = [k for k, v in high.items() if v >= 85]
    log(f"  모든 연도에서 결측률 85% 이상인 열: {dropped}")
    c.add("분류에서 뺀 열", ", ".join(sorted(dropped)), "극작가명, 원작자명", "보고서")
    c.save()


if __name__ == "__main__":
    main()
