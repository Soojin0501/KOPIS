"""보고서 결과물 ① 복합·비복합 분류

공연 단위로 축약한 목록에서 분석 대상을 고른다.
  - 서커스/마술은 뺀다. 중분류가 이미 서커스/마술이라 단일에도 복합에도 넣기 어렵다.
  - 복합 중 세부장르가 다원/융복합인 것은 뺀다. 공연 밖 예술과의 혼합이라 장르의 합으로 볼 수 없다.

원천에서 공연 단위로 축약하는 단계는 2025년에 MySQL 로 했고 코드가 남아 있지 않다.
같은 축약은 schema/04_queries.py 의 v_performance_metrics 로 다시 만들 수 있다.

실행: python src/step1_split.py
"""
from common import EXCLUDED_GENRE, EXCLUDED_SUBGENRE, OUT, PROCESSED, Check, load_performances, log, norm_genre, read_csv, write_csv


def main():
    log("① 복합·비복합 분류")
    raw_single = read_csv(PROCESSED / "except_complex.csv")
    raw_comp = read_csv(PROCESSED / "complex.csv")
    single, comp = load_performances()

    log(f"  단일: {len(raw_single):,} → 서커스/마술 {int((raw_single['장르명'] == EXCLUDED_GENRE).sum()):,}건 제외 → {len(single):,}")
    log(f"  복합: {len(raw_comp):,} → 다원/융복합 {int((raw_comp['세부장르명'] == EXCLUDED_SUBGENRE).sum()):,}건 제외 → {len(comp):,}")

    c = Check("step1")
    c.add("단일 장르 공연 수", len(single), 72611, "보고서")
    c.add("복합 장르 공연 수", len(comp), 1099, "보고서")
    c.save()

    dist = single["장르명"].map(norm_genre).value_counts().rename_axis("장르").reset_index(name="공연수")
    dist["비율"] = (dist["공연수"] / dist["공연수"].sum()).round(4)
    write_csv(dist, OUT / "step1" / "single_genre_distribution.csv")
    write_csv(single, OUT / "step1" / "single.csv")
    write_csv(comp, OUT / "step1" / "composite.csv")


if __name__ == "__main__":
    main()
