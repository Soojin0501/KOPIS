"""보고서 결과물 ③ 동명이인 처리

같은 이름이 여러 장르에 나올 때 어느 수준에서 가를지 정한다.
장르를 세 수준으로 놓고 이름별 분포를 본다.

  대분류  갈래가 적어 동명이인이 갈리지 않는다
  중분류  채택. 전산망의 '장르명'
  세분류  너무 잘게 나뉘어 한 사람이 여럿으로 쪼개진다. 전산망의 '세부장르명'

대분류는 전산망 데이터에 없다. 보고서가 든 예(뮤지컬, 연극, 음악, 무용)에 맞춰 이 코드에서 묶었다.

실행: python src/step3_person_genre.py
"""
import pandas as pd

from common import (EXCLUDED_GENRE, OUT, PROCESSED, SAVED, Check, load_performances, log, norm_genre, read_csv,
                    split_names, split_names_2025_model, write_csv)

BROAD = {"서양음악": "음악", "한국음악": "음악", "대중음악": "음악", "무용": "무용", "대중무용": "무용",
         "연극": "연극", "뮤지컬": "뮤지컬"}


def explode(single, col, role):
    d = single[["장르명", "세부장르명", col]].copy()
    d["name"] = d[col].map(split_names)
    d = d.explode("name").dropna(subset=["name"])
    d["mid"] = d["장르명"].map(norm_genre)
    d["broad"] = d["mid"].map(BROAD)
    d["fine"] = d["세부장르명"]
    d["role"] = role
    return d[["name", "role", "broad", "mid", "fine"]]


def scores(d, level):
    g = d.groupby(["name", level]).size().rename("count").reset_index()
    g["score"] = g["count"] / g.groupby("name")["count"].transform("sum")
    return g.rename(columns={"name": "person", level: "genre"})


def main():
    log("③ 동명이인 처리")
    single, _ = load_performances()
    people = pd.concat([explode(single, "출연진내용", "cast"), explode(single, "제작진내용", "crew")])
    log(f"  참여 기록 {len(people):,}건 · 이름 {people['name'].nunique():,}개")

    rows = []
    for level, label in (("broad", "대분류"), ("mid", "중분류"), ("fine", "세분류")):
        per = people.groupby("name")[level].nunique()
        top_share = people.groupby(["name", level]).size().groupby("name").apply(lambda s: s.max() / s.sum())
        rows.append({
            "수준": label, "갈래 수": int(people[level].nunique()),
            "여러 갈래에 걸친 이름": int((per > 1).sum()),
            "그 비율": round(float((per > 1).mean()), 4),
            "이름 하나의 평균 갈래 수": round(float(per.mean()), 3),
            "주 갈래의 평균 비중": round(float(top_share.mean()), 4),
            "준식별자 수 (이름×갈래)": int(people.groupby(["name", level]).ngroups),
        })
    table = pd.DataFrame(rows)
    log(table.to_string(index=False))
    write_csv(table, OUT / "step3" / "level_comparison.csv")

    c = Check("step3")
    # 보고서의 예: 박소영 — 뮤지컬 38, 연극 30, 한국음악 1, 무용 1
    ex = people[people["name"] == "박소영"].groupby("mid").size().sort_values(ascending=False)
    log("  보고서의 예 (박소영):", ex.to_dict())
    for genre, n in (("뮤지컬", 38), ("연극", 30), ("한국음악", 1), ("무용", 1)):
        c.add(f"박소영 {genre} 참여 수", int(ex.get(genre, 0)), n, "보고서")

    # 2025년에 저장한 세분류 점수 파일과 비교. 저장본은 서커스/마술을 빼기 전 목록으로 만들었다
    full = read_csv(PROCESSED / "except_complex.csv")
    people_full = pd.concat([explode(full, "출연진내용", "cast"), explode(full, "제작진내용", "crew")])
    for role, fname in (("cast", "출연진_장르_점수_전체.csv"), ("crew", "제작진_장르_점수_전체.csv")):
        new = scores(people_full[people_full["role"] == role], "fine")
        write_csv(new, OUT / "step3" / f"person_genre_score_{role}.csv")
        old = read_csv(SAVED / "genre_scores" / fname)
        m = old.merge(new, on=["person", "genre"], how="outer", suffixes=("_old", "_new"), indicator=True)
        both = m[m["_merge"] == "both"]
        same = int((both["count_old"] == both["count_new"]).sum())
        c.add(f"{role} 점수 파일 행 수", len(new), len(old), "2025 저장본", tol=5)
        c.add(f"{role} 참여 수까지 같은 행의 비율", round(same / len(old), 4), 1.0, "2025 저장본", tol=0.001)
        log(f"  {role}: 양쪽에 있는 행 {len(both):,} 중 참여 수가 같은 행 {same:,} · "
            f"저장본에만 {int((m['_merge'] == 'left_only').sum()):,} · 새 계산에만 {int((m['_merge'] == 'right_only').sum()):,}")

    # 분류 모델의 이름 정리는 달랐다
    wrong = pd.concat([single["출연진내용"].map(split_names_2025_model), single["제작진내용"].map(split_names_2025_model)]).explode().dropna()
    tail = wrong[wrong.str.contains(r"(?:등|외|外)$")]
    good = set(people["name"])
    ghost = tail[~tail.isin(good)]
    log(f"  2025년 분류 모델의 이름 정리로는 이름 {wrong.nunique():,}개. "
        f"끝에 '등·외' 가 붙은 채 남은 이름 {ghost.nunique():,}개 (참여 기록 {len(ghost):,}건)")
    write_csv(pd.DataFrame([{"이름 정리": "보고서 기준", "이름 수": people["name"].nunique()},
                            {"이름 정리": "2025년 분류 모델", "이름 수": wrong.nunique()},
                            {"이름 정리": "그중 '등·외' 가 붙은 가짜 이름", "이름 수": ghost.nunique()}]),
              OUT / "step3" / "name_cleaning.csv")

    # 채택한 기준으로 만든 인물 목록
    persons = scores(people, "mid").rename(columns={"genre": "중분류"})
    write_csv(persons, OUT / "step3" / "person_by_mid_genre.csv")
    c.save()


if __name__ == "__main__":
    main()
