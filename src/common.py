"""모든 단계가 함께 쓰는 경로와 도우미."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"      # 2025년에 가공한 입력
SAVED = ROOT / "results"                     # 2025년에 저장한 결과
OUT = ROOT / "outputs"                       # 이 코드가 새로 만드는 결과

SEED = 42
EXCLUDED_GENRE = "서커스/마술"
EXCLUDED_SUBGENRE = "다원/융복합"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def log(*a):
    print(*a, flush=True)


def read_csv(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


def write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"  저장 {path.relative_to(ROOT).as_posix()} ({len(df):,}행)")


def norm_genre(name):
    """'서양음악(클래식)' → '서양음악'"""
    return re.sub(r"\s*\(.*?\)", "", str(name)).strip()


def split_names(text):
    """한 칸에 쉼표로 든 이름을 나눈다. 괄호 안의 역할과 끝의 '등·외' 를 뗀다."""
    if pd.isna(text):
        return []
    out = []
    for x in str(text).split(","):
        x = re.sub(r"\(.*?\)", "", x)
        x = re.sub(r"\s*(등|외|外)$", "", x.strip()).strip()
        if x:
            out.append(x)
    return out


def split_names_2025_model(text):
    """2025년 분류 모델이 쓴 이름 정리. 공백을 먼저 지워 끝의 '등·외' 가 떼어지지 않는다.

    '박지훈, 김준태 등' → ['박지훈', '김준태등']. 당시 결과를 재현할 때만 쓴다.
    """
    if pd.isna(text):
        return []
    names = str(text).replace(" ", "").split(",")
    names = [re.sub(r"\(.*?\)| 등| 외| 外", "", n).strip() for n in names]
    return [n for n in names if n]


def venue_size(seats):
    """보고서 기준: 소규모 300석 이하, 중규모 301~1000석, 대규모 1001석 이상."""
    s = pd.to_numeric(seats, errors="coerce")
    return pd.Series(np.where(s <= 300, "소규모", np.where(s <= 1000, "중규모", "대규모")), index=s.index)


def pair_key(a, b):
    """장르 조합은 순서를 따지지 않는다. '뮤지컬+연극' 과 '연극+뮤지컬' 은 같다."""
    return "+".join(sorted([str(a), str(b)]))


def load_performances():
    """① 의 제외 규칙을 적용한 단일·복합 공연 목록."""
    single = read_csv(PROCESSED / "except_complex.csv")
    comp = read_csv(PROCESSED / "complex.csv")
    single = single[single["장르명"] != EXCLUDED_GENRE].reset_index(drop=True)
    comp = comp[comp["장르명"] != EXCLUDED_GENRE].reset_index(drop=True)
    comp = comp[comp["세부장르명"] != EXCLUDED_SUBGENRE].reset_index(drop=True)
    return single, comp


class Check:
    """계산한 값과 보고서·저장본의 값을 나란히 적는다."""

    def __init__(self, step):
        self.step = step
        self.rows = []

    def add(self, item, computed, reference, source, tol=0.0):
        try:
            same = abs(float(computed) - float(reference)) <= tol
        except (TypeError, ValueError):
            same = str(computed) == str(reference)
        self.rows.append({"단계": self.step, "항목": item, "계산값": computed, "기준값": reference,
                          "기준": source, "일치": "일치" if same else "다름"})
        log(f"  [{'일치' if same else '다름'}] {item}: 계산 {computed} · {source} {reference}")
        return same

    def save(self):
        if not self.rows:
            return None
        df = pd.DataFrame(self.rows)
        write_csv(df, OUT / "checks" / f"{self.step}.csv")
        return df
