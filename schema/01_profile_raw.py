"""원천 데이터의 열 구조를 파악한다.

열마다 고유값 수와 결측 비율을 재고, 어떤 열이 어떤 키에 종속되는지 확인한다.
이 결과가 테이블을 나누는 근거가 된다.

실행: python schema/01_profile_raw.py
"""
import sys
from pathlib import Path

import duckdb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "temp.csv"
DB = ROOT / "data" / "kopis.duckdb"

con = duckdb.connect(str(DB))
con.execute("SET preserve_insertion_order=false")
exists = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='raw'").fetchone()[0]
if not exists:
    print("원천 데이터를 적재한다:", RAW)
    con.execute(
        f"CREATE TABLE raw AS SELECT * FROM read_csv('{RAW.as_posix()}', all_varchar=true, header=true, sample_size=-1)"
    )

n = con.execute("SELECT count(*) FROM raw").fetchone()[0]
cols = [r[0] for r in con.execute("DESCRIBE raw").fetchall()]
print(f"행 {n:,} · 열 {len(cols)}")

# 한 번의 훑기로 모든 열의 고유값 수와 결측 비율을 잰다
parts = []
for c in cols:
    parts.append(f'count(DISTINCT "{c}")')
    parts.append(f"100.0*sum(CASE WHEN \"{c}\" IS NULL OR \"{c}\"='' THEN 1 ELSE 0 END)/count(*)")
    parts.append(f'any_value("{c}")')
row = con.execute("SELECT " + ", ".join(parts) + " FROM raw").fetchone()
print()
print(f"{'번호':>3}  {'고유값':>9} {'결측%':>7}  열 · 예시")
stats = {}
for i, c in enumerate(cols):
    d, miss, ex = row[i * 3], row[i * 3 + 1], row[i * 3 + 2]
    stats[c] = d
    print(f"{i+1:>3}  {d:>9,} {miss:>7.2f}  {c} · {str(ex)[:40]}")


def depends_on(key):
    """key 값 하나에 값이 둘 이상 붙는 키의 수를 열마다 센다. 0 이면 그 열은 key 에 종속된다."""
    others = [c for c in cols if c != key]
    inner = ", ".join(f'count(DISTINCT coalesce("{c}", \'\')) AS c{i}' for i, c in enumerate(others))
    outer = ", ".join(f"sum(CASE WHEN c{i} > 1 THEN 1 ELSE 0 END)" for i in range(len(others)))
    res = con.execute(f'SELECT {outer} FROM (SELECT {inner} FROM raw GROUP BY "{key}")').fetchone()
    return dict(zip(others, res))


print()
for key in ("전송사업자코드", "공연시설코드", "공연장코드", "공연코드"):
    if key not in cols:
        print(f"== {key}: 열 없음")
        continue
    dep = depends_on(key)
    full = [c for c, v in dep.items() if v == 0]
    near = sorted([(c, int(v)) for c, v in dep.items() if 0 < v <= stats[key] * 0.05], key=lambda x: x[1])
    print(f"== 키 {key} (고유 {stats[key]:,})")
    print("   완전 종속:", full)
    print("   거의 종속 (위반 5% 이하):", near)
