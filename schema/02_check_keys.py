"""테이블을 나누기 전에 키 후보가 실제로 유일한지 확인한다.

실행: python schema/02_check_keys.py
"""
import sys
from pathlib import Path

import duckdb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data" / "kopis.duckdb"), read_only=True)


def q(sql):
    return con.execute(sql).fetchall()


def uniq(label, cols):
    keys = ", ".join(f'"{c}"' for c in cols)
    total, distinct = q(f"SELECT count(*), count(DISTINCT ({keys})) FROM raw")[0]
    print(f"  {label:42s} 행 {total:,} · 고유 {distinct:,} · {'유일' if total == distinct else '중복 있음'}")


print("1) 거래 행의 키 후보")
uniq("입장권고유번호", ["입장권고유번호"])
uniq("입장권고유번호 + 구분", ["입장권고유번호", "예매/취소구분"])
uniq("입장권고유번호 + 구분 + 일시", ["입장권고유번호", "예매/취소구분", "예매/취소일시"])
print("   구분별 행 수:", q('SELECT "예매/취소구분", count(*) FROM raw GROUP BY 1 ORDER BY 1'))
print("   한 입장권에 붙은 행 수 분포:", q(
    'SELECT n, count(*) FROM (SELECT count(*) n FROM raw GROUP BY "입장권고유번호") GROUP BY 1 ORDER BY 1 LIMIT 8'))

print("\n2) 회차의 키 후보")
print("   공연코드 + 공연일시 조합 수:", q('SELECT count(DISTINCT ("공연코드","공연일시")) FROM raw')[0][0])
print("   공연회차 열의 값:", q('SELECT DISTINCT "공연회차" FROM raw'))
print("   공연 하나의 회차 수 분포 상위:", q(
    'SELECT n, count(*) FROM (SELECT count(DISTINCT "공연일시") n FROM raw GROUP BY "공연코드") GROUP BY 1 ORDER BY 2 DESC LIMIT 6'))

print("\n3) 공연과 공연장의 관계")
print("   공연 하나가 쓰는 공연장 수:", q(
    'SELECT n, count(*) FROM (SELECT count(DISTINCT "공연장코드") n FROM raw GROUP BY "공연코드") GROUP BY 1'))
print("   공연장 하나가 속한 시설 수:", q(
    'SELECT n, count(*) FROM (SELECT count(DISTINCT "공연시설코드") n FROM raw GROUP BY "공연장코드") GROUP BY 1'))
print("   시설 하나의 공연장 수 분포:", q(
    'SELECT n, count(*) FROM (SELECT count(DISTINCT "공연장코드") n FROM raw GROUP BY "공연시설코드") GROUP BY 1 ORDER BY 1'))

print("\n4) 공연명은 키가 될 수 있나")
print("   공연코드 수 · 공연명 수:", q('SELECT count(DISTINCT "공연코드"), count(DISTINCT "공연명") FROM raw')[0])
print("   같은 이름을 쓰는 공연코드:", q(
    'SELECT "공연명", count(DISTINCT "공연코드") FROM raw GROUP BY 1 HAVING count(DISTINCT "공연코드") > 1'))

print("\n5) 장르 계층")
for r in q('SELECT "장르명", list(DISTINCT "세부장르명") FROM raw GROUP BY 1 ORDER BY 1'):
    print("   ", r[0], "->", r[1])
print("   세부장르 하나가 속한 장르 수:", q(
    'SELECT n, count(*) FROM (SELECT count(DISTINCT "장르명") n FROM raw GROUP BY "세부장르명") GROUP BY 1'))

print("\n6) 여러 값이 한 칸에 든 열")
for c in ("출연진내용", "제작진내용", "기획제작사명"):
    r = q(f"""SELECT count(*), sum(CASE WHEN "{c}" LIKE '%,%' THEN 1 ELSE 0 END), max(len(string_split("{c}", ',')))
              FROM (SELECT DISTINCT "공연코드", "{c}" FROM raw WHERE "{c}" IS NOT NULL AND "{c}" <> '')""")[0]
    print(f"   {c}: 값 있는 공연 {r[0]} · 쉼표로 여럿 {r[1]} · 한 칸 최대 {r[2]}개")
print("   기획제작사명의 괄호 역할:", q(
    """SELECT org_role, count(*) FROM (
         SELECT regexp_extract(trim(x), '\\(([^)]*)\\)$', 1) AS org_role
         FROM (SELECT DISTINCT "공연코드", unnest(string_split("기획제작사명", ',')) x FROM raw WHERE "기획제작사명" <> '')
       ) GROUP BY 1 ORDER BY 2 DESC LIMIT 8"""))

print("\n7) 코드와 이름의 짝")
for code, name in (("예매/취소방식코드", "예매/취소방식명(관리시스템)"), ("결제수단코드", "결제수단명(관리시스템)"),
                   ("할인종류코드", "할인종류명(관리시스템)")):
    print(f"   {code}:", q(f'SELECT DISTINCT "{code}", "{name}" FROM raw ORDER BY 1'))

print("\n8) 가공 파일에는 공연코드가 있나")
for f in ("except_complex.csv", "complex.csv"):
    p = (ROOT / "data" / "processed" / f)
    if not p.exists():
        p = ROOT / "_source" / "data" / f
    cols = [r[0] for r in q(f"DESCRIBE SELECT * FROM read_csv('{p.as_posix()}', all_varchar=true, header=true)")]
    n, d = q(f"SELECT count(*), count(DISTINCT \"공연명\") FROM read_csv('{p.as_posix()}', all_varchar=true, header=true)")[0]
    print(f"   {f}: 열 {cols}")
    print(f"      행 {n:,} · 공연명 고유 {d:,} · 이름이 겹치는 행 {n - d:,}")
