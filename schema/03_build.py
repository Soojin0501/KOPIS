"""schema.sql 대로 테이블을 만들고 데이터를 적재한 뒤, 나눈 결과가 원천과 맞는지 검증한다.

입력
  data/raw/temp.csv                 거래 단위 원천 (표본)
  data/processed/except_complex.csv 단일 장르 공연 목록
  data/processed/complex.csv        복합 장르 공연 목록
  results/classification/complex_predictions_final.csv  복합 공연의 장르 추정

실행: python schema/03_build.py
"""
import sys
from pathlib import Path

import duckdb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "kopis.duckdb"


def first_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return Path(p).as_posix()
    raise FileNotFoundError(paths)


SINGLE = first_existing(ROOT / "data/processed/except_complex.csv", ROOT / "_source/data/except_complex.csv")
COMPOSITE = first_existing(ROOT / "data/processed/complex.csv", ROOT / "_source/data/complex.csv")
PRED = first_existing(
    ROOT / "results/classification/complex_predictions_final.csv",
    *(ROOT / "_source/model").glob("*진짜 최종*.csv"),
)

con = duckdb.connect(str(DB))
tables = [
    "ticket_event", "show_session", "performance_org", "performance_person", "organization", "person",
    "performance_genre", "performance", "discount_type", "payment_method", "booking_channel",
    "subgenre", "genre", "hall", "facility", "operator",
]
for t in tables:
    con.execute(f"DROP TABLE IF EXISTS {t}")
con.execute((ROOT / "schema" / "schema.sql").read_text(encoding="utf-8"))

B = "CASE WHEN {c} IN ('True','true','Y','1') THEN true WHEN {c} IN ('False','false','N','0') THEN false END"


def b(col):
    return B.format(c=f'"{col}"')


# ── 기준 정보 ──
con.execute('INSERT INTO operator SELECT DISTINCT "전송사업자코드", "전송사업자명" FROM raw')
con.execute(f"""
INSERT INTO facility
SELECT "공연시설코드", any_value("공연시설명"), any_value("시설특성"), TRY_CAST(any_value("개관연도") AS INTEGER),
       any_value("주소"), any_value("공연지역명"),
       any_value({b('편의시설_레스토랑 여부')}), any_value({b('편의시설_카페 여부')}), any_value({b('편의시설_편의점 여부')}),
       any_value({b('편의시설_놀이방 여부')}), any_value({b('편의시설_수유실 여부')}),
       any_value({b('장애인시설_주차장 여부')}), any_value({b('장애인시설_화장실 여부')}),
       any_value({b('장애인시설_경사로 여부')}), any_value({b('장애인시설_전용엘리베이터 여부')}),
       any_value({b('주차시설_자체 여부')}), any_value({b('주차시설_공영 여부')})
FROM raw GROUP BY 1""")
con.execute(f"""
INSERT INTO hall
SELECT "공연장코드", any_value("공연시설코드"), any_value("공연장명"),
       TRY_CAST(TRY_CAST(any_value("좌석수") AS DOUBLE) AS INTEGER), TRY_CAST(any_value("장애인석") AS INTEGER),
       any_value({b('무대시설_오케스트라피트 여부')}), any_value({b('무대시설_연습실 여부')}),
       any_value({b('무대시설_분장실 여부')}), any_value("무대시설_무대넓이")
FROM raw GROUP BY 1""")
con.execute('INSERT INTO booking_channel SELECT DISTINCT "예매/취소방식코드", "예매/취소방식명(관리시스템)" FROM raw')
con.execute('INSERT INTO payment_method SELECT DISTINCT "결제수단코드", "결제수단명(관리시스템)" FROM raw')
con.execute('INSERT INTO discount_type SELECT DISTINCT "할인종류코드", "할인종류명(관리시스템)" FROM raw')

# ── 공연 목록을 한곳에 모은다 ──
con.execute(f"""
CREATE OR REPLACE TEMP TABLE catalog AS
SELECT 'raw_sample' AS source, "공연코드" AS code, "공연명" AS title, any_value("공연장코드") AS hall_code,
       any_value("장르명") AS genre, any_value("세부장르명") AS subgenre,
       any_value("출연진내용") AS cast_txt, any_value("제작진내용") AS crew_txt, any_value("기획제작사명") AS org_txt,
       TRY_CAST(any_value("공연시작일자") AS DATE) AS start_date, TRY_CAST(any_value("공연종료일자") AS DATE) AS end_date,
       any_value("소요시간") AS running_time, any_value("관람연령") AS age_limit,
       any_value({b('아동공연 여부')}) AS is_child, any_value({b('축제 여부')}) AS is_festival,
       any_value({b('내한공연 여부')}) AS is_visiting, any_value({b('오픈런 여부')}) AS is_open_run,
       0 AS ord
FROM raw GROUP BY "공연코드", "공연명"
UNION ALL
SELECT 'single', NULL, "공연명", NULL, "장르명", "세부장르명", "출연진내용", "제작진내용", "기획제작사명",
       NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, row_number() OVER ()
FROM read_csv('{SINGLE}', all_varchar=true, header=true)
UNION ALL
SELECT 'composite', NULL, "공연명", NULL, "장르명", "세부장르명", "출연진내용", "제작진내용", "기획제작사명",
       NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, row_number() OVER ()
FROM read_csv('{COMPOSITE}', all_varchar=true, header=true)
""")
con.execute("ALTER TABLE catalog ADD COLUMN pid INTEGER")
con.execute("""
UPDATE catalog SET pid = s.pid FROM (
  SELECT rowid AS rid, row_number() OVER (ORDER BY CASE source WHEN 'raw_sample' THEN 0 WHEN 'single' THEN 1 ELSE 2 END, ord, code) AS pid
  FROM catalog) s WHERE catalog.rowid = s.rid""")

# 장르와 세부장르
con.execute("""
INSERT INTO genre
SELECT row_number() OVER (ORDER BY g), g, g = '복합'
FROM (SELECT DISTINCT trim(regexp_replace(genre, '\\s*\\(.*?\\)', '')) AS g FROM catalog WHERE genre IS NOT NULL)""")
con.execute("""
INSERT INTO subgenre
SELECT row_number() OVER (ORDER BY genre_id, s), genre_id, s
FROM (SELECT DISTINCT g.genre_id, c.subgenre AS s
      FROM catalog c JOIN genre g ON g.genre_name = trim(regexp_replace(c.genre, '\\s*\\(.*?\\)', ''))
      WHERE c.subgenre IS NOT NULL)""")
con.execute("""
INSERT INTO performance
SELECT c.pid, c.code, c.title, c.hall_code, sg.subgenre_id, c.start_date, c.end_date, c.running_time, c.age_limit,
       c.is_child, c.is_festival, c.is_visiting, c.is_open_run, c.source
FROM catalog c
JOIN genre g ON g.genre_name = trim(regexp_replace(c.genre, '\\s*\\(.*?\\)', ''))
LEFT JOIN subgenre sg ON sg.genre_id = g.genre_id AND sg.subgenre_name = c.subgenre""")

# 공연-장르: 전산망 기록
con.execute("""
INSERT INTO performance_genre
SELECT c.pid, g.genre_id, 'label', 1, NULL
FROM catalog c JOIN genre g ON g.genre_name = trim(regexp_replace(c.genre, '\\s*\\(.*?\\)', ''))""")

# ── 한 칸에 여럿 든 이름을 행으로 편다 ──
CLEAN = "trim(regexp_replace(regexp_replace(x, '\\(.*?\\)', '', 'g'), '\\s*(등|외|外)$', ''))"
con.execute(f"""
CREATE OR REPLACE TEMP TABLE people AS
SELECT DISTINCT pid, role, name, genre FROM (
  SELECT pid, 'cast' AS role, {CLEAN} AS name, trim(regexp_replace(genre, '\\s*\\(.*?\\)', '')) AS genre
  FROM (SELECT pid, genre, unnest(string_split(cast_txt, ',')) AS x FROM catalog WHERE cast_txt IS NOT NULL AND cast_txt <> '')
  UNION ALL
  SELECT pid, 'crew', {CLEAN}, trim(regexp_replace(genre, '\\s*\\(.*?\\)', ''))
  FROM (SELECT pid, genre, unnest(string_split(crew_txt, ',')) AS x FROM catalog WHERE crew_txt IS NOT NULL AND crew_txt <> '')
) WHERE name <> ''""")
# 인물의 준식별자는 (이름, 장르)다. 장르가 정해진 단일 공연에서만 만들 수 있다.
con.execute("""
INSERT INTO person
SELECT row_number() OVER (ORDER BY name, genre_id), name, genre_id
FROM (SELECT DISTINCT p.name, g.genre_id FROM people p JOIN genre g ON g.genre_name = p.genre
      WHERE p.genre <> '복합')""")
con.execute("""
INSERT INTO performance_person
SELECT DISTINCT p.pid, pe.person_id, p.role
FROM people p JOIN genre g ON g.genre_name = p.genre
JOIN person pe ON pe.person_name = p.name AND pe.genre_id = g.genre_id
WHERE p.genre <> '복합'""")

# 복합 공연의 참여자는 장르를 모른다. 같은 이름으로 단일 공연에 가장 많이 참여한 인물에 잇는다.
con.execute("""
CREATE OR REPLACE TEMP TABLE dominant AS
SELECT person_name, person_id, n_genres FROM (
  SELECT pe.person_name, pe.person_id, count(*) AS n,
         count(*) OVER (PARTITION BY pe.person_name) AS n_genres,
         row_number() OVER (PARTITION BY pe.person_name ORDER BY count(*) DESC, pe.person_id) AS rk
  FROM person pe JOIN performance_person pp USING (person_id)
  GROUP BY pe.person_name, pe.person_id) WHERE rk = 1""")
COMP_STATS = con.execute("""
SELECT count(DISTINCT p.name),
       count(DISTINCT CASE WHEN d.person_id IS NULL THEN p.name END),
       count(DISTINCT CASE WHEN d.n_genres = 1 THEN p.name END),
       count(DISTINCT CASE WHEN d.n_genres > 1 THEN p.name END)
FROM people p LEFT JOIN dominant d ON d.person_name = p.name WHERE p.genre = '복합'""").fetchone()
con.execute("""
INSERT INTO person
SELECT (SELECT max(person_id) FROM person) + row_number() OVER (ORDER BY name), name,
       (SELECT genre_id FROM genre WHERE genre_name = '복합')
FROM (SELECT DISTINCT p.name FROM people p LEFT JOIN dominant d ON d.person_name = p.name
      WHERE p.genre = '복합' AND d.person_id IS NULL)""")
con.execute("""
INSERT INTO performance_person
SELECT DISTINCT p.pid, coalesce(d.person_id, pe.person_id), p.role
FROM people p
LEFT JOIN dominant d ON d.person_name = p.name
LEFT JOIN person pe ON pe.person_name = p.name
     AND pe.genre_id = (SELECT genre_id FROM genre WHERE genre_name = '복합')
WHERE p.genre = '복합'""")

con.execute("""
CREATE OR REPLACE TEMP TABLE orgs AS
SELECT DISTINCT pid,
       trim(regexp_replace(x, '\\(([^)]*)\\)\\s*$', '')) AS name,
       coalesce(nullif(regexp_extract(trim(x), '\\(([^)]*)\\)\\s*$', 1), ''), '미상') AS role
FROM (SELECT pid, unnest(string_split(org_txt, ',')) AS x FROM catalog WHERE org_txt IS NOT NULL AND org_txt <> '')""")
con.execute("DELETE FROM orgs WHERE name = ''")
con.execute("INSERT INTO organization SELECT row_number() OVER (ORDER BY name), name FROM (SELECT DISTINCT name FROM orgs)")
con.execute("""
INSERT INTO performance_org
SELECT DISTINCT o.pid, og.org_id, o.role FROM orgs o JOIN organization og ON og.org_name = o.name""")

# 공연-장르: 모델 추정. 추정 파일은 제외 규칙을 거친 복합 공연과 순서가 같다
con.execute(f"""
CREATE OR REPLACE TEMP TABLE pred AS
SELECT row_number() OVER () AS rn, * FROM read_csv('{PRED}', all_varchar=true, header=true)""")
con.execute("""
CREATE OR REPLACE TEMP TABLE comp AS
SELECT row_number() OVER (ORDER BY c.ord) AS rn, c.pid, c.title
FROM catalog c
WHERE c.source = 'composite' AND c.genre <> '서커스/마술' AND coalesce(c.subgenre, '') <> '다원/융복합'""")
match = con.execute("""
SELECT count(*), sum(CASE WHEN c.title = p."공연명" THEN 1 ELSE 0 END)
FROM comp c JOIN pred p USING (rn)""").fetchone()
con.execute("""
INSERT INTO performance_genre
SELECT pid, genre_id, 'predicted', rank_no, prob FROM (
  SELECT c.pid, g.genre_id, 1 AS rank_no, CAST(p."확률1" AS DOUBLE) AS prob
  FROM comp c JOIN pred p USING (rn) JOIN genre g ON g.genre_name = p."장르1" WHERE c.title = p."공연명"
  UNION ALL
  SELECT c.pid, g.genre_id, 2, CAST(p."확률2" AS DOUBLE)
  FROM comp c JOIN pred p USING (rn) JOIN genre g ON g.genre_name = p."장르2" WHERE c.title = p."공연명")""")

# ── 회차와 거래 ──
con.execute("""
INSERT INTO show_session
SELECT row_number() OVER (ORDER BY pf.performance_id, s.show_at), pf.performance_id, s.show_at
FROM (SELECT DISTINCT "공연코드" AS code, CAST("공연일시" AS TIMESTAMP) AS show_at FROM raw) s
JOIN performance pf ON pf.performance_code = s.code""")
con.execute("""
INSERT INTO ticket_event
SELECT row_number() OVER (), r."입장권고유번호", ss.session_id, r."전송사업자코드",
       CASE r."예매/취소구분" WHEN '1' THEN 'booking' ELSE 'cancel' END,
       CAST(r."예매/취소일시" AS TIMESTAMP),
       CAST(CAST(r."예매/취소매수" AS DOUBLE) AS INTEGER), CAST(r."예매/취소금액" AS DOUBLE),
       TRY_CAST(r."장당금액" AS DOUBLE), TRY_CAST(r."할인금액" AS DOUBLE),
       r."예매/취소방식코드", r."결제수단코드", r."할인종류코드", r."성별", TRY_CAST(r."연령" AS INTEGER)
FROM raw r
JOIN performance pf ON pf.performance_code = r."공연코드"
JOIN show_session ss ON ss.performance_id = pf.performance_id AND ss.show_at = CAST(r."공연일시" AS TIMESTAMP)""")


# ── 검증 ──
def one(sql):
    return con.execute(sql).fetchone()


print("테이블별 행 수")
for t in reversed(tables):
    print(f"  {t:20s} {one(f'SELECT count(*) FROM {t}')[0]:>10,}")

print("\n검증")
raw_n = one("SELECT count(*) FROM raw")[0]
ev_n = one("SELECT count(*) FROM ticket_event")[0]
print(f"  거래 행 보존: 원천 {raw_n:,} = 거래 {ev_n:,} → {'일치' if raw_n == ev_n else '불일치'}")
a = one('SELECT sum(CAST("예매/취소금액" AS DOUBLE)) FROM raw')[0]
bsum = one("SELECT sum(amount) FROM ticket_event")[0]
print(f"  금액 합 보존: 원천 {a:,.0f} = 거래 {bsum:,.0f} → {'일치' if abs(a - bsum) < 1 else '불일치'}")
cells_raw = raw_n * 71
cells_new = sum(
    one(f"SELECT count(*) FROM {t}")[0] * len(con.execute(f"DESCRIBE {t}").fetchall())
    for t in ("operator", "facility", "hall", "show_session", "ticket_event", "booking_channel", "payment_method", "discount_type")
) + one("SELECT count(*) FROM performance WHERE source='raw_sample'")[0] * 14
print(f"  칸 수: 원천 {cells_raw:,} → 나눈 뒤 {cells_new:,} ({100 * cells_new / cells_raw:.1f}%)")
print(f"  복합 공연 추정 연결: {match[1]:,} / {match[0]:,} 건이 이름까지 일치")

print("\n설계 고민을 뒷받침하는 수치")
r = one("SELECT count(*), count(DISTINCT title) FROM performance WHERE source <> 'raw_sample'")
print(f"  공연명은 키가 못 된다: 공연 {r[0]:,} 건 중 이름 고유 {r[1]:,} 건, 겹치는 행 {r[0] - r[1]:,}")
r = one("SELECT count(*), count(DISTINCT person_name) FROM person")
multi = one("SELECT count(*) FROM (SELECT person_name FROM person GROUP BY 1 HAVING count(*) > 1)")[0]
print(f"  인물: 이름 {r[1]:,} 개가 (이름, 장르) 기준 {r[0]:,} 명으로 갈린다. 둘 이상 장르에 걸친 이름 {multi:,} 개")
top = con.execute("""
SELECT person_name, count(*) AS genres, string_agg(g.genre_name, ', ') FROM person p JOIN genre g USING (genre_id)
GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 3""").fetchall()
print("    가장 많은 장르에 걸친 이름:", top)
print(f"  복합 공연 참여자 {COMP_STATS[0]:,} 명: 단일 공연 이력 없음 {COMP_STATS[1]:,} · "
      f"한 장르에만 있음 {COMP_STATS[2]:,} · 여러 장르에 걸쳐 어느 쪽인지 모호 {COMP_STATS[3]:,}")
r = con.execute("""
SELECT o.org_name, count(DISTINCT po.role) FROM performance_org po JOIN organization o USING (org_id)
GROUP BY 1 HAVING count(DISTINCT po.role) > 1 ORDER BY 2 DESC LIMIT 3""").fetchall()
n_multi = one("""SELECT count(*) FROM (SELECT org_id FROM performance_org GROUP BY 1 HAVING count(DISTINCT role) > 1)""")[0]
print(f"  단체의 역할은 단체가 아니라 관계에 붙는다: 역할이 둘 이상인 단체 {n_multi:,} 곳. 예 {r}")
both = one("""
SELECT count(*) FROM (SELECT DISTINCT person_name FROM person) p
JOIN organization o ON o.org_name = p.person_name""")[0]
print(f"  인물 이름과 단체 이름이 같은 경우: {both:,} 건 (한 표에 섞으면 구분이 안 된다)")
r = one("""SELECT count(*) FROM (SELECT ticket_no FROM ticket_event GROUP BY 1
           HAVING sum(CASE WHEN event_type='booking' THEN 1 ELSE 0 END) > 0
              AND sum(CASE WHEN event_type='cancel' THEN 1 ELSE 0 END) > 0)""")[0]
print(f"  예매와 취소가 모두 있는 입장권: {r:,} 장")
con.close()
