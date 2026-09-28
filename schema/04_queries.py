"""나눈 테이블로 분석에 필요한 값을 다시 만들 수 있는지 확인한다.

  1. 거래에서 공연 단위 성과 지표를 집계한다 (성과지표의 입력)
  2. 관계 테이블에서 그래프의 엣지를 뽑는다 (분류 모델의 입력)
  3. 복합 공연의 장르 조합을 조회한다

실행: python schema/04_queries.py
"""
import sys
from pathlib import Path

import duckdb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
con = duckdb.connect(str(ROOT / "data" / "kopis.duckdb"))

Q1 = """
CREATE OR REPLACE VIEW v_performance_metrics AS
WITH session_sales AS (          -- 회차마다 예매에서 취소를 뺀다
    SELECT e.session_id,
           sum(CASE WHEN e.event_type = 'booking' THEN e.quantity ELSE -e.quantity END) AS seats_sold,
           sum(CASE WHEN e.event_type = 'booking' THEN e.amount   ELSE -e.amount   END) AS revenue
    FROM ticket_event e GROUP BY e.session_id
), perf_sales AS (             -- 공연마다 회차를 모은다
    SELECT ss.performance_id, count(*) AS sessions, sum(s.seats_sold) AS seats_sold, sum(s.revenue) AS revenue
    FROM show_session ss JOIN session_sales s USING (session_id) GROUP BY ss.performance_id
)
SELECT p.performance_id, p.title, g.genre_name, f.region, h.hall_name, h.seats,
       ps.sessions, ps.seats_sold, ps.revenue,
       ps.seats_sold / (ps.sessions * h.seats)      AS occupancy,
       ps.revenue / nullif(ps.seats_sold, 0)        AS avg_price,
       (CAST(f.has_restaurant AS INT) + CAST(f.has_cafe AS INT) + CAST(f.has_store AS INT)
        + CAST(f.has_playroom AS INT) + CAST(f.has_nursing_room AS INT)
        + CAST(f.acc_parking AS INT) + CAST(f.acc_restroom AS INT) + CAST(f.acc_ramp AS INT)
        + CAST(f.acc_elevator AS INT) + CAST(f.parking_own AS INT) + CAST(f.parking_public AS INT)) AS access_score
FROM performance p
JOIN perf_sales ps USING (performance_id)
JOIN hall h USING (hall_code)
JOIN facility f USING (facility_code)
JOIN subgenre sg USING (subgenre_id)
JOIN genre g USING (genre_id)
"""
con.execute(Q1)
print("1) 거래에서 집계한 공연 단위 지표 (매출 상위 5)")
for r in con.execute("""SELECT title, genre_name, region, sessions, seats_sold, round(revenue), round(occupancy, 3),
                               round(avg_price), access_score
                        FROM v_performance_metrics ORDER BY revenue DESC LIMIT 5""").fetchall():
    print("  ", r)
n, tot = con.execute("SELECT count(*), sum(revenue) FROM v_performance_metrics").fetchone()
raw = con.execute("""SELECT sum(CASE WHEN "예매/취소구분"='1' THEN CAST("예매/취소금액" AS DOUBLE)
                                     ELSE -CAST("예매/취소금액" AS DOUBLE) END) FROM raw""").fetchone()[0]
print(f"   공연 {n} 건 · 순매출 합 {tot:,.0f} · 원천에서 직접 계산 {raw:,.0f} → {'일치' if abs(tot - raw) < 1 else '불일치'}")
over = con.execute("SELECT count(*) FROM v_performance_metrics WHERE occupancy > 1").fetchone()[0]
print(f"   좌석점유율이 1 을 넘는 공연: {over} 건 (좌석수보다 많이 팔린 기록. 보정이 필요했던 이유)")

Q2 = """
CREATE OR REPLACE VIEW v_graph_edge AS
SELECT 'performance:' || performance_id AS source, 'person:' || person_id AS target, 'has_' || role AS relation
FROM performance_person
UNION ALL
SELECT 'performance:' || performance_id, 'org:' || org_id, 'by_org' FROM (SELECT DISTINCT performance_id, org_id FROM performance_org)
UNION ALL
SELECT 'performance:' || performance_id, 'genre:' || genre_id, 'labeled_as'
FROM performance_genre pg JOIN genre g USING (genre_id) WHERE pg.basis = 'label' AND NOT g.is_composite
"""
con.execute(Q2)
print("\n2) 관계 테이블에서 뽑은 그래프 엣지")
for r in con.execute("SELECT relation, count(*) FROM v_graph_edge GROUP BY 1 ORDER BY 2 DESC").fetchall():
    print("  ", r)

print("\n3) 복합 공연의 장르 조합 (둘째 장르 확률이 높은 순 5)")
for r in con.execute("""
SELECT p.title, g1.genre_name, round(a.probability, 3), g2.genre_name, round(b.probability, 3)
FROM performance p
JOIN performance_genre a ON a.performance_id = p.performance_id AND a.basis = 'predicted' AND a.rank_no = 1
JOIN performance_genre b ON b.performance_id = p.performance_id AND b.basis = 'predicted' AND b.rank_no = 2
JOIN genre g1 ON g1.genre_id = a.genre_id JOIN genre g2 ON g2.genre_id = b.genre_id
ORDER BY b.probability DESC LIMIT 5""").fetchall():
    print("  ", r)
r = con.execute("""
SELECT count(*), sum(CASE WHEN b.probability >= 0.1 THEN 1 ELSE 0 END)
FROM performance_genre b WHERE b.basis = 'predicted' AND b.rank_no = 2""").fetchone()
print(f"   둘째 장르 확률이 0.1 이상: {r[1]} / {r[0]}")
con.close()
