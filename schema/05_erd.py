"""schema.sql 을 읽어 전체 ERD 를 그림으로 만든다.

표와 열, 기본키, 외래키를 schema.sql 에서 그대로 읽으므로 스키마를 고치면 그림도 따라 바뀐다.
표의 자리만 이 파일에서 정한다.

실행: python schema/05_erd.py  →  schema/erd.svg
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding="utf-8")

KOREAN = {
    "facility": "공연시설", "hall": "공연장", "genre": "장르", "subgenre": "세부장르",
    "performance": "공연", "performance_genre": "공연-장르", "person": "인물", "organization": "단체",
    "performance_person": "공연-인물", "performance_org": "공연-단체", "show_session": "회차",
    "ticket_event": "거래 (예매·취소)", "operator": "전송사업자", "booking_channel": "예매 방식",
    "payment_method": "결제 수단", "discount_type": "할인 종류",
}
GROUP = {
    "facility": "place", "hall": "place",
    "genre": "genre", "subgenre": "genre", "performance_genre": "link",
    "performance": "core", "show_session": "core", "ticket_event": "core",
    "person": "party", "organization": "party", "performance_person": "link", "performance_org": "link",
    "operator": "code", "booking_channel": "code", "payment_method": "code", "discount_type": "code",
}
COLOR = {"core": "#1f4e79", "place": "#2e7d5b", "genre": "#8a5a00", "party": "#6a3d9a", "link": "#555f6b", "code": "#7a7a7a"}
LEGEND = [("core", "공연 · 회차 · 거래"), ("place", "시설 · 공연장"), ("genre", "장르"), ("party", "인물 · 단체"),
          ("link", "연결 테이블 (N:M)"), ("code", "코드 테이블")]
POS = {
    "facility": (40, 70), "hall": (40, 470),
    "genre": (380, 70), "subgenre": (380, 250),
    "performance": (380, 470),
    "performance_genre": (720, 70),
    "performance_person": (720, 330), "person": (1060, 330),
    "performance_org": (720, 500), "organization": (1060, 500),
    "show_session": (380, 850), "ticket_event": (720, 690),
    "operator": (1060, 690), "booking_channel": (1060, 800), "payment_method": (1060, 910), "discount_type": (1060, 1020),
}
W, HEAD, ROW = 280, 30, 19
DETOUR = {("person", "genre_id"): 292}    # 선이 지나갈 높이


def parse(sql):
    sql = re.sub(r"--[^\n]*", "", sql)
    tables = {}
    for m in re.finditer(r"CREATE TABLE (\w+)\s*\((.*?)\);", sql, flags=re.S):
        name, body = m.group(1), m.group(2)
        cols, pk, fks = [], set(), {}
        parts, depth, cur = [], 0, ""
        for ch in body:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        parts.append(cur)
        for p in parts:
            p = " ".join(p.split())
            if not p:
                continue
            if p.upper().startswith("PRIMARY KEY"):
                pk |= {c.strip() for c in re.search(r"\((.*?)\)", p).group(1).split(",")}
                continue
            if p.upper().startswith("UNIQUE"):
                continue
            col, typ = p.split()[0], p.split()[1]
            cols.append((col, typ))
            if "PRIMARY KEY" in p.upper():
                pk.add(col)
            ref = re.search(r"REFERENCES (\w+)\((\w+)\)", p)
            if ref:
                fks[col] = ref.group(1)
        tables[name] = {"cols": cols, "pk": pk, "fk": fks}
    return tables


def box(name, t):
    x, y = POS[name]
    h = HEAD + ROW * len(t["cols"]) + 8
    return x, y, W, h


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def draw(tables):
    height = max(box(n, t)[1] + box(n, t)[3] for n, t in tables.items()) + 60
    width = 1380
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
         f'font-family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif">',
         f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
         '<text x="40" y="38" font-size="20" font-weight="700" fill="#1a1a1a">KOPIS 공연 데이터 관계형 모델</text>',
         '<text x="40" y="58" font-size="12" fill="#555">굵은 열 = 기본키(PK) · 기울인 열 = 외래키(FK) · 선의 갈래 쪽이 여럿(N)</text>']

    lx = 470
    for g, label in LEGEND:
        o.append(f'<rect x="{lx}" y="28" width="12" height="12" rx="2" fill="{COLOR[g]}"/>')
        o.append(f'<text x="{lx + 17}" y="38" font-size="11" fill="#333">{label}</text>')
        lx += 30 + len(label) * 12

    # 선을 먼저 그린다
    ports = {}
    for name, t in tables.items():
        x, y, w, h = box(name, t)
        ys = {c: y + HEAD + ROW * i + ROW / 2 + 4 for i, (c, _) in enumerate(t["cols"])}
        ports[name] = (x, y, w, h, ys)
    for name, t in tables.items():
        x, y, w, h, ys = ports[name]
        for col, target in t["fk"].items():
            tx, ty, tw, th, tys = ports[target]
            cy = ys[col]
            pk_col = next(iter(tables[target]["pk"]))
            py = tys.get(pk_col, ty + th / 2)
            if (name, col) in DETOUR:             # 다른 표를 가로지르지 않게 바깥으로 돌린다
                c = COLOR[GROUP[target]]
                rx, top = x + w + 24, DETOUR[(name, col)]
                ex = tx + tw
                o.append(f'<path d="M {x + w + 12} {cy} L {rx} {cy} L {rx} {top} L {ex + 30} {top} L {ex + 30} {py} L {ex} {py}" '
                         f'fill="none" stroke="{c}" stroke-width="1.4" opacity="0.85"/>')
                o.append(f'<line x1="{x + w + 12}" y1="{cy}" x2="{x + w}" y2="{cy - 6}" stroke="{c}" stroke-width="1.4"/>')
                o.append(f'<line x1="{x + w + 12}" y1="{cy}" x2="{x + w}" y2="{cy + 6}" stroke="{c}" stroke-width="1.4"/>')
                o.append(f'<line x1="{x + w + 12}" y1="{cy}" x2="{x + w}" y2="{cy}" stroke="{c}" stroke-width="1.4"/>')
                o.append(f'<line x1="{ex + 8}" y1="{py - 6}" x2="{ex + 8}" y2="{py + 6}" stroke="{c}" stroke-width="1.6"/>')
                continue
            if tx + tw <= x:                      # 대상이 왼쪽
                sx, ex = x, tx + tw
                mid = (sx + ex) / 2
                fork = [(sx, cy - 6), (sx, cy + 6)]
                tip = sx - 12
            elif tx >= x + w:                     # 대상이 오른쪽
                sx, ex = x + w, tx
                mid = (sx + ex) / 2
                fork = [(sx, cy - 6), (sx, cy + 6)]
                tip = sx + 12
            else:                                 # 위아래로 놓임: 왼쪽으로 돌아간다
                sx, ex = x, tx
                mid = min(x, tx) - 22 - 6 * (list(t["fk"]).index(col))
                fork = [(sx, cy - 6), (sx, cy + 6)]
                tip = sx - 12
            c = COLOR[GROUP[target]]
            o.append(f'<path d="M {tip} {cy} L {mid} {cy} L {mid} {py} L {ex} {py}" fill="none" stroke="{c}" stroke-width="1.4" opacity="0.85"/>')
            for fx, fy in fork:
                o.append(f'<line x1="{tip}" y1="{cy}" x2="{fx}" y2="{fy}" stroke="{c}" stroke-width="1.4"/>')
            o.append(f'<line x1="{tip}" y1="{cy}" x2="{sx}" y2="{cy}" stroke="{c}" stroke-width="1.4"/>')
            bar = ex - 8 if ex > mid else ex + 8
            o.append(f'<line x1="{bar}" y1="{py - 6}" x2="{bar}" y2="{py + 6}" stroke="{c}" stroke-width="1.6"/>')

    for name, t in tables.items():
        x, y, w, h, ys = ports[name]
        c = COLOR[GROUP[name]]
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="{c}" stroke-width="1.6"/>')
        o.append(f'<path d="M {x} {y + HEAD} L {x} {y + 6} Q {x} {y} {x + 6} {y} L {x + w - 6} {y} Q {x + w} {y} {x + w} {y + 6} L {x + w} {y + HEAD} Z" fill="{c}"/>')
        o.append(f'<text x="{x + 10}" y="{y + 20}" font-size="13" font-weight="700" fill="#ffffff">{name}</text>')
        o.append(f'<text x="{x + w - 10}" y="{y + 20}" font-size="12" fill="#ffffff" text-anchor="end">{KOREAN[name]}</text>')
        for i, (col, typ) in enumerate(t["cols"]):
            cy = y + HEAD + ROW * i + ROW / 2 + 8
            if i % 2 == 1:
                o.append(f'<rect x="{x + 1}" y="{cy - 14}" width="{w - 2}" height="{ROW}" fill="#f5f7f9"/>')
            is_pk, is_fk = col in t["pk"], col in t["fk"]
            tag = "PK" if is_pk and not is_fk else ("PK·FK" if is_pk and is_fk else ("FK" if is_fk else ""))
            weight = ' font-weight="700"' if is_pk else ""
            style = ' font-style="italic"' if is_fk else ""
            o.append(f'<text x="{x + 10}" y="{cy}" font-size="11.5" fill="#1a1a1a"{weight}{style}>{esc(col)}</text>')
            o.append(f'<text x="{x + w - 52}" y="{cy}" font-size="10.5" fill="#777" text-anchor="end">{typ.lower()}</text>')
            if tag:
                o.append(f'<text x="{x + w - 10}" y="{cy}" font-size="10" font-weight="700" fill="{c}" text-anchor="end">{tag}</text>')
    o.append("</svg>")
    return "\n".join(o)


def main():
    tables = parse((HERE / "schema.sql").read_text(encoding="utf-8"))
    missing = set(tables) - set(POS)
    assert not missing, f"자리가 정해지지 않은 표: {missing}"
    (HERE / "erd.svg").write_text(draw(tables), encoding="utf-8")
    n_fk = sum(len(t["fk"]) for t in tables.values())
    print(f"표 {len(tables)}개 · 열 {sum(len(t['cols']) for t in tables.values())}개 · 외래키 {n_fk}개 → schema/erd.svg")
    for n, t in tables.items():
        print(f"  {n:20s} {KOREAN[n]:14s} 열 {len(t['cols']):2d} · PK {sorted(t['pk'])} · FK {t['fk']}")


if __name__ == "__main__":
    main()
