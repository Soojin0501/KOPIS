"""확인 1·2·3 — 복합 장르 분류 모델의 성능을 다시 잰다.

원래 노트북(genre_gcn_v3_final)과 같은 그래프, 같은 모델, 같은 학습 설정을 쓴다.
다른 점은 평가 방법 하나다.

  A. 원래 방식: 라벨이 있는 공연 전부로 학습하고, 그 공연들로 평가한다.
  B. 검증 분리: 라벨이 있는 공연의 80% 로 학습하고, 나머지 20% 로 평가한다.
  C. 기준선: 모델 없이, 참여자들이 학습 구간에서 가장 많이 참여한 장르로 맞힌다.

실행 (저장소 루트에서)
  python review/01_model_holdout.py
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, hamming_loss
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, to_undirected

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "review" / "out"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
EPOCHS = 100


def log(*a):
    print(*a, flush=True)


# ── 1. 데이터: 원래 노트북과 같은 순서 ──
df = pd.read_csv(ROOT / "data/processed/except_complex.csv", encoding="utf-8-sig")
dc = pd.read_csv(ROOT / "data/processed/complex.csv", encoding="utf-8-sig")
df = df[df["장르명"] != "서커스/마술"].reset_index(drop=True)
dc = dc[dc["장르명"] != "서커스/마술"].reset_index(drop=True)
dc = dc[dc["세부장르명"] != "다원/융복합"].reset_index(drop=True)


def split_and_clean(text):
    if pd.isna(text):
        return []
    names = text.replace(" ", "").split(",")
    names = [re.sub(r"\(.*?\)| 등| 외| 外", "", n).strip() for n in names]
    return [n for n in names if n]


for d in (df, dc):
    d["cast"] = d["출연진내용"].apply(split_and_clean)
    d["crew"] = d["제작진내용"].apply(split_and_clean)
    d["org"] = d["기획제작사명"].apply(split_and_clean)
df["source"], dc["source"] = "df", "df_complex"
df["oi"], dc["oi"] = df.index, dc.index
allp = pd.concat([df, dc], ignore_index=True)

node = {}
for _, r in allp.iterrows():
    for n in r["cast"] + r["crew"]:
        node.setdefault(f"인물_{n}", len(node))
for _, r in allp.iterrows():
    for n in r["org"]:
        node.setdefault(f"제작사_{n}", len(node))
for _, r in allp.iterrows():
    node.setdefault(f"공연_{r['source']}_{r['oi']}", len(node))
genres = sorted(df["장르명"].apply(lambda x: re.sub(r"\s*\(.*?\)", "", str(x)).strip()).unique())
for g in genres:
    node.setdefault(f"장르_{g}", len(node))
N, G = len(node), len(genres)
log(f"노드 {N:,} · 장르 {G} · 단일 공연 {len(df):,} · 복합 공연 {len(dc):,}")

corpus = [" ".join(r["cast"] + r["crew"] + r["org"]) for _, r in allp.iterrows()]
vec = TfidfVectorizer(token_pattern=r"[^ ]+")
tfidf = vec.fit_transform(corpus)
edges, weights, deg = [], [], []
for i, r in allp.iterrows():
    pid = node[f"공연_{r['source']}_{r['oi']}"]
    parts = r["cast"] + r["crew"] + r["org"]
    deg.append(len(parts))
    for n in parts:
        key = f"제작사_{n}" if n in r["org"] else f"인물_{n}"
        edges.append([pid, node[key]])
        col = vec.vocabulary_.get(n)
        weights.append(float(tfidf[i, col]) if col is not None else 0.0)
zero_w = sum(1 for w in weights if w == 0.0)
log(f"엣지 {len(edges):,} · 가중치가 0 인 엣지 {zero_w:,} ({100 * zero_w / len(edges):.1f}%)")
log(f"참여자가 한 명도 없는 공연: 단일 {sum(1 for d_ in deg[:len(df)] if d_ == 0):,} · 복합 {sum(1 for d_ in deg[len(df):] if d_ == 0):,}")

ei = torch.tensor(np.array(edges).T, dtype=torch.long)
ew = torch.tensor(weights, dtype=torch.float)
ei, ew = to_undirected(ei, ew, num_nodes=N)
ei, ew = add_self_loops(ei, ew, fill_value=1.0, num_nodes=N)

gid = {g: i for i, g in enumerate(genres)}
y = torch.zeros((N, G))
single_nodes = []
for _, r in df.iterrows():
    nid = node[f"공연_df_{r['oi']}"]
    y[nid, gid[re.sub(r"\s*\(.*?\)", "", str(r["장르명"])).strip()]] = 1
    single_nodes.append(nid)
single_nodes = torch.tensor(single_nodes)
comp_nodes = torch.tensor([node[f"공연_df_complex_{i}"] for i in dc["oi"]])


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N, 64)
        self.c1 = GCNConv(64, 64)
        self.c2 = GCNConv(64, G)

    def forward(self):
        x = self.c1(self.emb.weight, ei, ew).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.c2(x, ei, ew)


def train(train_idx, label):
    torch.manual_seed(SEED)
    counts = y[train_idx].sum(0).numpy()
    pw = torch.tensor((len(y) - counts) / (counts + 1e-6), dtype=torch.float)  # 원래 코드와 같게 둔다
    model = GCN()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = crit(model()[train_idx], y[train_idx])
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            log(f"  [{label}] epoch {ep + 1} loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model())


def metrics(probs, idx):
    yt = y[idx].numpy()
    p = probs[idx].numpy()
    yb = (p > 0.5).astype(int)
    true = yt.argmax(1)
    order = np.argsort(-p, axis=1)
    return {
        "n": int(len(idx)),
        "micro_f1": round(float(f1_score(yt, yb, average="micro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(yt, yb, average="macro", zero_division=0)), 4),
        "hamming_acc": round(float(1 - hamming_loss(yt, yb)), 4),
        "top1_acc": round(float((order[:, 0] == true).mean()), 4),
        "top2_acc": round(float(((order[:, 0] == true) | (order[:, 1] == true)).mean()), 4),
    }


res = {"genres": genres, "nodes": N, "edges": len(edges)}

log("\nA. 원래 방식: 전부로 학습, 같은 데이터로 평가")
pA = train(single_nodes, "A")
res["A_train_on_all_eval_on_train"] = metrics(pA, single_nodes)
log("  ", res["A_train_on_all_eval_on_train"])

log("\nB. 검증 분리: 80% 학습, 20% 평가")
g = torch.Generator().manual_seed(SEED)
perm = single_nodes[torch.randperm(len(single_nodes), generator=g)]
cut = int(0.8 * len(perm))
tr, va = perm[:cut], perm[cut:]
pB = train(tr, "B")
res["B_holdout_train_part"] = metrics(pB, tr)
res["B_holdout_valid_part"] = metrics(pB, va)
log("   학습 구간", res["B_holdout_train_part"])
log("   검증 구간", res["B_holdout_valid_part"])

# 검증 구간을 참여자 유무로 나눠 본다
deg_by_node = {node[f"공연_df_{i}"]: deg[i] for i in range(len(df))}
va_has = torch.tensor([n for n in va.tolist() if deg_by_node[n] > 0])
va_none = torch.tensor([n for n in va.tolist() if deg_by_node[n] == 0])
res["B_valid_with_participants"] = metrics(pB, va_has)
res["B_valid_without_participants"] = metrics(pB, va_none) if len(va_none) else None
log("   검증 중 참여자 있는 공연", res["B_valid_with_participants"])
log("   검증 중 참여자 없는 공연", res["B_valid_without_participants"])

# 장르별 검증 성능
yt = y[va].numpy().argmax(1)
pr = pB[va].numpy().argmax(1)
per = {}
for i, name in enumerate(genres):
    m = yt == i
    per[name] = {"n": int(m.sum()), "top1_acc": round(float((pr[m] == i).mean()), 4) if m.sum() else None}
res["B_valid_per_genre"] = per
log("   장르별", per)

log("\nC. 기준선: 참여자가 학습 구간에서 가장 많이 참여한 장르")
tr_set, idx_of = set(tr.tolist()), {node[f"공연_df_{i}"]: i for i in range(len(df))}
hist = defaultdict(Counter)
for n in tr.tolist():
    r = df.iloc[idx_of[n]]
    gname = re.sub(r"\s*\(.*?\)", "", str(r["장르명"])).strip()
    for p in r["cast"] + r["crew"] + r["org"]:
        hist[p][gname] += 1
major = Counter(re.sub(r"\s*\(.*?\)", "", str(x)).strip() for x in df.iloc[[idx_of[n] for n in tr.tolist()]]["장르명"]).most_common(1)[0][0]
hit1 = hit2 = 0
for n in va.tolist():
    r = df.iloc[idx_of[n]]
    gname = re.sub(r"\s*\(.*?\)", "", str(r["장르명"])).strip()
    c = Counter()
    for p in r["cast"] + r["crew"] + r["org"]:
        c.update(hist.get(p, {}))
    top = [k for k, _ in c.most_common(2)] or [major]
    hit1 += top[0] == gname
    hit2 += gname in top
res["C_baseline_valid"] = {"n": len(va), "top1_acc": round(hit1 / len(va), 4), "top2_acc": round(hit2 / len(va), 4),
                           "majority_genre": major}
log("  ", res["C_baseline_valid"])

log("\n확인 3. 복합 공연 예측에서 둘째 장르의 확률 (A 모델, 원래 방식)")
pc = pA[comp_nodes].numpy()
srt = -np.sort(-pc, axis=1)
p1, p2 = srt[:, 0], srt[:, 1]
res["composite_second_prob"] = {
    "n": int(len(pc)),
    "median_p2": round(float(np.median(p2)), 4),
    **{f"p2_ge_{t}": int((p2 >= t).sum()) for t in (0.5, 0.3, 0.1)},
    "p1_lt_0.5": int((p1 < 0.5).sum()),
    "no_participants": int(sum(1 for d_ in deg[len(df):] if d_ == 0)),
}
log("  ", res["composite_second_prob"])

# 저장된 최종 예측과 이번 재현이 얼마나 같은가
old = pd.read_csv(ROOT / "results/classification/complex_predictions_final.csv", encoding="utf-8-sig")
top2 = np.argsort(-pc, axis=1)[:, :2]
new1 = [genres[i] for i in top2[:, 0]]
new_pair = [frozenset((genres[a], genres[b])) for a, b in top2]
old_pair = [frozenset((a, b)) for a, b in zip(old["장르1"], old["장르2"])]
res["agreement_with_saved_prediction"] = {
    "same_first_genre": round(float(np.mean([a == b for a, b in zip(new1, old["장르1"])])), 4),
    "same_pair": round(float(np.mean([a == b for a, b in zip(new_pair, old_pair)])), 4),
}
log("   저장된 예측과의 일치", res["agreement_with_saved_prediction"])

(OUT / "01_model_holdout.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
log("\n저장:", OUT / "01_model_holdout.json")
