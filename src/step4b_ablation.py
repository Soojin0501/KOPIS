"""분류 모델 2판에서 3판까지, 바뀐 것을 하나씩 넣으며 효과를 잰다.

2025년에는 모델을 바꾸면서 평가 방법도 함께 바꿨다. 2판의 0.66 과 3판의 수치를 바로 비교할 수 없었다.
여기서는 평가 방법을 고정하고 변경을 하나씩 더한다.

더하는 순서
  0. 2판 그대로: 출연진과 제작진을 다른 점으로, 단체 없음, 선의 굵기 없음, 장르 하나 고르기, 200회
  1. 출연진과 제작진을 한 사람으로 합침
  2. 단체를 점으로 추가
  3. 선에 굵기(TF-IDF) 추가
  4. 장르마다 확률을 내는 방식과 장르별 가중치
  5. 학습 100회  (= 3판)

평가 방법 둘
  끊음  학습할 때 검증 공연의 선을 그래프에서 끊는다 (2판의 방식)
  둠    학습할 때 검증 공연의 선을 그대로 둔다. 정답만 숨긴다 (복합 공연의 처지와 같다)

실행: python src/step4b_ablation.py
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, to_undirected

from common import OUT, Check, load_performances, log, norm_genre, split_names_2025_model, write_csv

SEEDS = [42, 7]
STEPS = [
    ("0. 2판 그대로", dict(merge=False, orgs=False, weights=False, multilabel=False, epochs=200)),
    ("1. + 출연·제작을 한 사람으로", dict(merge=True, orgs=False, weights=False, multilabel=False, epochs=200)),
    ("2. + 단체 추가", dict(merge=True, orgs=True, weights=False, multilabel=False, epochs=200)),
    ("3. + 선의 굵기", dict(merge=True, orgs=True, weights=True, multilabel=False, epochs=200)),
    ("4. + 장르별 확률과 가중치", dict(merge=True, orgs=True, weights=True, multilabel=True, epochs=200)),
    ("5. 학습 100회 (3판)", dict(merge=True, orgs=True, weights=True, multilabel=True, epochs=100)),
]


def build(cfg, single, comp):
    allp = pd.concat([single, comp], ignore_index=True)
    node = {}

    def key(role, name):
        return f"인물_{name}" if cfg["merge"] else f"{role}_{name}"

    for _, r in allp.iterrows():
        for n in r["cast"]:
            node.setdefault(key("출연진", n), len(node))
        for n in r["crew"]:
            node.setdefault(key("제작진", n), len(node))
    if cfg["orgs"]:
        for _, r in allp.iterrows():
            for n in r["org"]:
                node.setdefault(f"제작사_{n}", len(node))
    for _, r in allp.iterrows():
        node.setdefault(f"공연_{r['source']}_{r['oi']}", len(node))
    genres = sorted(single["장르명"].map(norm_genre).unique())
    for g in genres:
        node.setdefault(f"장르_{g}", len(node))

    tfidf = vec = None
    if cfg["weights"]:
        corpus = [" ".join(r["cast"] + r["crew"] + (r["org"] if cfg["orgs"] else [])) for _, r in allp.iterrows()]
        vec = TfidfVectorizer(token_pattern=r"[^ ]+")
        tfidf = vec.fit_transform(corpus)

    edges, weights = [], []
    for i, r in allp.iterrows():
        pid = node[f"공연_{r['source']}_{r['oi']}"]
        parts = [(key("출연진", n), n) for n in r["cast"]] + [(key("제작진", n), n) for n in r["crew"]]
        if cfg["orgs"]:
            parts += [(f"제작사_{n}", n) for n in r["org"]]
        for k, n in parts:
            edges.append([pid, node[k]])
            if cfg["weights"]:
                col = vec.vocabulary_.get(n)
                weights.append(float(tfidf[i, col]) if col is not None else 0.0)
            else:
                weights.append(1.0)

    N = len(node)
    gid = {g: i for i, g in enumerate(genres)}
    labels = torch.full((N,), -1, dtype=torch.long)
    perf = []
    for _, r in single.iterrows():
        nid = node[f"공연_df_{r['oi']}"]
        labels[nid] = gid[norm_genre(r["장르명"])]
        perf.append(nid)
    return {"N": N, "G": len(genres), "edges": np.array(edges), "w": np.array(weights, dtype=np.float32),
            "labels": labels, "perf": np.array(perf)}


def graph_tensors(edges, w, n, use_w):
    ei = torch.tensor(edges.T, dtype=torch.long)
    ew = torch.tensor(w, dtype=torch.float)
    ei, ew = to_undirected(ei, ew, num_nodes=n)
    ei, ew = add_self_loops(ei, ew, fill_value=1.0, num_nodes=n)
    return ei, (ew if use_w else None)


class GCN(nn.Module):
    def __init__(self, n, g):
        super().__init__()
        self.emb = nn.Embedding(n, 64)
        self.c1 = GCNConv(64, 64)
        self.c2 = GCNConv(64, g)

    def forward(self, ei, ew):
        x = self.c1(self.emb.weight, ei, ew).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.c2(x, ei, ew)


def run_one(g, cfg, protocol, seed):
    tr, va = train_test_split(g["perf"], test_size=0.2, random_state=42)
    full_ei, full_ew = graph_tensors(g["edges"], g["w"], g["N"], cfg["weights"])
    if protocol == "끊음":
        vset = set(va.tolist())
        keep = np.array([a not in vset and b not in vset for a, b in g["edges"]])
        tr_ei, tr_ew = graph_tensors(g["edges"][keep], g["w"][keep], g["N"], cfg["weights"])
    else:
        tr_ei, tr_ew = full_ei, full_ew

    torch.manual_seed(seed)
    model = GCN(g["N"], g["G"])
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    tr_t, va_t = torch.tensor(tr), torch.tensor(va)
    y = g["labels"]
    if cfg["multilabel"]:
        onehot = torch.zeros((g["N"], g["G"]))
        onehot[tr_t, y[tr_t]] = 1
        counts = onehot[tr_t].sum(0).numpy()
        pw = torch.tensor((g["N"] - counts) / (counts + 1e-6), dtype=torch.float)   # 3판과 같게 둔다
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(cfg["epochs"]):
        model.train()
        opt.zero_grad()
        out = model(tr_ei, tr_ew)
        if cfg["multilabel"]:
            loss = crit(out[tr_t], onehot[tr_t])
        else:
            loss = F.nll_loss(F.log_softmax(out[tr_t], dim=1), y[tr_t])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(full_ei, full_ew)[va_t]
    true = y[va_t].numpy()
    order = np.argsort(-out.numpy(), axis=1)
    return {"정확도": float((order[:, 0] == true).mean()),
            "상위 2개 정확도": float(((order[:, 0] == true) | (order[:, 1] == true)).mean()),
            "매크로 F1": float(f1_score(true, order[:, 0], average="macro", zero_division=0))}


def main():
    log("④-2 분류 모델의 변경별 효과")
    single, comp = load_performances()
    for d in (single, comp):
        d["cast"] = d["출연진내용"].map(split_names_2025_model)
        d["crew"] = d["제작진내용"].map(split_names_2025_model)
        d["org"] = d["기획제작사명"].map(split_names_2025_model)
    single["source"], comp["source"] = "df", "df_complex"
    single["oi"], comp["oi"] = single.index, comp.index

    rows = []
    for name, cfg in STEPS:
        g = build(cfg, single, comp)
        log(f"\n  {name} · 점 {g['N']:,} · 선 {len(g['edges']):,}")
        for protocol in ("끊음", "둠"):
            rs = [run_one(g, cfg, protocol, s) for s in SEEDS]
            row = {"단계": name, "평가": protocol, "점": g["N"], "선": len(g["edges"])}
            for k in ("정확도", "상위 2개 정확도", "매크로 F1"):
                vals = [r[k] for r in rs]
                row[k] = round(float(np.mean(vals)), 4)
                row[k + " 범위"] = f"{min(vals):.4f}~{max(vals):.4f}"
            rows.append(row)
            log(f"    [{protocol}] 정확도 {row['정확도']:.4f} ({row['정확도 범위']}) · 상위 2개 {row['상위 2개 정확도']:.4f} · 매크로 F1 {row['매크로 F1']:.4f}")
    res = pd.DataFrame(rows)
    write_csv(res, OUT / "step4b" / "ablation.csv")

    piv = res.pivot(index="단계", columns="평가", values="정확도")
    piv["평가 방법의 차이"] = (piv["둠"] - piv["끊음"]).round(4)
    piv["앞 단계 대비 (둠)"] = piv["둠"].diff().round(4)
    piv["앞 단계 대비 (끊음)"] = piv["끊음"].diff().round(4)
    write_csv(piv.reset_index(), OUT / "step4b" / "ablation_accuracy.csv")
    log("\n" + piv.to_string())

    c = Check("step4b")
    c.add("2판 그대로, 선을 끊고 평가한 정확도", float(piv.loc[STEPS[0][0], "끊음"]), 0.6621, "2판 노트북", tol=0.02)
    c.add("2판 그대로, 상위 2개 정확도", float(res[(res["단계"] == STEPS[0][0]) & (res["평가"] == "끊음")]["상위 2개 정확도"].iloc[0]), 0.7690, "2판 노트북", tol=0.02)
    c.add("2판 점의 수", int(res[res["단계"] == STEPS[0][0]]["점"].iloc[0]), 138935, "2판 노트북")
    c.add("3판 점의 수", int(res[res["단계"] == STEPS[-1][0]]["점"].iloc[0]), 142215, "3판 노트북")
    c.save()


if __name__ == "__main__":
    main()
