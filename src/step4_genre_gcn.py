"""보고서 결과물 ④ GNN 기반 복합 장르 세분화

공연, 인물, 단체를 점으로, 참여를 선으로 하는 그래프를 만든다.
장르를 아는 단일 공연으로 배우고, 복합 공연에는 확률이 높은 장르 둘을 조합으로 붙인다.

두 가지 방식으로 실행한다.

  as_run     2025년 최종 노트북과 같다. 난수 씨앗만 고정했다.
  improved   같은 모델에서 아래만 고쳤다.
               - 이름 끝의 '등·외' 를 제대로 뗀다
               - 장르별 가중치의 분모를 학습 표본 수로 바꾼다
               - 쓰이지 않던 장르 노드를 만들지 않는다
               - 선의 굵기를 매길 때 이름을 찾지 못해 0 이 되던 것을 고친다

방식마다 세 가지를 잰다.

  1. 학습에 쓴 공연으로 평가 (2025년에 보고한 방식)
  2. 단일 공연의 20% 를 떼어 평가 (처음 보는 공연에 대한 성능)
  3. 씨앗을 바꿔 세 번 학습했을 때 복합 공연의 조합이 얼마나 같은가

실행: python src/step4_genre_gcn.py [as_run|improved]
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, hamming_loss
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, to_undirected

from common import (OUT, SAVED, SEED, Check, load_performances, log, norm_genre, pair_key, read_csv, split_names,
                    split_names_2025_model, write_csv)

EPOCHS, LR, DIM = 100, 0.01, 64
SEEDS = [42, 7, 2025]


def build_graph(variant):
    single, comp = load_performances()
    clean = split_names_2025_model if variant == "as_run" else split_names
    for d in (single, comp):
        d["cast"] = d["출연진내용"].map(clean)
        d["crew"] = d["제작진내용"].map(clean)
        d["org"] = d["기획제작사명"].map(clean)
    single["source"], comp["source"] = "df", "df_complex"
    single["oi"], comp["oi"] = single.index, comp.index
    allp = pd.concat([single, comp], ignore_index=True)

    node = {}
    for _, r in allp.iterrows():
        for n in r["cast"] + r["crew"]:
            node.setdefault(f"인물_{n}", len(node))
    for _, r in allp.iterrows():
        for n in r["org"]:
            node.setdefault(f"제작사_{n}", len(node))
    for _, r in allp.iterrows():
        node.setdefault(f"공연_{r['source']}_{r['oi']}", len(node))
    genres = sorted(single["장르명"].map(norm_genre).unique())
    if variant == "as_run":                       # 2025년 코드는 장르 노드를 만들고 잇지 않았다
        for g in genres:
            node.setdefault(f"장르_{g}", len(node))

    if variant == "as_run":
        # 2025년 코드: 이름을 공백으로 이어 붙여 다시 나눈다. 소문자로 바꾸므로 대문자가 든 이름은 찾지 못한다
        corpus = [" ".join(r["cast"] + r["crew"] + r["org"]) for _, r in allp.iterrows()]
        vec = TfidfVectorizer(token_pattern=r"[^ ]+")
    else:
        # 이름 목록을 그대로 넘긴다. 이름 안의 공백과 대문자가 보존된다
        corpus = [r["cast"] + r["crew"] + r["org"] for _, r in allp.iterrows()]
        vec = TfidfVectorizer(analyzer=lambda names: names)
    tfidf = vec.fit_transform(corpus)
    edges, weights, degree = [], [], []
    for i, r in allp.iterrows():
        pid = node[f"공연_{r['source']}_{r['oi']}"]
        parts = r["cast"] + r["crew"] + r["org"]
        degree.append(len(parts))
        for n in parts:
            key = f"제작사_{n}" if n in r["org"] else f"인물_{n}"
            edges.append([pid, node[key]])
            col = vec.vocabulary_.get(n)
            weights.append(float(tfidf[i, col]) if col is not None else 0.0)

    N, G = len(node), len(genres)
    ei = torch.tensor(np.array(edges).T, dtype=torch.long)
    ew = torch.tensor(weights, dtype=torch.float)
    ei, ew = to_undirected(ei, ew, num_nodes=N)
    ei, ew = add_self_loops(ei, ew, fill_value=1.0, num_nodes=N)

    gid = {g: i for i, g in enumerate(genres)}
    y = torch.zeros((N, G))
    single_nodes = []
    for _, r in single.iterrows():
        nid = node[f"공연_df_{r['oi']}"]
        y[nid, gid[norm_genre(r["장르명"])]] = 1
        single_nodes.append(nid)
    comp_nodes = [node[f"공연_df_complex_{i}"] for i in comp["oi"]]
    info = {"nodes": N, "edges": len(edges), "genres": genres,
            "people": sum(k.startswith("인물_") for k in node), "orgs": sum(k.startswith("제작사_") for k in node),
            "zero_weight_edges": int(sum(w == 0.0 for w in weights)),
            "single_without_participants": int(sum(d == 0 for d in degree[:len(single)])),
            "composite_without_participants": int(sum(d == 0 for d in degree[len(single):]))}
    return {"N": N, "G": G, "ei": ei, "ew": ew, "y": y, "single": torch.tensor(single_nodes),
            "comp": torch.tensor(comp_nodes), "comp_df": comp, "genres": genres, "info": info}


class GenreGCN(nn.Module):
    def __init__(self, n, g):
        super().__init__()
        self.emb = nn.Embedding(n, DIM)
        self.c1 = GCNConv(DIM, DIM)
        self.c2 = GCNConv(DIM, g)

    def forward(self, ei, ew):
        x = self.c1(self.emb.weight, ei, ew).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.c2(x, ei, ew)


def train(g, idx, seed, variant, label):
    torch.manual_seed(seed)
    counts = g["y"][idx].sum(0).numpy()
    total = len(g["y"]) if variant == "as_run" else len(idx)
    pw = torch.tensor((total - counts) / (counts + 1e-6), dtype=torch.float)
    model = GenreGCN(g["N"], g["G"])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = crit(model(g["ei"], g["ew"])[idx], g["y"][idx])
        loss.backward()
        opt.step()
        if (ep + 1) % 25 == 0:
            log(f"      [{label}] {ep + 1}/{EPOCHS} 손실 {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(g["ei"], g["ew"]))


def metrics(g, probs, idx):
    yt, p = g["y"][idx].numpy(), probs[idx].numpy()
    yb = (p > 0.5).astype(int)
    true, order = yt.argmax(1), np.argsort(-p, axis=1)
    return {"표본": int(len(idx)),
            "마이크로 F1": round(float(f1_score(yt, yb, average="micro", zero_division=0)), 4),
            "매크로 F1": round(float(f1_score(yt, yb, average="macro", zero_division=0)), 4),
            "해밍 정확도": round(float(1 - hamming_loss(yt, yb)), 4),
            "정확도": round(float((order[:, 0] == true).mean()), 4),
            "상위 2개 정확도": round(float(((order[:, 0] == true) | (order[:, 1] == true)).mean()), 4)}


def predict(g, probs):
    p = probs[g["comp"]].numpy()
    order = np.argsort(-p, axis=1)[:, :2]
    names = g["genres"]
    d = pd.DataFrame({"공연명": g["comp_df"]["공연명"].to_numpy(),
                      "장르1": [names[i] for i in order[:, 0]], "확률1": p[np.arange(len(p)), order[:, 0]],
                      "장르2": [names[i] for i in order[:, 1]], "확률2": p[np.arange(len(p)), order[:, 1]]})
    d["예측복합장르"] = d["장르1"] + "+" + d["장르2"]
    d["조합"] = [pair_key(a, b) for a, b in zip(d["장르1"], d["장르2"])]
    return d


def run(variant):
    log(f"\n  [{variant}] 그래프 구성")
    g = build_graph(variant)
    info = g["info"]
    log(f"    점 {info['nodes']:,} (인물 {info['people']:,} · 단체 {info['orgs']:,}) · 선 {info['edges']:,} · "
        f"참여자 없는 공연: 단일 {info['single_without_participants']:,} 복합 {info['composite_without_participants']:,}")
    out = OUT / "step4" / variant
    res = {"graph": info}

    log("    1) 학습에 쓴 공연으로 평가")
    preds, full = [], {}
    for seed in SEEDS:
        probs = train(g, g["single"], seed, variant, f"전체 학습 씨앗 {seed}")
        full[seed] = metrics(g, probs, g["single"])
        d = predict(g, probs)
        d["씨앗"] = seed
        preds.append(d)
        write_csv(d, out / f"predictions_seed{seed}.csv")
    res["train_on_all"] = full
    log(f"      {full[SEEDS[0]]}")

    log("    2) 단일 공연의 20% 를 떼어 평가")
    gen = torch.Generator().manual_seed(SEED)
    perm = g["single"][torch.randperm(len(g["single"]), generator=gen)]
    cut = int(0.8 * len(perm))
    probs = train(g, perm[:cut], SEED, variant, "검증 분리")
    res["holdout_train"] = metrics(g, probs, perm[:cut])
    res["holdout_valid"] = metrics(g, probs, perm[cut:])
    log(f"      학습 구간 {res['holdout_train']}")
    log(f"      검증 구간 {res['holdout_valid']}")
    yt = g["y"][perm[cut:]].numpy().argmax(1)
    pr = probs[perm[cut:]].numpy().argmax(1)
    per = [{"장르": n, "검증 표본": int((yt == i).sum()),
            "정확도": round(float((pr[yt == i] == i).mean()), 4) if (yt == i).any() else None}
           for i, n in enumerate(g["genres"])]
    write_csv(pd.DataFrame(per), out / "holdout_by_genre.csv")

    log("    3) 씨앗을 바꿨을 때 조합이 같은 비율")
    stab = []
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            a, b = preds[i], preds[j]
            stab.append({"씨앗 A": SEEDS[i], "씨앗 B": SEEDS[j],
                         "첫째 장르 일치": round(float((a["장르1"] == b["장르1"]).mean()), 4),
                         "조합 일치": round(float((a["조합"] == b["조합"]).mean()), 4)})
    sd = pd.DataFrame(stab)
    write_csv(sd, out / "stability.csv")
    res["stability_pair_mean"] = round(float(sd["조합 일치"].mean()), 4)
    log(f"      조합 일치 평균 {res['stability_pair_mean']}")

    first = preds[0]
    res["second_genre"] = {"확률 0.5 이상": int((first["확률2"] >= 0.5).sum()), "확률 0.3 이상": int((first["확률2"] >= 0.3).sum()),
                           "확률 0.1 이상": int((first["확률2"] >= 0.1).sum()), "중앙값": round(float(first["확률2"].median()), 4)}
    old = read_csv(SAVED / "classification" / "complex_predictions_final.csv")
    old_pair = [pair_key(a, b) for a, b in zip(old["장르1"], old["장르2"])]
    res["agreement_with_2025"] = {"첫째 장르 일치": round(float((first["장르1"].to_numpy() == old["장르1"].to_numpy()).mean()), 4),
                                  "조합 일치": round(float(np.mean(np.array(old_pair) == first["조합"].to_numpy())), 4)}
    log(f"      2025년 저장 예측과 조합 일치 {res['agreement_with_2025']['조합 일치']}")

    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


def main():
    log("④ 복합 장르 세분화")
    variants = sys.argv[1:] or ["as_run", "improved"]
    c = Check("step4_" + "_".join(variants))
    for v in variants:
        r = run(v)
        if v == "as_run":
            c.add("점의 수", r["graph"]["nodes"], 142215, "보고서")
            c.add("학습 데이터 평가 마이크로 F1", r["train_on_all"][SEEDS[0]]["마이크로 F1"], 0.9925, "발표 자료", tol=0.002)
            c.add("학습 데이터 평가 매크로 F1", r["train_on_all"][SEEDS[0]]["매크로 F1"], 0.9474, "발표 자료", tol=0.005)
            c.add("검증 구간 정확도", r["holdout_valid"]["정확도"], 0.9925, "발표 자료 (학습 데이터 기준 수치)", tol=0.002)
    c.save()


if __name__ == "__main__":
    main()
