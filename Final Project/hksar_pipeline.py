"""
hksar_network_pipeline.py
=========================

Mapping Word Use: Network Analysis of HKSAR Government Communications
across an Extended Political Crisis (2019-2020)

Kei Hou and Lesley Zhang  -  SoDA 501

Pipeline:
    1. Preprocess Chinese (jieba) and English (regex) press-release corpora
    2. Build daily 30-day sliding-window co-occurrence networks
    3. Compute density / modularity / degree centralization time series
    4. PELT change-point detection (per metric, per language)
    5. Build phase-level PPMI-weighted co-occurrence networks
    6. Louvain community detection + Jaccard lineage across phases
    7. Track focal-term degree / betweenness / top neighbors across phases
    8. Cross-linguistic correlation of metric trajectories within phases
    9. Plots: fig1 (metric series), fig2 (focal centrality),
              fig3 (protest ego networks), fig4 (cross-ling heatmap)

Inputs (place under DATA_DIR):
    press_releases_zh.csv   (cols: date [YYYY-MM-DD], text; optional: title)
    press_releases_en.csv   (cols: date [YYYY-MM-DD], text; optional: title)
    stopwords_zh.txt        (one token per line; optional)
    stopwords_en.txt        (one token per line; optional)

Dependencies:
    pip install pandas numpy scipy scikit-learn networkx python-louvain \
                ruptures jieba matplotlib tqdm
"""

from __future__ import annotations

import os, re, json, math, pickle
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain          # python-louvain
import ruptures as rpt
import jieba
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================ CONFIG ============================
DATA_DIR      = Path("./data")
OUT_DIR       = Path("./output")
FIG_DIR       = OUT_DIR / "figures"
PHASE_NET_DIR = OUT_DIR / "phase_networks"
for d in (OUT_DIR, FIG_DIR, PHASE_NET_DIR):
    d.mkdir(parents=True, exist_ok=True)

START_DATE, END_DATE = "2019-06-09", "2020-09-30"

WINDOW_DAYS       = 30      # sensitivity: 15 / 30 / 60
STRIDE_DAYS       = 2
DAILY_TOP_TOKENS  = 4000
DAILY_MIN_EDGE    = 3

PELT_KERNEL       = "rbf"
PELT_PEN_MULT     = 1.5     # sensitivity: 1.0 / 1.5 / 2.0

PHASE_TOP_TOKENS  = 1500
PHASE_MIN_COUNT   = 5
PPMI_THRESHOLD    = 1.0     # sensitivity: 0.5 / 1.0 / 1.5

FOCAL_ZH = ["示威者", "暴徒", "警察", "對話", "市民", "政府"]
FOCAL_EN = ["protesters", "rioters", "police", "dialogue", "citizens", "government"]

# Subset for fig3 (protest ego networks)
PROTEST_FOCAL_ZH = ["示威者", "暴徒", "對話"]
PROTEST_FOCAL_EN = ["protesters", "rioters", "dialogue"]

ZH_TOKEN_RE = re.compile(r"^[\u4e00-\u9fff]{2,}$")
EN_TOKEN_RE = re.compile(r"^[a-z]{3,}$")

RANDOM_SEED = 42


# ====================== 1. PREPROCESSING ========================
def load_stopwords(path):
    if not path.exists(): return set()
    with open(path, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip() and not l.startswith("#")}

def tokenize_zh(text, stopwords):
    text = re.sub(r"<[^>]+>", " ", str(text))
    return [t for t in jieba.cut(text, cut_all=False)
            if ZH_TOKEN_RE.match(t) and t not in stopwords]

def tokenize_en(text, stopwords):
    text = re.sub(r"<[^>]+>", " ", str(text)).lower()
    return [t for t in re.findall(r"[a-z]+", text)
            if EN_TOKEN_RE.match(t) and t not in stopwords]

def split_sentences_zh(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    return [p.strip() for p in re.split(r"[。！？；\n]+", text) if len(p.strip()) > 1]

def split_sentences_en(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    return [p.strip() for p in re.split(r"(?<=[\.\!\?])\s+", text) if len(p.strip()) > 1]

def preprocess_corpus(csv_path, lang, stopwords):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
    # Concatenate title + text if title column exists
    if "title" in df.columns:
        joiner = "。" if lang == "zh" else ". "
        df["text"] = df["title"].fillna("") + joiner + df["text"].fillna("")
    if lang == "zh":
        tok_fn, sent_fn = tokenize_zh, split_sentences_zh
    else:
        tok_fn, sent_fn = tokenize_en, split_sentences_en
    df["tokens"]    = df["text"].apply(lambda t: tok_fn(t, stopwords))
    df["sentences"] = df["text"].apply(
        lambda t: [tok_fn(s, stopwords) for s in sent_fn(t)])
    return df[df["tokens"].apply(len) >= 1].reset_index(drop=True)


# ============ 2. DAILY NETWORKS + GLOBAL METRICS ================
def build_cooc_graph(sentences, vocab_cap, min_edge):
    """Sentence-level co-occurrence: same sentence = co-occurring."""
    tok_counts = Counter()
    for s in sentences: tok_counts.update(set(s))
    vocab = {w for w, _ in tok_counts.most_common(vocab_cap)}
    edge_counts = Counter()
    for s in sentences:
        kept = [t for t in set(s) if t in vocab]
        for a, b in combinations(sorted(kept), 2):
            edge_counts[(a, b)] += 1
    G = nx.Graph()
    for (a, b), c in edge_counts.items():
        if c >= min_edge: G.add_edge(a, b, weight=c)
    return G

def graph_metrics(G):
    if G.number_of_nodes() < 5 or G.number_of_edges() < 5:
        return {"density": np.nan, "modularity": np.nan, "centralization": np.nan,
                "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}
    density = nx.density(G)
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        part = community_louvain.best_partition(lcc, weight="weight",
                                                random_state=RANDOM_SEED)
        mod = community_louvain.modularity(part, lcc, weight="weight")
    except Exception:
        mod = np.nan
    n = G.number_of_nodes()
    degs = np.array([d for _, d in G.degree()])
    cent = ((degs.max() - degs).sum() / ((n - 1) * (n - 2))
            if n > 2 and degs.max() > 0 else np.nan)
    return {"density": density, "modularity": mod, "centralization": cent,
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}

def daily_metric_series(df, lang, window_days=WINDOW_DAYS, stride_days=STRIDE_DAYS):
    dates = pd.date_range(pd.to_datetime(START_DATE) + pd.Timedelta(days=window_days - 1),
                          pd.to_datetime(END_DATE), freq=f"{stride_days}D")
    rows = []
    for d in tqdm(dates, desc=f"daily metrics [{lang}]"):
        win_start = d - pd.Timedelta(days=window_days - 1)
        sl = df[(df["date"] >= win_start) & (df["date"] <= d)]
        sents = [s for sl_ in sl["sentences"] for s in sl_ if len(s) >= 2]
        if not sents:
            rows.append({"date": d, "density": np.nan, "modularity": np.nan,
                         "centralization": np.nan, "n_nodes": 0, "n_edges": 0,
                         "n_statements": len(sl)})
            continue
        G = build_cooc_graph(sents, DAILY_TOP_TOKENS, DAILY_MIN_EDGE)
        m = graph_metrics(G); m.update({"date": d, "n_statements": len(sl)})
        rows.append(m)
    return pd.DataFrame(rows)


# ============== 3. PELT CHANGE-POINT DETECTION ==================
def detect_breakpoints(series, pen_mult=PELT_PEN_MULT):
    """PELT with RBF kernel; BIC-style penalty: pen_mult * log(n) * sigma^2."""
    s = series[~np.isnan(series)]
    if len(s) < 20: return []
    pen = pen_mult * math.log(len(s)) * float(np.var(s))
    algo = rpt.Pelt(model=PELT_KERNEL).fit(s.reshape(-1, 1))
    return algo.predict(pen=pen)[:-1]

def detect_breakpoints_all(metrics_df, lang, pen_mults=(1.0, 1.5, 2.0)):
    out = []
    for metric in ["density", "modularity", "centralization"]:
        valid = metrics_df.dropna(subset=[metric]).reset_index(drop=True)
        s = valid[metric].values
        for pm in pen_mults:
            for i in detect_breakpoints(s, pen_mult=pm):
                if 0 <= i < len(valid):
                    out.append({"lang": lang, "metric": metric, "pen_mult": pm,
                                "bkp_idx": i, "bkp_date": valid.loc[i, "date"]})
    return pd.DataFrame(out)


# ================ 4. PHASE PPMI NETWORKS ========================
def build_ppmi_network(df_phase, top_tokens=PHASE_TOP_TOKENS,
                       min_count=PHASE_MIN_COUNT, ppmi_thresh=PPMI_THRESHOLD):
    sents = [s for sl in df_phase["sentences"] for s in sl if len(s) >= 2]
    tok_counts = Counter()
    for s in sents: tok_counts.update(set(s))
    vocab = {w for w, _ in tok_counts.most_common(top_tokens)}

    pair_counts, word_counts = Counter(), Counter()
    n_sents = 0
    for s in sents:
        kept = sorted({t for t in s if t in vocab})
        if len(kept) < 2: continue
        n_sents += 1
        for w in kept: word_counts[w] += 1
        for a, b in combinations(kept, 2): pair_counts[(a, b)] += 1

    G = nx.Graph()
    if n_sents == 0: return G
    log_n = math.log(n_sents)
    for (a, b), c in pair_counts.items():
        if c < min_count: continue
        # PPMI = max(0, log(P(a,b) / (P(a)P(b))))
        pmi = math.log(c) + log_n - math.log(word_counts[a]) - math.log(word_counts[b])
        ppmi = max(pmi, 0.0)
        if ppmi >= ppmi_thresh:
            G.add_edge(a, b, weight=ppmi, count=c)
    return G

def phase_summary(G):
    if G.number_of_nodes() == 0:
        return {"density": np.nan, "modularity": np.nan, "centralization": np.nan,
                "n_communities": 0, "lcc_frac": np.nan, "n_nodes": 0, "n_edges": 0}
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    part = community_louvain.best_partition(lcc, weight="weight",
                                            random_state=RANDOM_SEED)
    n = G.number_of_nodes()
    degs = np.array([d for _, d in G.degree()])
    cent = (degs.max() - degs).sum() / ((n - 1) * (n - 2)) if n > 2 else np.nan
    return {"density": nx.density(G),
            "modularity": community_louvain.modularity(part, lcc, weight="weight"),
            "centralization": cent,
            "n_communities": len(set(part.values())),
            "lcc_frac": lcc.number_of_nodes() / n,
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}


# =========== 5. COMMUNITY DETECTION + LINEAGE ===================
def label_communities(G, top_n=8):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if not comps: return {}
    lcc = G.subgraph(comps[0]).copy()
    part = community_louvain.best_partition(lcc, weight="weight",
                                            random_state=RANDOM_SEED)
    by_comm = defaultdict(list)
    for node, c in part.items(): by_comm[c].append(node)
    return {c: sorted(nodes,
                      key=lambda n: lcc.degree(n, weight="weight"),
                      reverse=True)[:top_n]
            for c, nodes in by_comm.items()}

def jaccard_lineage(comm_a, comm_b):
    rows = []
    for ca, ta in comm_a.items():
        sa = set(ta)
        for cb, tb in comm_b.items():
            sb = set(tb)
            j = len(sa & sb) / len(sa | sb) if sa | sb else 0.0
            rows.append({"comm_a": ca, "comm_b": cb, "jaccard": j,
                         "top_a": "|".join(ta), "top_b": "|".join(tb)})
    return pd.DataFrame(rows)


# ================ 6. FOCAL-TERM CENTRALITY =====================
def focal_term_metrics(G, term, top_neighbors=10):
    if term not in G:
        return {"in_lcc": False, "degree": np.nan, "weighted_degree": np.nan,
                "betweenness": np.nan, "top_neighbors": ""}
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if term not in lcc:
        return {"in_lcc": False,
                "degree": G.degree(term),
                "weighted_degree": G.degree(term, weight="weight"),
                "betweenness": np.nan, "top_neighbors": ""}
    if lcc.number_of_nodes() > 1500:
        bet = nx.betweenness_centrality(lcc, k=500, seed=RANDOM_SEED).get(term, np.nan)
    else:
        bet = nx.betweenness_centrality(lcc).get(term, np.nan)
    nbrs = sorted(G[term].items(),
                  key=lambda kv: kv[1].get("weight", 0), reverse=True)
    return {"in_lcc": True,
            "degree": G.degree(term),
            "weighted_degree": G.degree(term, weight="weight"),
            "betweenness": bet,
            "top_neighbors": "|".join(n for n, _ in nbrs[:top_neighbors])}


# =========== 7. CROSS-LINGUISTIC COMPARISON =====================
def cross_lang_corr(zh, en, phases):
    from scipy.stats import pearsonr, spearmanr
    merged = zh.merge(en, on="date", suffixes=("_zh", "_en"))
    rows = []
    for pname, (start, end) in phases.items():
        sub = merged[(merged["date"] >= start) & (merged["date"] <= end)]
        for metric in ["density", "modularity", "centralization"]:
            x, y = sub[f"{metric}_zh"].values, sub[f"{metric}_en"].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 5:
                rows.append({"phase": pname, "metric": metric,
                             "pearson_r": np.nan, "spearman_r": np.nan,
                             "n": int(mask.sum())})
                continue
            r_p, _ = pearsonr(x[mask], y[mask])
            r_s, _ = spearmanr(x[mask], y[mask])
            rows.append({"phase": pname, "metric": metric,
                         "pearson_r": r_p, "spearman_r": r_s,
                         "n": int(mask.sum())})
    return pd.DataFrame(rows)


# =========================== 8. PLOTS ===========================
def plot_metric_series(zh, en, phases, out_path):
    """Fig 1: daily metrics, Chinese vs English, with phase shading."""
    metrics = ["density", "modularity", "centralization"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    colors = ["#cdeac0", "#fce5cd", "#cfe2f3", "#f4cccc", "#d9d2e9"]
    for ax, m in zip(axes, metrics):
        for (pname, (s, e)), col in zip(phases.items(), colors):
            ax.axvspan(pd.to_datetime(s), pd.to_datetime(e), color=col, alpha=0.4)
        ax.plot(zh["date"], zh[m], label="Chinese", color="#9c1f1f", lw=1)
        ax.plot(en["date"], en[m], label="English",  color="#1f4e79", lw=1)
        ax.set_ylabel(m); ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("date")
    fig.suptitle("Daily 30-day rolling-window network metrics", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_focal_centrality(focal_df, out_path):
    """Fig 2: focal-term degree (top) and betweenness (bottom), zh vs en."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, lang in enumerate(["zh", "en"]):
        sub = focal_df[focal_df["lang"] == lang]
        for term in sub["term"].unique():
            tdf = sub[sub["term"] == term].sort_values("phase")
            axes[0, j].plot(tdf["phase"], tdf["degree"], marker="o", label=term)
            axes[1, j].plot(tdf["phase"], tdf["betweenness"], marker="o", label=term)
        axes[0, j].set_title(f"degree centrality - {lang}")
        axes[1, j].set_title(f"betweenness - {lang}")
        axes[0, j].legend(fontsize=7, ncol=2)
        axes[1, j].set_xlabel("phase")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_protest_ego_networks(phase_graphs, lang, focal_terms,
                              phases, out_path, top_neighbors=12):
    """Fig 3: induced subgraph on protest focal terms + top PPMI neighbors per phase."""
    n_phases = len(phases)
    fig, axes = plt.subplots(1, n_phases, figsize=(4 * n_phases, 4.5))
    if n_phases == 1: axes = [axes]
    for ax, (pname, _) in zip(axes, phases.items()):
        G = phase_graphs[(lang, pname)]
        present = [t for t in focal_terms if t in G]
        if not present:
            ax.set_title(f"{pname} (no focal terms in graph)")
            ax.axis("off"); continue
        nodes = set(present)
        for term in present:
            nbrs = sorted(G[term].items(),
                          key=lambda kv: kv[1].get("weight", 0),
                          reverse=True)[:top_neighbors]
            nodes.update(n for n, _ in nbrs)
        H = G.subgraph(nodes).copy()
        # Mark nodes connected to multiple focal terms
        bridge = {n for n in H.nodes()
                  if n not in present
                  and sum(1 for t in present if H.has_edge(n, t)) >= 2}
        node_colors = []
        node_sizes = []
        for n in H.nodes():
            if n in present:
                node_colors.append("#c0392b")
                node_sizes.append(800)
            elif n in bridge:
                node_colors.append("#8e44ad")
                node_sizes.append(400)
            else:
                node_colors.append("#bdc3c7")
                node_sizes.append(200 + 30 * H.degree(n))
        pos = nx.spring_layout(H, seed=RANDOM_SEED, k=0.6)
        edge_widths = [0.3 + 0.6 * H[u][v].get("weight", 1) for u, v in H.edges()]
        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.3, width=edge_widths)
        nx.draw_networkx_nodes(H, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.85)
        nx.draw_networkx_labels(H, pos, ax=ax, font_size=7,
                                font_family="Arial Unicode MS")
        ax.set_title(pname); ax.axis("off")
    fig.suptitle(f"Protest-focal ego networks ({lang})", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_cross_ling_heatmap(corr_df, out_path):
    """Fig 4: phase x metric heatmap of Pearson r between zh and en."""
    pivot = corr_df.pivot(index="metric", columns="phase", values="pearson_r")
    pivot = pivot.reindex(["density", "modularity", "centralization"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if abs(v) < 0.5 else "white", fontsize=10)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Cross-linguistic correlation of metric trajectories within phases")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================== MAIN ===============================
def main():
    np.random.seed(RANDOM_SEED)

    # ---- Load + preprocess ----
    sw_zh = load_stopwords(DATA_DIR / "stopwords_zh.txt")
    sw_en = load_stopwords(DATA_DIR / "stopwords_en.txt")
    df_zh = preprocess_corpus(DATA_DIR / "press_releases_zh.csv", "zh", sw_zh)
    df_en = preprocess_corpus(DATA_DIR / "press_releases_en.csv", "en", sw_en)
    print(f"[corpus] zh: {len(df_zh)}  |  en: {len(df_en)}")

    # ---- Daily metric series ----
    metrics_zh = daily_metric_series(df_zh, "zh")
    metrics_en = daily_metric_series(df_en, "en")
    metrics_zh.to_csv(OUT_DIR / "daily_metrics_zh.csv", index=False)
    metrics_en.to_csv(OUT_DIR / "daily_metrics_en.csv", index=False)

    # ---- Change-point detection ----
    bkps_zh = detect_breakpoints_all(metrics_zh, "zh")
    bkps_en = detect_breakpoints_all(metrics_en, "en")
    pd.concat([bkps_zh, bkps_en], ignore_index=True).to_csv(
        OUT_DIR / "breakpoints_all.csv", index=False)

    # ---- Phases (paper-reported, from PELT alignment + sanity check) ----
    phases = {
        "P1": ("2019-06-09", "2019-09-07"),
        "P2": ("2019-09-08", "2019-11-15"),
        "P3": ("2019-11-16", "2020-04-24"),
        "P4": ("2020-04-25", "2020-08-01"),
        "P5": ("2020-08-02", "2020-09-30"),
    }
    with open(OUT_DIR / "phases.json", "w", encoding="utf-8") as f:
        json.dump(phases, f, ensure_ascii=False, indent=2)

    # ---- Phase networks ----
    phase_graphs, summaries = {}, []
    for lang, df_lang in [("zh", df_zh), ("en", df_en)]:
        for pname, (start, end) in phases.items():
            sub = df_lang[(df_lang["date"] >= start) & (df_lang["date"] <= end)]
            G = build_ppmi_network(sub)
            phase_graphs[(lang, pname)] = G
            with open(PHASE_NET_DIR / f"{lang}_{pname}.gpickle", "wb") as fh:
                pickle.dump(G, fh)
            s = phase_summary(G)
            s.update({"lang": lang, "phase": pname,
                      "n_statements": len(sub), "start": start, "end": end})
            summaries.append(s)
    pd.DataFrame(summaries).to_csv(OUT_DIR / "phase_global_metrics.csv", index=False)

    # ---- Communities + Jaccard lineage ----
    for lang in ["zh", "en"]:
        comm_by_phase = {p: label_communities(phase_graphs[(lang, p)]) for p in phases}
        rows = []
        ordered = list(phases.keys())
        for i in range(len(ordered) - 1):
            df_jac = jaccard_lineage(comm_by_phase[ordered[i]],
                                     comm_by_phase[ordered[i + 1]])
            df_jac["phase_a"] = ordered[i]; df_jac["phase_b"] = ordered[i + 1]
            rows.append(df_jac)
        pd.concat(rows, ignore_index=True).to_csv(
            OUT_DIR / f"community_lineage_{lang}.csv", index=False)
        with open(OUT_DIR / f"communities_{lang}.json", "w", encoding="utf-8") as f:
            json.dump({p: {str(k): v for k, v in d.items()}
                       for p, d in comm_by_phase.items()},
                      f, ensure_ascii=False, indent=2)

    # ---- Focal-term centrality ----
    focal_rows = []
    for lang, focal_list in [("zh", FOCAL_ZH), ("en", FOCAL_EN)]:
        for pname in phases:
            for term in focal_list:
                m = focal_term_metrics(phase_graphs[(lang, pname)], term)
                m.update({"lang": lang, "phase": pname, "term": term})
                focal_rows.append(m)
    focal_df = pd.DataFrame(focal_rows)
    focal_df.to_csv(OUT_DIR / "focal_term_centrality.csv", index=False)

    # ---- Cross-linguistic correlations ----
    corr_df = cross_lang_corr(metrics_zh, metrics_en, phases)
    corr_df.to_csv(OUT_DIR / "cross_ling_correlations.csv", index=False)

    # ---- Plots ----
    plot_metric_series(metrics_zh, metrics_en, phases, FIG_DIR / "fig1_metrics.png")
    plot_focal_centrality(focal_df, FIG_DIR / "fig2_focal.png")
    plot_protest_ego_networks(phase_graphs, "zh", PROTEST_FOCAL_ZH, phases,
                              FIG_DIR / "fig3_protest_ego_zh.png")
    plot_protest_ego_networks(phase_graphs, "en", PROTEST_FOCAL_EN, phases,
                              FIG_DIR / "fig3_protest_ego_en.png")
    plot_cross_ling_heatmap(corr_df, FIG_DIR / "fig4_crossling_heatmap.png")

    print(f"\nDone. Outputs in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
