import sys, math, itertools
import pandas as pd
import networkx as nx

def load_data(path):
    return pd.read_csv(path)

def get_cardinalities(df):
    return {col: int(df[col].max()) for col in df.columns}

def _dirichlet_multinomial_logscore_row(counts, r_child, alpha=1.0):
    # counts: list/iterable of length r_child for one parent configuration
    m0 = sum(counts)
    alpha0 = r_child * alpha
    s = math.lgamma(alpha0) - math.lgamma(alpha0 + m0)
    for m_k in counts:
        s += math.lgamma(alpha + m_k) - math.lgamma(alpha)
    return s

def local_bdeu_score(df, child, parents, card, alpha=1.0):
    r_child = card[child]
    if not parents:
        # counts of child values 1..r_child
        vc = df[child].value_counts()
        counts = [int(vc.get(k, 0)) for k in range(1, r_child+1)]
        return _dirichlet_multinomial_logscore_row(counts, r_child, alpha)

    # group by parents + child, count rows
    gb = df.groupby(list(parents) + [child], dropna=False).size().rename("n").reset_index()
    # pivot to one row per parent config with r_child columns (child=1..r_child)
    # Note: we only consider observed parent configs; missing configs contribute 0 to the total score.
    s = 0.0
    for _, sub in gb.groupby(list(parents)):
        # counts for child = 1..r_child within this parent config
        row_counts = [0] * r_child
        for _, rec in sub.iterrows():
            k = int(rec[child])  # 1..r_child
            row_counts[k-1] = int(rec["n"])
        s += _dirichlet_multinomial_logscore_row(row_counts, r_child, alpha)
    return s

def graph_score(df, G, card, alpha=1.0):
    total = 0.0
    for v in G.nodes():
        parents = tuple(G.predecessors(v))
        total += local_bdeu_score(df, v, parents, card, alpha)
    return total

def would_create_cycle(G, u, v):
    # adding u->v creates a cycle iff v can already reach u
    return nx.has_path(G, v, u)

def greedy_add_only(df, parent_cap=3, alpha=1.0):
    G = nx.DiGraph()
    G.add_nodes_from(df.columns)
    card = get_cardinalities(df)

    # start at empty
    current_score = graph_score(df, G, card, alpha)

    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_edge = None

        for u, v in itertools.permutations(G.nodes, 2):
            if G.has_edge(u, v): 
                continue
            # parent cap on v
            if len(list(G.predecessors(v))) >= parent_cap:
                continue
            if would_create_cycle(G, u, v):
                continue

            # delta = score(v with parents ∪ {u}) - score(v with parents)
            old_parents = tuple(G.predecessors(v))
            old_local = local_bdeu_score(df, v, old_parents, card, alpha)

            new_parents = tuple(sorted(old_parents + (u,)))
            # temporarily compute new local without mutating G
            new_local = local_bdeu_score(df, v, new_parents, card, alpha)

            delta = new_local - old_local
            if delta > best_delta:
                best_delta = delta
                best_edge = (u, v)

        if best_edge is not None and best_delta > 0.0:
            u, v = best_edge
            G.add_edge(u, v)
            current_score += best_delta
            improved = True

    return G

def write_gph(dag, _idx2names_unused, filename):
    with open(filename, 'w') as f:
        for u, v in dag.edges():
            f.write(f"{u}, {v}\n")

def compute(infile, outfile, parent_cap=3, alpha=1.0):
    df = load_data(infile)
    dag = greedy_add_only(df, parent_cap=parent_cap, alpha=alpha)
    # idx2names: project spec wants index->name mapping; here nodes are names already
    names = list(df.columns)
    name2idx = {name: i for i, name in enumerate(names)}
    idx2names = {i: n for n, i in name2idx.items()}
    write_gph(dag, idx2names, outfile)

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)

if __name__ == "__main__":
    main()
