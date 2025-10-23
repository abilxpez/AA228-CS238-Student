import sys, math, itertools
import pandas as pd
import networkx as nx

def load_data(path):
    return pd.read_csv(path)

def get_cardinalities(df):
    return {col: int(df[col].max()) for col in df.columns}

# get log Bayesian score for one parent configuration j of one child node i
def row_score(m_ijk, r_i, alpha=1.0):
    m_ij0 = sum(m_ijk)           
    alpha_ij0 = r_i * alpha     

    # implement log Γ(α_ij0) - log Γ(α_ij0 + m_ij0)
    log_score = math.lgamma(alpha_ij0) - math.lgamma(alpha_ij0 + m_ij0)

    # implement Σ_k [log Γ(α + m_ijk) - log Γ(α)]
    for count in m_ijk:
        log_score += math.lgamma(alpha + count) - math.lgamma(alpha)

    return log_score

    # get log Bayesian score for node i by summing over all parent configurations j

def node_score(df, child, parents, cardinality, alpha=1.0):

    r_i = cardinality[child]

    # handle no-parent case, build single counts row and score it
    if not parents:
        vc = df[child].value_counts()
         # build counts row m_ijk in ascending k 
        m_ijk = [int(vc.get(k, 0)) for k in range(1, r_i + 1)]
        return row_score(m_ijk, r_i, alpha)

    # group by parents and child to get m_ijk counts
    # reshape rows=j (parent) and columns=k (child)
    grouped = (
        df.groupby(list(parents) + [child], dropna=False)
          .size()
          .rename("n")
          .reset_index()
          .pivot_table(index=list(parents), columns=child, values="n", fill_value=0)
    )

    # ensure columns cover child values in order
    grouped = grouped.reindex(columns=range(1, r_i + 1), fill_value=0)

    # sum over parent configurations j, apply the row scorer to each
    total_log_score = 0.0
    for _, row in grouped.iterrows():
        # build counts row m_ijk in ascending k 
        m_ijk = [int(row[k]) for k in range(1, r_i + 1)]
        total_log_score += row_score(m_ijk, r_i, alpha)

    return total_log_score

# sum over nodes i
def graph_score(df, G, cardinality, alpha=1.0):
    total_log_score = 0.0
    for i in G.nodes():
        # parents 
        Pa_i = tuple(G.predecessors(i))
        # add node i contribution Σ_j [...]
        total_log_score += node_score(df, i, Pa_i, cardinality, alpha)
    return total_log_score

# adding u->v creates a cycle iff v can already reach u
def would_create_cycle(G, u, v):
    return nx.has_path(G, v, u)

# check if we can add u->v under DAG and parent cap
def can_add_edge(G, u, v, parent_cap):
    # skip self-loops
    if u == v:
        return False
    # skip duplicate edges
    if G.has_edge(u, v):
        return False
    # skip if v already reached parent limit
    if len(list(G.predecessors(v))) >= parent_cap:
        return False
    # skip if adding edge would create a cycle
    if would_create_cycle(G, u, v):
        return False
    return True

# local delta for adding u->v, only node v changes
def delta_add(df, G, u, v, cardinality, alpha):
    Pa_v = tuple(G.predecessors(v))
    old_local = node_score(df, v, Pa_v, cardinality, alpha)
    # new local -> old local (Pa_v) + new u 
    new_local = node_score(df, v, tuple(sorted(Pa_v + (u,))), cardinality, alpha)
    return new_local - old_local

# use greedy algorithm to iteratively add edges that improve the Bayesian score
def greedy_add_only(df, parent_cap=3, alpha=1.0):
    # initialize empty DAG
    G = nx.DiGraph()
    G.add_nodes_from(df.columns)
    card = get_cardinalities(df)

    # sum over nodes i
    current_score = graph_score(df, G, card, alpha)

    # hill climb by adding best edge
    improved = True
    while improved:
        improved = False
        best_delta = 0.0
        best_edge = None

        # search all possible directed edges u->v to 
        # find the one that improves score the most
        for u, v in itertools.permutations(G.nodes, 2):
            # skip edges, call helper 
            if not can_add_edge(G, u, v, parent_cap):
                continue
            # compute score change delta for adding u->v, call helper
            d = delta_add(df, G, u, v, card, alpha)
            if d > best_delta:
                best_delta = d
                best_edge = (u, v)

        # add the best edge if it improves the total graph score
        if best_edge is not None and best_delta > 0.0:
            u, v = best_edge
            G.add_edge(u, v)
            current_score += best_delta
            improved = True

    return G

# write learned DAG edges to .gph file
def write_gph(dag, _idx2names_unused, filename):
    # write each edge as "parent, child" on a new line
    with open(filename, "w") as f:
        for u, v in dag.edges():
            f.write(f"{u}, {v}\n")


# load data, learn structure, and save output
def compute(infile, outfile, parent_cap=3, alpha=1.0):
    # load dataset
    df = load_data(infile)

    # learn DAG using greedy add-only algorithm
    dag = greedy_add_only(df, parent_cap=parent_cap, alpha=alpha)

    # map variable names to indices for .gph format
    names = list(df.columns)
    name2idx = {name: i for i, name in enumerate(names)}
    idx2names = {i: n for n, i in name2idx.items()}

    # write resulting graph to file
    write_gph(dag, idx2names, outfile)

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)

if __name__ == "__main__":
    main()
