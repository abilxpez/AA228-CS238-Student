import sys
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


# read .gph file and return directed graph
def load_gph(filepath):
    G = nx.DiGraph()
    with open(filepath, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                parent, child = parts
                G.add_edge(parent, child)
    return G


# plot DAG using circular layout
def plot_graph(G, title, save_path=None):
    plt.figure(figsize=(6, 6))
    nx.draw_circular(
        G,
        with_labels=True,
        node_size=1200,
        node_color="#CCE5FF",
        arrowsize=15,
        font_size=10,
        edgecolors="black",
    )
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


# generate plots for one or more .gph files
def main():
    # if no args, default to plotting all three graphs
    if len(sys.argv) == 1:
        gph_files = ["small.gph", "medium.gph", "large.gph"]
    else:
        # accept one or more .gph filenames as arguments
        gph_files = sys.argv[1:]

    for file in gph_files:
        path = Path(file)
        if not path.exists():
            print(f"File not found: {path}")
            continue

        G = load_gph(path)
        title = path.stem
        save_path = path.with_suffix(".png")

        plot_graph(G, f"{title} graph", save_path)


if __name__ == "__main__":
    main()
