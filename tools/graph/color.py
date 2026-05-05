import argparse
import pickle
import time
from pathlib import Path

import networkx as nx
from networkx.drawing.nx_agraph import read_dot


def read_graph(path: str) -> nx.Graph:
    """Read a graph from a dot file or a pickle file."""
    if path.endswith(".pkl"):
        return load_pickle(path)
    return nx.Graph(read_dot(path))


def save_pickle(G: nx.Graph, path: str):
    """Save a graph to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> nx.Graph:
    """Load a graph from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def cmd_convert(args):
    """Convert a dot graph to pickle format for fast loading."""
    start = time.perf_counter()
    print(f"Reading graph from {args.input}...", flush=True)
    G = read_graph(args.input)
    read_time = time.perf_counter() - start

    output = args.output
    if output is None:
        output = str(Path(args.input).with_suffix(".pkl"))

    print(f"Saving graph to {output}...", flush=True)
    save_pickle(G, output)
    write_time = time.perf_counter() - start - read_time

    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Read time:  {read_time:.4f}s")
    print(f"Write time: {write_time:.4f}s")


ALL_STRATEGIES = [
    "largest_first",
    "random_sequential",
    "smallest_last",
    "independent_set",
    "connected_sequential_bfs",
    "connected_sequential_dfs",
    "DSATUR",
]


def color_graph(G: nx.Graph, strategy: str):
    """Color a graph and return the coloring, number of colors, and time taken."""
    color_start = time.perf_counter()
    coloring = nx.coloring.greedy_color(G, strategy=strategy)
    color_time = time.perf_counter() - color_start
    num_colors = max(coloring.values()) + 1 if coloring else 0
    return coloring, num_colors, color_time


def print_coloring_stats(
    G: nx.Graph,
    coloring: dict,
    strategy: str,
    num_colors: int,
    color_time: float,
    components: list,
):
    """Print coloring statistics for a single strategy."""
    print(f"\n  Strategy:            {strategy}")
    print(f"  Colors used (total): {num_colors}")
    print(f"  Color time:          {color_time:.4f}s")
    print("  Colors per component:")
    for i, component in enumerate(components):
        colors_in_component = {coloring[node] for node in component}
        print(f"    Component {i}: {len(colors_in_component)} colors")


def cmd_color(args):
    """Color a graph and print statistics."""
    start = time.perf_counter()
    print(f"Reading graph from {args.input}...", flush=True)
    G = read_graph(args.input)
    read_time = time.perf_counter() - start

    components = list(nx.connected_components(G))

    print("\n--- Graph Statistics ---")
    print(f"Nodes:                {G.number_of_nodes()}")
    print(f"Edges:                {G.number_of_edges()}")
    print(f"Connected components: {len(components)}")

    strategies = ALL_STRATEGIES if args.all else [args.strategy]

    print("\n--- Coloring Results ---")
    for strategy in strategies:
        coloring, num_colors, color_time = color_graph(G, strategy)
        print_coloring_stats(G, coloring, strategy, num_colors, color_time, components)

    print("\n--- Timing ---")
    print(f"Read time:  {read_time:.4f}s")
    print(f"Total time: {time.perf_counter() - start:.4f}s")


def cmd_stats(args):
    """Print graph statistics including connected components and sample node labels."""
    print(f"Reading graph from {args.input}...", flush=True)
    G = read_graph(args.input)

    components = list(nx.connected_components(G))
    print("\n--- Graph Statistics ---")
    print(f"Nodes:                {G.number_of_nodes()}")
    print(f"Edges:                {G.number_of_edges()}")
    print(f"Connected components: {len(components)}")

    print("\n--- Components (first 5 node labels each) ---")
    for i, component in enumerate(components):
        sample = list(component)[:5]
        labels = []
        for node in sample:
            label = G.nodes[node].get("label", node)
            labels.append(label)
        print(f"  Component {i}: {labels}")


def main():
    parser = argparse.ArgumentParser(description="Graph coloring utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Convert subcommand.
    convert_parser = subparsers.add_parser(
        "convert", help="Convert a dot graph to pickle format for fast loading."
    )
    convert_parser.add_argument("input", help="Path to the input graph (.dot or .pkl).")
    convert_parser.add_argument(
        "-o",
        "--output",
        help="Output pickle path (default: input with .pkl extension).",
    )
    convert_parser.set_defaults(func=cmd_convert)

    # Color subcommand.
    color_parser = subparsers.add_parser(
        "color", help="Color a graph and print statistics."
    )
    color_parser.add_argument("input", help="Path to the input graph (.dot or .pkl).")
    color_parser.add_argument(
        "-s",
        "--strategy",
        default="DSATUR",
        choices=ALL_STRATEGIES,
        help="Coloring strategy (default: DSATUR).",
    )
    color_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Run all coloring strategies and compare results.",
    )
    color_parser.set_defaults(func=cmd_color)

    # Stats subcommand.
    stats_parser = subparsers.add_parser(
        "stats", help="Print graph statistics and sample node labels per component."
    )
    stats_parser.add_argument("input", help="Path to the input graph (.dot or .pkl).")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
