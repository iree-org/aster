# Graph Coloring Utility

A command-line tool for reading, converting, and coloring graphs using NetworkX.

## Prerequisites

- Python 3.10+
- Graphviz development libraries (required to build `pygraphviz`)

On Ubuntu/Debian:

```bash
sudo apt install graphviz libgraphviz-dev
```

## Setup

```bash
python -m venv .graph
source .graph/bin/activate
pip install -r requirements.txt
```

## Usage

The script provides three subcommands: `convert`, `color`, and `stats`.

### convert

Convert a DOT graph to pickle format for fast subsequent loading.

```bash
python color.py convert graph.dot              # Output: graph.pkl
python color.py convert graph.dot -o out.pkl   # Custom output path
```

### color

Color a graph using greedy graph coloring and print statistics.

```bash
python color.py color graph.dot                # Default strategy (DSATUR)
python color.py color graph.pkl -s largest_first  # Specific strategy
python color.py color graph.dot --all          # Compare all strategies
```

Available strategies:

| Strategy | Description |
|----------|-------------|
| `DSATUR` | Saturation-based ordering (default) |
| `largest_first` | Nodes ordered by decreasing degree |
| `smallest_last` | Reverse smallest-last ordering |
| `random_sequential` | Random node ordering |
| `independent_set` | Greedy independent set removal |
| `connected_sequential_bfs` | BFS-based connected ordering |
| `connected_sequential_dfs` | DFS-based connected ordering |

### stats

Print graph statistics including connected components and sample node labels.

```bash
python color.py stats graph.dot
python color.py stats graph.pkl
```

## Supported Formats

| Extension | Format | Notes |
|-----------|--------|-------|
| `.dot` | Graphviz DOT | Read via `pygraphviz` |
| `.pkl` | Python pickle | Fastest read/write, use `convert` to create |
