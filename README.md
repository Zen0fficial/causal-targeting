# Stable Causal Targeting

This repository is organized around the final manuscript and its supporting
materials.

- `main_article/` contains the manuscript source, tables, figures, and PDF.
- `reference/` contains cited and review materials.
- `talk/` contains the presentation source, figures, and PDF.
- `reproducing_code/` contains the standalone reproduction project.

## Reproduction

Install the Python dependencies and run the cached workflow:

```sh
cd reproducing_code
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run.sh
```

The default command reads the canonical computation cache and regenerates the
tables and figures under `outputs/`. Direct refitting is available with
`./run.sh regenerate`; it is substantially more time consuming and writes to a
separate cache directory.
