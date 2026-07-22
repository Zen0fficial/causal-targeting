# Stable Causal Targeting

This repository is organized around the final manuscript and its supporting
materials.

- `document/main_article/` contains the manuscript source, tables, figures, and PDF.
- `document/reference/` contains cited and review materials.
- `document/talk/` contains the presentation source, figures, and PDF.
- `document/reproducing_code/` contains the standalone reproduction project.

## Reproduction

Install the Python dependencies and run the cached workflow:

```sh
cd document/reproducing_code
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run.sh
```

The default command reads the canonical computation cache and regenerates the
tables and figures under `outputs/`. Direct refitting is available with
`./run.sh regenerate`; it is substantially more time consuming and writes to a
separate cache directory.
