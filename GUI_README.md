



## Installation

```bash
# From your CiproSeg project root:
pip install -e ".[gui]"
```

## Usage

```bash
# Launch the GUI (auto-opens browser)
ciproseg-gui

# Custom port
ciproseg-gui --port 8080

# Without auto-opening browser
ciproseg-gui --no-browser

# Run on different host (e.g. for remote server)
ciproseg-gui --host 0.0.0.0 --port 7860
```

The GUI will be available at `http://127.0.0.1:7860`.

## How It Works

- The backend wraps your existing CLIs (`ciproseg-train`, `ciproseg-eval`,
  `ciproseg-infer`) as subprocess calls, so all training/evaluation/inference
  logic stays untouched.
- Log output is captured and streamed to the browser in real time.
- The runs directory is scanned for `run_summary.json`, `best.json`, and
  `config_resolved.yaml` files that your trainer already generates.
- Images (training plots, inference overlays) are served directly from disk.
- Zero modifications to any existing CiproSeg module.

## Environment Variable

Set `CIPROSEG_RUNS` to point to a custom runs root directory:

```bash
CIPROSEG_RUNS=/data/experiments ciproseg-gui
```

Default is `./runs` (relative to where you launch the command).
