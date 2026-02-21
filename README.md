# ROSE — Reflectivity Open Science Engine

Automated neutron reflectivity analysis, powered by multi-agent AI coding.

> **AuRE v2** — successor to [AuRE](https://github.com/neutrons-ai/aure).

---

## Overview

ROSE automates the analysis of neutron reflectivity data from thin-film
experiments. Given raw reflectivity measurements and a plain-language
description of the sample, ROSE fits a layered model to the data and returns
structural parameters (thickness, roughness, scattering length density) with
uncertainties.

## Development Model

This project is built by a team of **four AI agents** coordinated through
[OpenHands](https://github.com/All-Hands-AI/OpenHands), using a local
LLM server:

| Agent        | Role               | Description                             |
|--------------|--------------------|-----------------------------------------|
| **PM**       | Project Manager    | Decomposes work, delegates, coordinates |
| **dev-alpha**| Senior Developer   | Core algorithms and scientific code     |
| **dev-beta** | Senior Developer   | I/O, CLI, data pipelines, integration   |
| **tester**   | QA Engineer        | Runs tests, verifies quality            |

Agent personas are defined in [.openhands/microagents/](.openhands/microagents/).

---

## Quick Start

### Prerequisites

- Python >= 3.9
- Docker (for OpenHands)
- A local OpenAI-compatible LLM server (default: `http://localhost:8555/v1`)

### Install the package

```bash
# Clone the repository
git clone https://github.com/neutrons-ai/rose.git
cd rose

# Install in editable mode with dev dependencies
pip install -e ".[dev,cli,science]"

# Run the test suite
pytest
```

### Launch OpenHands (multi-agent environment)

```bash
# Recommended: uses uv (installs automatically if needed)
./scripts/launch-openhands.sh

# Alternative: use Docker directly
./scripts/launch-openhands.sh docker
```

The script will:
1. Check Docker and LLM server connectivity
2. Auto-switch from Docker Desktop to native Docker Engine if needed
3. Install `uv` and `openhands` if not present (uv method)
4. Mount the project directory into the sandbox
5. Print a ready-to-paste startup prompt (and copy it to your clipboard if `xclip` is available)

Open `http://localhost:3000`, configure the LLM in Settings (LLM tab → Advanced):

| Setting       | Value                                     |
|---------------|-------------------------------------------|
| Custom Model  | `openai/gpt-oss-120b`                     |
| Base URL      | `http://host.docker.internal:8555/v1`     |
| API Key       | `not-needed`                              |

Then paste the startup prompt and the PM agent will take over.

---

## Project Structure

```
rose/
├── .openhands/
│   └── microagents/         # Agent persona definitions
│       ├── pm.md            # Project Manager
│       ├── dev-alpha.md     # Developer Alpha
│       ├── dev-beta.md      # Developer Beta
│       └── tester.md        # QA Engineer
├── docs/
│   └── project.md           # Full project plan and reference data
├── scripts/
│   └── launch-openhands.sh  # Docker launch script
├── src/
│   └── rose/
│       ├── __init__.py
│       └── cli.py           # Click CLI entry point
├── tests/
│   └── test_cli.py
├── pyproject.toml
└── README.md
```

---

## Reference Data

Validation data for test cases lives in a separate repository:

- **Fit models:** `$USER/git/experiments-2024/val-sep24/models/corefined/`
  - `<model>-refl.dat` — reflectivity data with ground truth `theory` column
  - `<model>.err` — ground truth model parameters with uncertainties
- **Input data:** `$USER/git/experiments-2024/val-sep24/data/`

### Sample Description (for all test cases)

> Copper main layer (50 nm) on a titanium sticking layer (5 nm) on a silicon
> substrate. The ambient medium is most likely dTHF electrolyte, but may be THF.
> The reflectivity was measured from the back of the film, with the incoming beam
> coming from the silicon side.

### Data Set Mapping

| Cu Substrate | Condition           | Runs            |
|--------------|---------------------|-----------------|
| D            | Cycling             | 213032 & 213036 |
| I            | Sustained           | 213082 & 213086 |
| F            | 0% ethanol          | 213046 & 213050 |
| D            | 1% ethanol          | —               |
| E            | 2% ethanol          | 213039 & 213043 |
| D            | d8-THF + EtOH       | —               |
| G            | d8-THF + d6-EtOH    | 213056 & 213060 |
| M            | THF + d6-EtOH       | 213136 & 213140 |
| K            | THF + EtOH          | 213110 & 213114 |
| D            | 0.2 M Cabhfip       | —               |
| L            | 0.1 M Cabhfip       | 213126 & 213130 |

---

## Configuration

All configuration can be overridden via environment variables when launching
OpenHands:

| Variable          | Default                                    | Description            |
|-------------------|--------------------------------------------|------------------------|
| `LLM_BASE_URL`   | `http://localhost:8555/v1`                 | LLM server endpoint   |
| `LLM_MODEL`      | `openai/gpt-oss-120b`                     | Model identifier       |
| `OPENHANDS_PORT`  | `3000`                                     | UI port                |
| `OPENHANDS_IMAGE` | `ghcr.io/all-hands-ai/openhands:0.20`     | Docker image           |

---

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.
