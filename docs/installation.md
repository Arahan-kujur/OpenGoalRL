# Installation

**Python 3.10+** on **Linux** or **WSL2** (Windows).

## System dependencies (Ubuntu / Debian)

Google Research Football needs C++/SDL build tooling:

```bash
sudo apt update
sudo apt install -y git cmake build-essential libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
    libboost-all-dev libfontconfig1-dev
```

## Python package

```bash
git clone https://github.com/Arahan-kujur/OpenGoalRL.git
cd OpenGoalRL
pip install -e ".[dev]"
```

## Optional dependency groups

| Extra   | Install                     | Purpose                                              |
|---------|-----------------------------|------------------------------------------------------|
| `dev`   | `pip install -e ".[dev]"`   | Full install + pytest (includes `gfootball`)         |
| `test`  | `pip install -e ".[test]"`  | GRF-free test deps (no `gfootball`) — used by CI     |
| `docs`  | `pip install -e ".[docs]"`  | MkDocs Material for building this site               |

!!! tip "Avoiding the GRF build"
    The `test` extra deliberately omits `gfootball` so the GRF-free test suite
    and tooling install cleanly on machines without the C++/SDL dependencies.
    See [Reproduce](reproduce.md) for a zero-setup Docker / Colab path.

## Verify the install

Run the GRF-free test suite (no football engine required):

```bash
pytest opengoalrl/tests/ -v -k "not test_env"
```
