# SIMPLE — Canonical Environment Setup, updated for IsaacSim 5.1 + MuJoCo 3.9 + Python 3.11

Migrated stack (branch `migrate/isaacsim-5.1-mujoco-3.9`), required by the **RTX 5090
(Blackwell, sm_120)** — no support before IsaacSim 5.0.

This updates SIMPLE's official **uv-based** install for the migration, and adds a **conda
fallback** that validates the heavy core (IsaacSim 5.1 + MuJoCo 3.9) install.

---

## 0. Prerequisites (verified on this host)
- **GPU:** RTX 5090 (Blackwell, compute 12.0 / sm_120)
- **Driver:** ≥ 580.65.06 (host: 580.126.09) — ref: IsaacSim 5.1 requirements.html
- **OS:** Linux (Ubuntu 22.04/24.04 class), GLIBC ≥ 2.35
- **Disk:** ~20 GB free for IsaacSim wheels (host: 243 GB)
- **Network:** `https://pypi.nvidia.com` reachable; SSH access to `git@github.com:songlin/*`
  for the private submodules (see §2)
- **git-lfs** installed (assets are LFS-tracked; the official flow uses `GIT_LFS_SKIP_SMUDGE=1`)

## 1. What the migration changed (vs upstream)
`pyproject.toml` (already edited on the branch):
- `requires-python` `==3.10.*` → **`==3.11.*`**  (IsaacSim 5.1 requires Python 3.11)
- `isaacsim[all,extscache]` `4.5.0` → **`5.1.0`**
- `mujoco` `3.3.6` → **`3.9.0`**
- (numpy stays `1.26.4`; torch/torchvision stay `2.7.0/0.22.0` cu128 — already Blackwell-ready)

> ⚠️ **`uv.lock` is NOT regenerated** by the edit — it still pins the old versions. It must be
> re-locked (`uv lock`) after pulling submodules. See §3.

## 2. Clone + submodules (official)
```bash
git clone git@github.com:physical-superintelligence-lab/SIMPLE.git
cd SIMPLE
git checkout migrate/isaacsim-5.1-mujoco-3.9        # the migrated branch
git submodule update --init --recursive
```
Submodules (from `.gitmodules`) — **several are private (`git@github.com:songlin/*`) and need SSH auth**:
- `third_party/curobo` (songlin fork, branch `bodex-support`) — built by `scripts/install_curobo.sh`
- `third_party/gsnet` (graspnet-baseline, public)
- `third_party/AMO` (songlin, the lower-body locomotion policy)
- `third_party/openpi-client` (songlin, **PRIVATE**) — **a CORE dep** (`[tool.uv.sources] openpi-client = path`), so it is required even for the minimal install
- `third_party/gear_sonic`, `unitree_sdk2_python`, `XRoboToolkit-…`, `decoupled_wbc` (songlin, private — only needed for the `sonic` group / decoupled-WBC eval)

## 3. [Option 1] Official uv setup (canonical) — updated
```bash
# install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# RE-LOCK first (required: pyproject changed but uv.lock is stale)
GIT_LFS_SKIP_SMUDGE=1 uv lock                       # regenerates uv.lock for 3.11/5.1/3.9

# then sync all groups
UV_HTTP_TIMEOUT=3000 GIT_LFS_SKIP_SMUDGE=1 uv sync --all-groups --index-strategy unsafe-best-match

# build CuRobo from the submodule
bash scripts/install_curobo.sh                       # honors UV_PROJECT_ENVIRONMENT=.venv, MAX_JOBS

source .venv/bin/activate
python -c "import simple; print(simple.__version__)"
```
**Expected resolve conflicts to handle at `uv lock` (flagged by the migration audit):**
- `dm-control==1.0.21` may pin an older MuJoCo than 3.9.0 → bump `dm-control` if the lock fails.
- `tensorflow==2.15.0` (in `rlds`/`full`, from 2023) — verify a cp311 wheel resolves with numpy 1.26;
  if not, drop the `rlds`/`full` extras or bump tensorflow.
- `lerobot` git fork rev `09929d8` — verify it allows Python 3.11; bump the fork rev if not.
- If a private `sonic`-group submodule is missing, run `uv sync` without `--all-groups`
  (core only) or select groups you have access to.

## 4. [Option 2] conda + pip (fallback; validates the heavy core without uv)
Validated on this host (env `simple51`, Python 3.11.15):
```bash
conda create -n simple51 python=3.11 -y
conda activate simple51
python -m pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
pip install 'mujoco==3.9.0'
# core public deps:
pip install 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu128
pip install 'numpy==1.26.4' dm-control rich transforms3d trimesh typer-slim gymnasium python-dotenv python-fcl
# SIMPLE itself + the private/path deps (openpi-client, gear_sonic, …) still need the submodules:
#   pip install -e .   # only after submodules are present
```
> This conda path gets IsaacSim 5.1 + MuJoCo 3.9 importing for smoke-tests, but `import simple`
> needs the path-sourced submodules (esp. the private `openpi-client`).

## 5. Runtime note — system libgcc (IsaacSim 5.x)
IsaacSim 5.x's bundled hdf5 needs `GCC_12+` in `libgcc_s.so.1`; conda's bundled libgcc is older →
`ImportError` in `omni.isaac.sensor`. Preload the system libgcc when launching:
```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1  <command>
```
(Established with IsaacSim 5.0 on this host; applies to the conda path. Re-verify on 5.1.)

## 6. Verification
IsaacSim requires accepting the EULA on first run — do it non-interactively:
```bash
export OMNI_KIT_ACCEPT_EULA=YES
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1   # §5
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python -c "import torch; print('torch', torch.__version__)"   # must be +cu128 for the 5090
python -c "from isaacsim.simulation_app import SimulationApp; a=SimulationApp({'headless':True}); a.close(); print('isaacsim 5.1 boots')"
python -c "import simple; print('simple', simple.__version__)"   # needs submodules
```
**End-to-end robot render smoke-test** (validated — produces an mp4 of a Franka articulating):
```bash
OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 \
  python /tmp/isaac_robot_render.py        # loads Franka, drives joints, captures RTX camera
```
Pattern: `World()` → `world.scene.add(Franka(prim_path=...))` → `world.reset()` (registering
the robot with the scene BEFORE reset is required, else the articulation view is empty) →
`Camera(...).get_rgba()` per `world.step(render=True)`.

## 7. Resolved versions (VALIDATED in env `simple51`, Python 3.11.15)
| package | installed | note |
|---|---|---|
| isaacsim | **5.1.0.0** | `isaacsim[all,extscache]` from pypi.nvidia.com; SimulationApp boots on the 5090 ✓ |
| mujoco | **3.9.0** | imports ✓ |
| dm-control | **1.0.21** | installs cleanly **with** mujoco 3.9.0 (no conflict — the flagged risk did not materialize) |
| numpy | **1.26.0** | (`<2.0` cap respected) |
| torch / torchvision | **2.7.0+cu128 / 0.22.0+cu128** | MUST force the cu128 index — isaacsim pulls cu126 by default, which lacks Blackwell sm_120 kernels |

**Validated end-to-end:** all migrated `isaacsim.*` import paths resolve on real 5.1; a Franka
robot loads from the 5.1 asset server, runs physics + articulation control, and RTX-renders to
camera frames (160-frame mp4). Only cosmetic warning at install: `packaging 23.0 vs wheel wants
≥24.0` (harmless; isaacsim pins packaging 23.0).

**Still requires the submodules** for `import simple` (all 8 are PUBLIC over HTTPS after the
`.gitmodules` fix; `git submodule update --init --recursive`). The full `uv sync --all-groups`
+ `uv lock` regeneration + `scripts/install_curobo.sh` are the canonical path (§3).
