# SIMPLE @ IsaacSim 5.1 — full eval-stack replication on an RTX 5090 (sm_120)

This branch (`migrate/isaacsim-5.1-mujoco-3.9`) is the validated 4.5→5.1 port
behind every published ScrewFlow/G1 number. VERIFIED 2026-07-12 against the
production machine: all 468 tracked code files byte-identical to the working
copy; ALL FOUR required submodules (gear_sonic, unitree_sdk2_python, AMO,
decoupled_wbc) verified byte-identical to their pinned SHAs (2026-07-12; the
two with stale gitdir pointers were rebuilt as standalone repos — zero drift). One local delta, captured here as
`patches_gear_sonic_py311.patch` (gear_sonic pyproject requires-python
`~=3.10.0` → `>=3.10` so it installs under py3.11 — apply after submodule init).

## 1. Environment (exact production versions; full lock: simple51_requirements_frozen.txt)

```bash
conda create -n simple51 python=3.11 -y && conda activate simple51
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install mujoco==3.3.6
# remaining pins: pip install -r simple51_requirements_frozen.txt  (best effort;
# torch/isaacsim above take precedence)
```

## 2. This repo + submodules

```bash
git clone -b migrate/isaacsim-5.1-mujoco-3.9 https://github.com/ErvinZzz/SIMPLE
cd SIMPLE
git submodule update --init third_party/gear_sonic third_party/unitree_sdk2_python \
    third_party/AMO third_party/decoupled_wbc     # ONLY these 4 are needed for
                                                  # dp_g1 / eval-decoupled-wbc;
                                                  # curobo/gsnet/openpi are NOT
(cd third_party/gear_sonic && git apply ../../patches_gear_sonic_py311.patch)
SETUPTOOLS_SCM_PRETEND_VERSION=0.7.6 pip install -e .
pip install -e third_party/gear_sonic -e third_party/unitree_sdk2_python
```

## 3. Assets (~6.2 GB, NOT in git: robot USDs, hssd scenes, materials)

Follow upstream `docs/source/tutorials/installation.md` for the asset download
into `data/` (robots/g1/*.usd, scenes/hssd/*, vMaterials_2). Eval episode sets:
`simple-eval/<env-id>.zip` per task from the psi-data HF dataset (see
ScrewFlow scripts/cluster/fetch_data.sh for URLs + expected episode counts).

## 4. Machine prep (each cost real debugging time — do NOT skip)

1. **inotify** (Isaac crashes at startup otherwise, especially with VS Code):
   `sudo sysctl fs.inotify.max_user_watches=1048576 fs.inotify.max_user_instances=1024`
   (persist in /etc/sysctl.d/90-inotify.conf).
2. Required env vars for every eval invocation (see the ScrewFlow drivers,
   scripts/g1/serve_and_eval_*.sh, which set them):
   `OMNI_KIT_ACCEPT_EULA=YES`, `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1`
   (conda-vs-system libgcc conflict), `SIMPLE_FILM_ISO=85`,
   `SETUPTOOLS_SCM_PRETEND_VERSION=0.7.6`, and a PATH that puts the simple51 env
   first.
3. ONE Isaac instance at a time; Kit may crash at TEARDOWN after successful work
   — judge success by output episode counts, never exit codes.

## 5. Validation gates (run in order; do not use the machine for results until all pass)

1. `eval --help` and `eval-decoupled-wbc --help` resolve (console scripts).
2. GT demo replay on BendPickMP = 10/10 (physics faithfulness; see ScrewFlow
   docs/g1_runbook.md gotcha #6 for replay CLI flags).
3. Official released DP checkpoint through this stack at level-0 = **8/10**
   (the 5.1-port anchor; 4.5 upstream reports 10/10 — the port is known to be
   slightly harder, this is expected).
4. ScrewFlow flagship vtok_dino ckpt (HF archive) official level-0 = **10/10**.
Numbers off by more than ±1 episode → stop and diff your stack, do not proceed.
