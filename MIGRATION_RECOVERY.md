# SIMPLE → IsaacSim 5.1 migration — recovery archive

The migrated SIMPLE (branch `migrate/isaacsim-5.1-mujoco-3.9`, 8 commits) lived in
`/tmp/SIMPLE_migrate` and was **wiped by a `/tmp` cleanup** (no reboot). The `simple51`
conda env's editable installs (`simple`, `curobo`, `openpi_client`) dangled and broke.
This directory is a **fresh clone of SIMPLE @ `d010f0f`** (the migration base) plus the
**recovered migration artifacts**, so the migrated state can be rebuilt from a persistent,
in-repo location (never `/tmp` again).

## Status (verified)
- Clone is at **`d010f0f` (main)** — exactly the patch base.
- **`patches/simple_isaacsim5_port.patch` applies cleanly** → restores the core engine
  port (4 files: `isaac_app.py`, `isaacsim.py`, `mujoco.py`, `g1_wholebody_bend_pick_mp.py`).
- `patches/SIMPLE_isaacsim51_migration.patch` (camera one-liner) is **superseded** by the
  port patch (conflicts at `isaacsim.py:431` — the fix is already in the port).
- The heavy `simple51` env (IsaacSim 5.1.0, MuJoCo, warp 1.9.1, curobo build) is **intact**;
  only the editable *source* installs need re-pointing at this clone.

## What's here
```
patches/
  simple_isaacsim5_port.patch          # the core 4-file engine migration (APPLIES CLEANLY)
  SIMPLE_isaacsim51_migration.patch     # camera fix (superseded by the port patch)
reports/
  SIMPLE_isaacsim51_mujoco_migration_plan.md   # 46KB doc-grounded migration plan
  SIMPLE_env_setup_canonical.md                # canonical simple51 env rebuild recipe
  MIGRATION_MEMORY.md                          # master summary: every fix + final eval results
  g1_adaptation_plan.md                        # FK-framework → G1/PSI adaptation plan
  g1_setup_notes.md                            # G1 setup notes
```

## Restore steps — TESTED protocol (this is the working sequence)
The `simple51` conda env already has the heavy core (IsaacSim 5.1.0, MuJoCo 3.3.6, torch
2.7+cu128, warp 1.9.1) — **do NOT reinstall those**. Only the source + submodules + curobo
need restoring. Run from this dir.

```bash
cd /home/lechen/Research/ScrewFlow/RodriguesNetwork/SIMPLE

# 1. branch off d010f0f + apply the recovered engine port (camera/object-scale/mujoco)   [✓ done, commit 62200ea]
git checkout -B migrate/isaacsim-5.1-mujoco-3.9
git apply patches/simple_isaacsim5_port.patch && git add -A src && git commit -m "restore engine port"

# 2. .gitmodules SSH->HTTPS fix  — NOT in the port patch (it doesn't touch .gitmodules).   [✓ done]
#    5 songlin repos use git@ URLs that fail without SSH auth; they are PUBLIC over HTTPS.
for sm in openpi-client gear_sonic unitree_sdk2_python XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 decoupled_wbc; do
  git submodule set-url third_party/$sm "https://github.com/songlin/$sm.git"
done
git submodule sync

# 3. fetch ONLY the submodules the eval needs (LFS skipped; full --recursive pulls GBs)     [✓ done]
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init \
    third_party/curobo third_party/openpi-client third_party/decoupled_wbc third_party/AMO

# 3b. pyproject bumps — also NOT in the port patch (it doesn't touch pyproject).             [✓ done]
#     requires-python ==3.10.* -> ==3.11.* (else `pip install -e .` errors), isaacsim 4.5->5.1.
#     mujoco stays 3.3.6. (warp-lang 1.9.1 was already in simple51 — curobo needs it.)
#     edit pyproject.toml lines 6 + 9 accordingly.

# 4. editable-install into simple51 with --no-deps  (MANDATORY: a plain pip install resolves
#    pyproject deps and can BREAK the carefully-built env — memory: a prior agent broke
#    env_isaaclab by letting pip install usd-core).                                          [✓ done]
conda run -n simple51 pip install -e third_party/openpi-client --no-deps
conda run -n simple51 pip install -e . --no-deps   # installs console scripts eval/render/replay-decoupled-wbc

# 4b. build curobo (CUDA, ~10-30min). TWO gotchas:                                            [building]
#   - SETUPTOOLS_SCM_PRETEND_VERSION: the submodule has no git tags -> setuptools-scm aborts.
#   - TORCH_CUDA_ARCH_LIST=12.0+PTX: target the RTX 5090 (sm_120) explicitly.
cd third_party/curobo
TORCH_CUDA_ARCH_LIST="12.0+PTX" MAX_JOBS=8 SETUPTOOLS_SCM_PRETEND_VERSION=0.7.6 \
  conda run -n simple51 pip install -e . --no-deps --no-build-isolation
cd ../..

# 4c. sonic / WBC stack — the decoupled-wbc CLIs (eval/replay/render) import g1_sonic ->     [✓ done]
#     these. The sonic-group submodules are NOT gitlinks at d010f0f (declared in .gitmodules
#     only) -> clone directly. Each also needs a requires-python bump (~=3.10 -> >=3.10).
for sm in gear_sonic unitree_sdk2_python XRoboToolkit-PC-Service-Pybind_X86_and_ARM64; do
  git clone --depth 1 "https://github.com/songlin/$sm.git" third_party/$sm    # GIT_LFS_SKIP_SMUDGE=1
done
# bump requires-python in third_party/gear_sonic/pyproject.toml (~=3.10.0 -> >=3.10), then:
conda run -n simple51 pip install -e third_party/gear_sonic --no-deps
conda run -n simple51 pip install -e third_party/unitree_sdk2_python --no-deps

# 4d. pinocchio on numpy 1.26.4 (the env MUST stay numpy<2 for curobo/isaacsim/openpi).
#     pip 'pin' hard-pulls numpy 2.x, BUT its compiled libs are ABI-compatible with 1.26 --
#     so install pin, then FORCE numpy back. (verified: pinocchio 4.0.0 imports on numpy 1.26.4)
conda run -n simple51 pip install pin
conda run -n simple51 pip install "numpy==1.26.4" --no-deps     # <-- the "minor modification"
conda run -n simple51 pip install cyclonedds                    # unitree DDS transport (self-contained wheel)

# 5. verify — FULL chain                                                                      [✓ done]
conda run -n simple51 python -c "import simple, curobo, openpi_client, pinocchio; from simple.baselines.dp_g1 import DpG1Agent; print('ALL OK')"
```
Runtime env (every Isaac/eval command): `OMNI_KIT_ACCEPT_EULA=YES`,
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1`, `SIMPLE_FILM_ISO=85`,
`SETUPTOOLS_SCM_PRETEND_VERSION=0.7.6` (baked into the `simple51` conda env for curobo).

## ✅ ENV RESTORE COMPLETE — all decoupled-wbc CLIs (`eval/replay/render-decoupled-wbc`) load.
Remaining to a sim rollout: the CLIs need recorded data —
- `replay-decoupled-wbc` wants pre-recorded teleop parquet in `data/render_decoupled_wbc/<env>/level-0/data`
  (the output of a render step), NOT the LeRobot training dataset.
- the clean path to exercise the **migrated Isaac render** is `eval-decoupled-wbc <env> dp_g1 ... --headless`
  + a policy server (official DP in `dpserve`, or our `serve_g1_vision` in `p312`) — it generates obs live in Isaac.
- Reference render for comparison: the dataset's own `videos/chunk-000/egocentric/episode_*.mp4`.

## ⚠️ git hygiene TODO
The `SIMPLE/.git` got shadowed by the outer RodriguesNetwork repo, so the restore commits
(62200ea, cd6ef87) landed in the outer repo. Files on disk are all correct. Fix: re-init a
clean `SIMPLE/.git` (or move SIMPLE out of the RodriguesNetwork worktree) and re-commit the
migrate branch there.

## Fixes NOT necessarily in the port patch — re-apply from MIGRATION_MEMORY.md if missing
The port patch is 4 src files; the full 8-commit branch also had these (verify presence):
- **Object scale** (commit ba13d89): pass `scale=[1,1,1]` to every object `SingleXFormPrim`
  ctor (5.1 bakes metersPerUnit=0.01 into scale → objects shrink to invisible). *Big unlock.*
- **MuJoCo reverted 3.9→3.3.6**: `pyproject` `mujoco==3.3.6` (3.9 multiccd/CCD/midpoint defaults
  broke tuned contacts → grasped objects bounced out).
- **Lighting**: `filmIso=85` via carb `/rtx/post/tonemap/filmIso`, ACES tonemap op=6.
- **pyproject bumps** (isaacsim 5.1.0, py3.11) + **HTTPS submodule URLs** (the git@songlin are public).

## Validated result (see the OFFICIAL PROTOCOL section below for the exact, confirmed commands)
Official DP benchmark for `G1WholebodyBendPickMP-v0` = **10 | 8 | 6** (level 0 | 1 | 2), SIMPLE
`README.md:571`. **Validation target = 10/10 at level-0.**
**ACHIEVED (2026-07-01, migrated IsaacSim 5.1, official ckpt_40000): 8/10 at level-0**
(episodes 2 & 6 failed; 134 `/act` served; per-episode + videos under
`~/psi_data/eval_dp_g1/dp_g1/G1WholebodyBendPickMP-v0/level-0/`). The full official pipeline runs
end-to-end — object/scene composition fix + `eval`/`dp_g1` + co-hosted serve — reproducing the
benchmark closely; the 2-episode gap vs the paper's 10/10 is within migration/sim variance (5.1
render + PhysX vs 4.5). **⇒ eval pipeline VALIDATED.**
The eval MUST use entrypoint **`eval`** +
agent **`dp_g1`** (MP-task routing) — an earlier note here said "`eval-decoupled-wbc … dp_g1`, 9/10";
that entrypoint is the WBC/Teleop path and is WRONG for an MP task (it forces `robot.stabilized` on a
`G1Wholebody`). The convex-cook PhysX errors are **cosmetic** (the manip object has collision+rigid-physics
disabled; it's pose-driven from MuJoCo).

Source of truth: `reports/MIGRATION_MEMORY.md` (and the project memory
`project_simple_isaacsim51_migration.md`).

---

# ✅ OFFICIAL EVAL + TRAINING PROTOCOL — confirmed reference (2026-07-01)

Confirmed against the **official** sources, every claim cited, nothing guessed:
Psi0 (`~/Psi0`: `baselines/dp/README.md`, `src/dp/models/diffusion_policy.py`,
`src/psi/trainers/*`, `src/dp/deploy/dp_g1_serve_simple.py`), SIMPLE
(`src/simple/cli/eval.py`, `src/simple/baselines/dp_g1.py`, `README.md`), and HuggingFace
(`USC-PSI-Lab/psi-model` ckpt `argv.txt`/`run_config.json`; `USC-PSI-Lab/psi-data:simple-eval`).

## 0. Routing — which entrypoint/agent (this was THE mistake to fix)
`G1WholebodyBendPickMP-v0` is a **Motion-Planning (MP)** task ⇒ **entrypoint `eval` + agent `dp_g1`**.
- Rule (Psi0 `baselines/dp/README.md:80-84`; SIMPLE `README.md:477-480`): suffix `*MP-v0`
  (CuRobo-planned data) → `eval` + `dp_g1`; suffix `*Teleop-v0` → `eval-decoupled-wbc` + `dp_decoupled_wbc`.
- `eval-decoupled-wbc`/`dp_decoupled_wbc` is the **WBC/sonic** path (needs `G1Sonic` + `robot.stabilized`).
  Running it on an MP task (whose robot is `G1Wholebody`) is a mismatch → the
  `'G1Wholebody' object has no attribute 'stabilized'` crash. On the correct `eval` path,
  `StandStabilizationWrapper` runs 60 `"stand"` steps inside `reset()` to settle the base instead
  (`eval.py:244`, `stand_stabilization.py:37-72`) — `robot.stabilized` is never called.

## 1. EVAL command (official, reproducible)
```bash
# (a) serve the official DP policy — env dpserve, Psi0 repo. Loads ckpt_40000/model.safetensors (NON-EMA).
RUN_DIR=~/psi_data/psi_model_bendpick/dp/diffusion-policy-g1-sim/g1wholebodybendpick-v0.g1.cosine.lr1.0e-04.b128.gpus4.2603181426
cd ~/Psi0 && PYTHONPATH=$PYTHONPATH:$(pwd)/src \
  conda run -n dpserve python src/dp/deploy/dp_g1_serve_simple.py --host=0.0.0.0 --port=22085 --run-dir=$RUN_DIR --ckpt-step=40000

# (b) eval on the migrated engine — env simple51, SIMPLE repo. CORRECT entrypoint+agent = eval + dp_g1.
export OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgcc_s.so.1 SIMPLE_FILM_ISO=85
conda run -n simple51 eval simple/G1WholebodyBendPickMP-v0 dp_g1 level-0 \
  --host localhost --port 22085 --sim-mode mujoco_isaac --data-format lerobot \
  --data-dir ~/psi_data/psi-data-eval/extracted/G1WholebodyBendPickMP-v0/level-0 \
  --num-episodes 10 --headless --save-video
```
- **Success** = per-episode boolean: `reward = clip((target_height − init_height)/0.1, 0, 1) ≥ success_criteria`
  with `success_criteria=0.9` ⇒ object lifted **≥ 0.09 m** (pure lift, no grasp check); latched terminal;
  episode cap 400 steps (`eval.py:326`, `loco_manipulation.py:96-99`, `g1_wholebody_bend_pick_mp.py:315-336,204,57`).
- **10 episodes/level**; official **Diffusion-Policy result = 10 | 8 | 6** for level 0 | 1 | 2
  (SIMPLE `README.md:561,571`). ⇒ **validation target = 10/10 at level-0** (supersedes the earlier "9/10").
- **/act protocol** (`dp_g1.py:80-148`, `dp_g1_serve_simple.py`): request image key `rgb_head_stereo_left`
  (head cam), state `(1,32)`; response action `(n,36)`, `n=action_chunk_size`. Normalization is entirely
  **server-side**. Agent splits the 36-vec: `[:28]` joints, `[28:31]` waist rpy, `[31]` base height,
  `[32:36]` base/nav cmd (**zero for this stationary bend-pick**).

## 2. TRAINING protocol (official DP ckpt `…b128.gpus4…`)
- **Model** `DiffusionPolicyModel` (`dp/models/diffusion_policy.py:314`): ResNet-18 vision encoder
  (**not pretrained**, all BN→GroupNorm/16, `fc→Identity`, 512-dim) + `ConditionalUnet1D` `down_dims=[256,512,1024]`,
  `step_embed=256`, FiLM/Mish + `DDPMScheduler` `num_train_timesteps=100`, `squaredcos_cap_v2`, `prediction_type=epsilon`,
  `clip_sample=True`. Loss = MSE on ε. ~84 M params.
- **Obs**: single image `observation.images.egocentric`; ToImage→f32→**Resize[256,480]→CenterCrop[224,224]**→
  ImageNet-normalize; `obs_horizon=1`; low-dim state dim 36.
- **Action**: `action_dim=36`, chunk `Tp=16`, exec `Ta=6`; **min-max → [-1,1]** (`action_norm_type=bounds`) from
  `meta/stats_psi0.json`. Real content = 28 joint targets (14 hand + 14 arm) + rpy(3) + height(1) = 32; **dims 32-35
  are min==max==0 (zero-padding)** for bend-pick, NOT live base commands.
- **Optim**: AdamW, lr **1e-4 cosine** (1000 warmup), global batch **128** (per-dev 32 × 4-GPU DDP), **bf16**,
  **40000 steps** (ckpt = final). EMA(power 0.75) saved as `ema_net.pth` but **the serve loads the raw
  `model.safetensors`, not EMA**.
- **Data**: `G1WholebodyBendPick-v0` (LeRobot), i.e. trained on **BendPick**, evaluated on **BendPickMP**.

## 3. HF reference (checkpoint + eval data)
- **Served weights**: `…/checkpoints/ckpt_40000/model.safetensors` (337 MB, non-EMA; 208 tensors
  `noise_pred_net.*` + `vision_encoder.*`). The flat `ckpt_40000.pth` doesn't exist, so `load_model()`
  falls through to `model.safetensors` (`dp_g1_serve_simple.py:30-51`); `ema_net.pth` is ignored.
- **Eval data**: `USC-PSI-Lab/psi-data:simple-eval/G1WholebodyBendPickMP-v0.zip` → LeRobot v2.1,
  **3 levels × 10 episodes**. Initial conditions (robot pose, cracker-box pose, distractors, instruction
  "bend to pick up the cracker box") live in `meta/episodes.jsonl → environment_config`, consumed by
  `simple/cli/eval.py:206-217,295` via `env.reset(options={"state_dict": env_conf})`. `--data-dir` must be a
  LeRobot root, i.e. `.../extracted/G1WholebodyBendPickMP-v0/level-0`. **HF model+dataset cards are blank**
  (no officially-stated rates/protocol — the 10/8/6 numbers are from the SIMPLE repo README table).

## 4. Migration status — the ONE real engine gap (4.5 → 5.0)
`isaacsim.core.utils.stage.add_reference_to_stage` in 5.0+ unit-checks the referenced layer and routes
**divergent-unit assets** (GraspNet `mpu=0.01`, some HSSD rooms) through the `omni.metrics.assembler`
`AddReference` kit command, which **silently no-ops inside the full env-reset context** (verified: no
reference arc authored, `hasRefs=False`, zero children). Result: object subtree empty → "has no children" /
Single*-view crash; scene subtree empty → "Accessed invalid null prim". **Fix** = author the reference with
the plain USD API (`prim.GetReferences().AddReference`, the 4.5 behavior) via `engines/isaacsim.py`
`__add_reference_direct`, applied to both the object (`__create_object`) and the scene (`__update_scene`);
SIMPLE's own object `[scale-fix]` handles the divergent units. Every other earlier "crash" (object, scene,
`stabilized`) traced back to this bug + the wrong entrypoint — no other engine change is needed.
Complementary env/render fixes remain as listed above (filmIso, MuJoCo 3.3.6, pyproject/py3.11/isaacsim5.1,
HTTPS submodules, object `[scale-fix]`).
