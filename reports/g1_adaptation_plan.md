# Adapting the FK-Factorized Framework to Unitree G1 (PSI / SIMPLE)

*Feasibility + concrete plan for transferring the FK-feature + Jacobian-factorized
BC/diffusion-policy framework to humanoid (Unitree G1) loco-manipulation, and for
testing whether the FK prior improves diffusion policy there. Targets the
USC-PSI-Lab / Physical-Superintelligence-Lab stack.*

> Sourcing note: Hugging Face was 403-blocked from this environment. The PSI DP spec
> below was reconstructed from the **authoritative training/serving code** in
> `github.com/physical-superintelligence-lab/Psi0` and `…/SIMPLE` (two independent
> agents agree). Fields only in HF blobs (exact image resolution, dataset fps/size,
> checkpoint manifest) are flagged UNVERIFIED.

---

## 1. The target stack (verified from source)

**Benchmark — SIMPLE** (arXiv 2606.08278): humanoid loco-manipulation sim
(MuJoCo physics + IsaacSim rendering), Unitree G1, ~6 core whole-body tasks
(`G1WholebodyXMovePickTeleop-v0`, `…BendPickMP-v0`, `…HandoverTeleop-v0`,
`…LocomotionPickBetweenTables…`, `…TabletopGraspMP-v0`, `…XMoveBendPick…`). Eval =
success over 10 trials at 3 progressive OOD levels (visual rand → lighting →
object-pose perturb). The learned baselines in the SIMPLE repo are **HTTP inference
clients**; the model runs on a server.

**Embodiment — G1 29-DoF + two 7-DoF Dex3-1 dexterous hands.** Manipulation policy
controls upper body + hands + a base command; **legs are handled by a separate
downstream whole-body/AMO controller** (decoupled WBC), not by direct joint targets.

**Action — 36-D absolute whole-body target** (from `src/psi/trainers/diffusion_policy_g1.py`):

| Indices | Meaning | DoF |
|---|---|---|
| 0–13 | Both Dex3 hands (7 + 7) | 14 |
| 14–27 | Both arms (7 + 7) | 14 |
| 28–30 | Waist / torso RPY | 3 |
| 31 | Base height | 1 |
| 32–34 | Base velocities vx, vy, vyaw | 3 |
| 35 | Torso target yaw | 1 |

Absolute joint/pose targets for arms/hands/waist + a **velocity** command for the base.

**PSI "Diffusion Policy" baseline (the thing to beat) — verified:**
- Custom (real-stanford-lineage) **ConditionalUnet1D**, `down_dims=[256,512,1024]`,
  kernel 5, step-embed 256, n_groups 8.
- Vision: **ResNet-18, random init (no ImageNet)**, BatchNorm→GroupNorm, 512-D, **1
  egocentric RGB camera**; conditioning = concat(image feat, proprio).
- **obs_horizon = 1**, **pred_horizon = 16**, **exec_horizon = 6** (receding).
- **DDPM, 100 train + 100 inference steps**, cosine (`squaredcos_cap_v2`),
  `prediction_type='epsilon'`, `clip_sample=True`, **no EMA**.
- action/state **bounds (min-max) normalized** to [-1,1]; hands normalized jointly with
  everything else, **no special gripper/hand head**.
- AdamW (betas 0.95/0.999, wd 1e-6), lr 1e-4 cosine + 1000 warmup, bs 32, bf16,
  **40k steps**, seed 2026. Data in **LeRobot** format, one zip per task; a skill can be
  fine-tuned from "~80 trajectories."
- obs = `observation.images.egocentric` (RGB) + `states` (proprio, 36-D padded) +
  `task` (language). **No privileged object/goal pose in the observation.**

---

## 2. What this repo already has for G1

- `configs/robots/g1.yaml` — **full 43-DoF branched tree** (12 legs + 3 waist + 2×7 arms
  + 2×7 Dex3-hand) with URDF; `configs/robots/g1_23dof.yaml` — 23-DoF arms-only
  (ends at `…_rubber_hand`). Both run through the repo's differentiable
  `RobotModel.forward_kinematics`, which already handles G1's branching (two arms, two
  hands, two legs).
- `configs/datasets/FK/g1*` — but these wire G1 **only into the FK-fitting auxiliary
  benchmark** (`root_fixed: true`, random qpos ∈ ±π). There is **no G1 policy, no
  `KinematicSpec`, no `ActionJacobian`, no G1 imitation data** in the repo.

**Verdict:** differentiable FK for G1 is ~ready. The factorized **decoder** for G1 must
be built new, and several framework mismatches (below) must be resolved first.

---

## 3. The mapping (how the framework lands on G1)

The PSI 36-D action maps onto the factorization cleanly **if you treat it like a
bimanual Fetch**:

| Action block | Framework role | Notes |
|---|---|---|
| Arms 0–6 (left), 7–13 (right) → joints 14–27 | **Kinematic** — two Jacobians | Two chains: `torso_link → left_wrist`, `torso_link → right_wrist`. Each task_dim 3 (pos) or 6 (pose). Decode each arm's 7 joints via its own `J⁺Δx + (I−J⁺J)n`. |
| Hands 0–13 (Dex3 ×2) | **Direct head** | The "gripper" analog, but 14-D and dexterous. Keep on the direct head initially (as Fetch gripper is). |
| Waist RPY 28–30 | **Shared / direct** | Waist is the *parent* of both arm chains → it perturbs both end-effectors. Cleanest first cut: **direct head** (don't put it in either arm Jacobian); later ablate putting it in a shared task term. |
| Base height 31, base vel 32–34, torso yaw 35 | **Direct head** | Exactly the Fetch "base channels are direct" pattern. Legs are not policy DoFs (WBC handles them) — **do not factorize legs.** |

So: **two per-arm Jacobians (bimanual), Dex3 hands + waist + base on the direct head.**
The repo's single-chain `ActionJacobian` needs a bimanual extension (block-diagonal over
the two arm chains, each to its own `end_link`); `_FactorizedMixin._decode` already
supports a generic `action_indices` / `direct_indices` split, so the wiring is mostly a
new `KinematicSpec` + a two-chain Jacobian.

---

## 4. The mismatches that decide the experiment (ranked)

1. **Vision vs privileged state — the big one.** This repo's FK features are mostly
   *object-relative* (tcp/object/goal in each link's local frame) and require privileged
   object/goal pose. **PSI's obs has none of that** (RGB + proprio only). Consequences:
   - The **object-relative** FK features cannot be built from PSI's observation.
   - Use the repo's existing **robot-only FK feature mode** (`use_object_features=False`
     in `KinematicFeatureBuilder.fk_features`, `fk_generalized.py:124-137`): per-link
     frames + qpos/qvel from proprio — task-agnostic, vision-compatible. This is the
     honest transfer of the *feature* half.
   - The **factorization (decoder) half** needs only `qpos` (for `J`) → fully transferable
     from proprio. This is the part most likely to help and least blocked by vision.
   - You must add a **vision encoder** to the framework (it is currently state-only) to
     compare against PSI's DP on equal footing, OR run a **privileged-state sim ablation**
     (sim exposes object pose) to isolate the FK-feature contribution. Recommend doing
     both: vision for the headline number, privileged-state for the mechanism ablation.
2. **Absolute targets vs deltas.** Fetch path uses `pd_joint_delta_pos`; PSI uses
   **absolute** joint targets. The Jacobian maps a joint *delta* to a task delta, so
   factorize `Δq = q_target − q_current` (compute from proprio `qpos`), apply `J⁺`, then
   re-add `q_current`. Straightforward but must be implemented (the current decode assumes
   the kinematic channels *are* the delta).
3. **Bimanual coordination.** Two end-effectors, two task targets, a shared waist. The
   null-space term per arm now resolves a 7-DoF arm against a 3/6-D task — lots of
   redundancy, which is exactly where null-space control should pay off (good for the story).
4. **Dexterous hand ≠ parallel gripper.** The "gripper closure / jitter" narrative becomes
   "dexterous-hand closure." 14 hand DoFs predicted jointly and bounds-normalized with no
   special handling — *more* room for high-frequency artifacts, so the smoothness metrics
   (see review §3) are even more relevant here. The FK-contact-feature mechanism (finger
   link frames vs object) is natural but needs object pose (privileged/sim).
5. **No EMA, epsilon-prediction in PSI's DP.** Convenient: your factorized DP uses
   `sample`-prediction; to compare cleanly you should run the FK variant in PSI's exact
   regime (epsilon, no EMA) or hold both constant — don't reintroduce the
   `prediction_type` confound flagged in the code review.

---

## 5. The single cleanest experiment

**Claim to test:** "An FK-factorized action decoder improves diffusion policy on G1
manipulation (success and/or arm/hand smoothness) over PSI's DP, at matched
compute."

**Setup:**
- **Task:** one psi-data task where DP already partly works, e.g.
  `G1WholebodyTabletopGraspMP-v0` (MP-generated, less teleop noise) — or `…BendPickMP-v0`.
- **Baseline (exact):** reproduce PSI's DP from the spec in §1 (their UNet1D, DDPM-100,
  epsilon, obs_horizon 1 / pred 16 / exec 6, 40k steps, their normalization). This is the
  reference number; validate you can match their reported `L0|L1|L2` first.
- **Treatment:** same backbone + **FK-factorized arm decode** (two-arm Jacobians, hands +
  waist + base direct) + **robot-only FK features** appended to conditioning. **Match
  params/FLOPs** (grow baseline trunk if needed — do not let FK be the bigger model, the
  mistake flagged in the MS-HAB review).
- **Controls / ablations (factorial):** {±FK features} × {joint-space, J⁺-only,
  J⁺+null-space}; ±waist-in-Jacobian; vision vs privileged-state (sim) to isolate where
  any gain comes from.
- **Seeds:** ≥3 (PSI uses one seed 2026; you need variance).
- **Metrics:** SIMPLE success at `L0|L1|L2` **plus** the arm/hand smoothness suite from
  the code review (TV, jerk, hand-joint sign-change rate, spectral energy >fps/4),
  reported FK vs PSI-DP vs **expert demo** as the floor.

**Why it's credible:** it beats the *authors' own* published baseline on *their* benchmark
and data, with matched compute and the ablations that attribute the gain. If the FK
decoder helps the redundant bimanual arms and/or smooths the dexterous hand, that is a
clean, novel humanoid result; if it only helps via features, you'll know (and can say so).

**Effort estimate:** medium-high. Reusable: differentiable G1 FK (done), the factorized
decode logic (`_FactorizedMixin`), the diffusion agent skeleton. New: bimanual two-chain
`ActionJacobian` + G1 `KinematicSpec`, absolute-target delta handling, a vision encoder
in the framework (or a LeRobot data bridge into the existing trunk), and a LeRobot
data loader for psi-data. The vision encoder + LeRobot ingestion is the bulk of the work;
the FK/factorization port is comparatively small.

---

## 6. Risks specific to the humanoid claim

- **Vision blocker is real:** without object pose, the strongest (object-relative) FK
  features don't transfer; the contribution narrows to the *decoder* + robot-only features
  unless you add perception. Be explicit about this in the paper.
- **Decoupled WBC confound:** the base-velocity command is consumed by a separate RL
  controller; success depends on that controller, not just the policy — hold it fixed and
  identical across methods.
- **Bimanual Jacobian conditioning / singularities:** two arms + shared waist raises
  singularity and damping-sensitivity questions; reuse the existing
  `factorization_diagnostics` to report null-leak/task-recon for G1.
- **Don't overclaim "universal/generalized":** as in the MS-HAB review, the framework is
  still embodiment-hardcoded; a working G1 instantiation would be the *first* real evidence
  for the multi-robot claim — worth doing precisely for that reason.
