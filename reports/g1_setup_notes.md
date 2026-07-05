# G1 setup notes (for the methods section)

## Iteration 1 (archived): G1 43-DoF with [-π, π] joint sampling

Initial G1 setup used the **43-DoF G1 URDF** (full body + 16 finger DoFs)
generated via:

```bash
python -m src.scripts.gen_robot_config \
    --robot_name g1 --urdf robot_models/urdf/g1.urdf \
    --output configs/robots/g1.yaml \
    --write_dataset_configs --joint_range pi
```

The `--joint_range pi` choice matched UR5's convention (uniform [-π, π] per
joint regardless of URDF mechanical limits). For UR5 (6-DoF arm, 80 cm reach)
this was harmless — the swept workspace stayed manageable. For G1 with
43 DoF, [-π, π] sampling produces predominantly **physically impossible
configurations**: knees bent backward (URDF limit ≈ [0, 2.85], not [-π, π]),
fingers folded through the palm, hips with feet through the torso, etc.

The forward kinematics is mathematically well-defined for any qpos, but the
swept link-pose volume is enormous compared to the URDF-limit workspace, and
the network is implicitly being asked to fit FK over a domain that's mostly
non-physical.

### Iteration 1 results (3 backbones × 3 seeds × 100k iters, fixed root)

| Backbone | Final loss (mean) | Best-ever | Status |
|---|---|---|---|
| RodriguesFK (Rod) | ~8.3e-2 | ~7.4e-2 | barely above iter-0 baseline (~0.13) |
| ScrewFK_pc1 | ~3.4e-2 | ~1.9e-2 | learning, but slowly |
| ScrewFK_pc1_dmask | ~7.7e-2 | ~5.2e-2 | barely above iter-0 baseline |

### Diagnosis

Three signals from these numbers:

1. **Absolute loss scale 5-7 orders worse than LEAP/Stretch** (1e-2 vs 1e-7).
   The model is at the "barely better than predicting the mean" floor;
   training has not converged.
2. **First-iter losses ~0.12-0.15** uniformly. After 100k iters, Rod and dmask
   only manage 1.5-1.8× improvement → essentially stuck.
3. **Relative ordering pc1 >> dmask ≈ Rod**, *opposite* of the LEAP capacity-
   sweep prediction. Consistent with undertraining: pc1 has more raw
   parameters (0.79M vs Rod's 0.64M), and in the undertrained regime extra
   capacity helps regardless of whether it's geometrically useful. The
   wasted-capacity-on-revolute argument requires the converged regime to
   surface.

The dmask ≈ Rod tie within seed noise (8.5e-2 vs 9.3e-2) does confirm the
mask is structurally correct on all-revolute G1 (mask = all zeros ⇒ same
behavior as RodriguesFK). The implementation is fine; the experiment setup
isn't.

### Logs preserved

`reports/training_logs/_archive_g1_43dof/g1_seed*_*.txt` — 9 files,
3 backbones × 3 seeds at 100k iters each. Useful as evidence in the methods
section that we attempted [-π, π] + 43-DoF before pivoting.

## Iteration 2 (active): G1 29-DoF with URDF mechanical limits

Two changes:

1. **29-DoF URDF** (legs + 3-DoF waist + arms, no hands). Removes 16 finger
   DoFs which inflate output dimension without probing the dmask hypothesis
   (fingers are all small revolute joints with similar geometry to the wrist).
2. **`--joint_range urdf`** in `gen_robot_config.py`. Each joint samples its
   URDF mechanical limits instead of uniform [-π, π]. Workspace shrinks by
   orders of magnitude to physically realisable configurations.

Regen command:

```bash
cp ~/path/to/g1_29dof.urdf robot_models/urdf/g1.urdf
python -m src.scripts.gen_robot_config \
    --robot_name g1 --urdf robot_models/urdf/g1.urdf \
    --output configs/robots/g1.yaml \
    --write_dataset_configs --joint_range urdf
```

### Predicted iteration-2 outcome

In the converged regime the wasted-capacity-on-revolute finding should
re-assert itself:

| Predicted | Rod | pc1 | pc1_dmask |
|---|---|---|---|
| Final | ~1e-5 to 1e-4 | possibly worse than Rod | ≈ Rod |
| Best-ever | ~1e-6 (LEAP scale) | similar | similar |
| Ordering | dmask ≈ Rod < pc1 | | |

If iteration-2 **also** has pc1 winning, the next thing to investigate is
whether 100k iters is enough at 29-DoF or whether the chain depth itself
warrants longer training (the kinematic chain depth from base to fingertip
on G1 is much larger than UR5/LEAP, even with hands removed).

## Why pc1 winning in iteration 1 doesn't refute the dmask hypothesis

The dmask hypothesis predicts: **at convergence**, on revolute robots, the
d-basis kernel slots add optimization noise without contributing useful
signal, so dmask (no d-basis) ≈ Rod < pc1 (penalized for wasted capacity).

In the *undertrained* regime, this prediction doesn't apply. With 100k iters
on a 43-DoF system whose loss is still descending, the comparison is
dominated by raw parameter capacity (pc1 wins because it has more params,
not because the d-basis is useful). The architecture-level finding can only
be evaluated once training converges, which iteration 2 (smaller workspace,
fewer DoFs) is designed to enable.
