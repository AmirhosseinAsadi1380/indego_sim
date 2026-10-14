# Results

A person walking on their own in the Indego: sagittal-plane model, no crutches or help, 30 ms reaction delay,
walking parameters tuned for the person (`HumanWalkParams`).

| person | exo | walked the full time | distance | output |
|---|---|---|---|---|
| T12 AIS D (incomplete SCI) | zero torque | 8/8 seeds x 30 s | 17.6-18.0 m | `logs/eval_own_walk_t12_zero.txt`, `own_walking_sci_t12_aisD_zero.mp4` |
| T12 AIS D (incomplete SCI) | zero torque | 2/2 seeds x 60 s | 37.3-37.4 m | `logs/eval_own_walk_t12_60s.txt` |
| healthy | zero torque | 8/8 seeds x 30 s | 20.3-20.5 m | `logs/eval_own_walk_healthy_zero.txt`, `own_walking_healthy_zero.mp4` |
| healthy | transparent | 8/8 seeds x 30 s | 19.4-20.0 m | `logs/eval_own_walk_healthy_transparent.txt` |

Reproduce:

```bash
python examples/01_own_walking.py --video                    # T12 AIS D (--subject healthy)
python tools/diag/eval_own_walk.py '{}' 0,1,2,3,4,5,6,7 30   # '{"subj":"healthy"}', '{"ctrl":"transparent"}'
```

`model_overview.png`: the model (`python tools/render_overview.py`).
