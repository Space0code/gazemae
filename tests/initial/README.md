# Initial ETRA/MIT evaluation run

Generated on 2026-02-26.

## Environment
- Conda env: `trust-me`
- Output root: `tests/initial/`
- Generated data root used by code:
  `GAZEMAE_GENERATED_DATA_ROOT=../tests/initial/generated-data/`

## Commands run

From `gazemae/`:

```bash
python evaluate.py --model-vel vel-i8528 --signal-type vel -hz 500 -vt 2 --corpora ETRA2019
python evaluate.py --model-vel vel-i8528 --signal-type vel -hz 250 -vt 2 --corpora MIT-LowRes
```

## Key results

### ETRA2019 (`logs/eval_etra.log`)
- `Biometrics` (subject ID): `Acc: 0.7292`
- `ETRAStimuli` (Blank/Natural/Puzzle/Waldo): `Acc: 0.5437`

### MIT-LowRes (`logs/eval_mit.log`)
- `Biometrics_MIT_LR` (subject ID): `Acc: 0.1756`

## Artifacts
- `generated-data/ETRA2019-data.pickle`
- `generated-data/MIT_LowRes-data.pickle`
- `logs/eval_etra.log`
- `logs/eval_mit.log`
