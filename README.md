# Version 2 Sensitivity Package

This version keeps the experiment as a one-parameter-at-a-time sensitivity test,
but expands the parameter set and automatically produces change-efficiency
analysis.

## Folder Layout

```text
run_sensitivity.py
configs/
  representation.yaml
  compression_method.yaml
  robustness.yaml
  pipeline.yaml
analysis/
  summarize_results.py
outputs/
  ...
```

## Run One Stage

Use the `speech311` environment:

```bash
cd "/Users/home2/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Summer project 2/version 2"
/opt/anaconda3/envs/speech311/bin/python run_sensitivity.py --config configs/representation.yaml
```

Change the config path to run the other stages:

```bash
/opt/anaconda3/envs/speech311/bin/python run_sensitivity.py --config configs/compression_method.yaml
/opt/anaconda3/envs/speech311/bin/python run_sensitivity.py --config configs/robustness.yaml
/opt/anaconda3/envs/speech311/bin/python run_sensitivity.py --config configs/pipeline.yaml
```

## Outputs

Each stage writes:

```text
raw_results.csv
summary_by_value.csv
change_efficiency.csv
recommendations.md
plots/
```

`raw_results.csv` has one row per audio file per tested parameter value.
`summary_by_value.csv` averages accuracy, WER, compression rate, waveform SNR,
and mel MSE for each value.

`change_efficiency.csv` compares each value with that parameter's baseline:

```text
accuracy_drop = baseline_accuracy - average_accuracy
compression_gain = average_compression_rate - baseline_compression_rate
change_efficiency = compression_gain / accuracy_drop
```

`recommendations.md` gives suggested worth-testing ranges and increment advice.

## Scaling Up

The configs currently use:

```yaml
files_per_label: 2
labels: ["yes", "no"]
```

For a larger run, increase `files_per_label` or add more labels.
