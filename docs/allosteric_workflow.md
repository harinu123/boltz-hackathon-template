# Allosteric Template-Matching Workflow

This guide walks through running the allosteric pocket steering pipeline that ships with the hackathon template. The workflow combines the pre-computed MSAs provided with the datasets, the `AllostericTemplateMatcher` heuristics, and the evaluation scripts so that you can reproduce the behaviour from the latest code drop with copy/paste commands.

## 1. Environment setup
Create and activate the recommended conda environment, then install Boltz in editable mode. Replace `<PATH_TO_REPO>` with the absolute path to your cloned fork if you are not already inside the repository directory.

```bash
cd <PATH_TO_REPO>
conda env create -f environment.yml --name boltz
conda activate boltz
pip install -e ".[cuda]"
```

> **Tip:** If you are on CPU-only hardware, drop the `[cuda]` extra. The workflow still runs but inference is significantly slower.

## 2. Fetch the hackathon datasets
Download and unpack the public hackathon bundle. This contains the allosteric/orthosteric (ASOS) validation set, precomputed MSAs, and baseline submissions.

```bash
wget https://d2v9mdonbgo0hk.cloudfront.net/hackathon_data.tar.gz
mkdir -p hackathon_data
tar -xvf hackathon_data.tar.gz -C hackathon_data
```

After extraction the layout that matters for the allosteric task is:

```
hackathon_data/
├── datasets/
│   └── asos_public/
│       ├── asos_public.jsonl      # task definitions
│       ├── msa/                   # per-target MSAs consumed by the matcher
│       └── baseline_submission/   # optional reference predictions
├── submission/                    # (optional) where you keep your own predictions
├── evaluation/                    # (optional) where you keep evaluation outputs
└── intermediate_files/            # (optional) scratch space between runs
```

## 3. Run allosteric predictions
The driver script wires in the `AllostericTemplateMatcher` and automatically applies pocket constraints and tuned Boltz-2 sampling configs when a confident template hit is found. Point the script at the ASOS JSONL file, the corresponding MSA directory, and choose output folders for predictions and evaluation artifacts.

To make copy/paste easier regardless of where you extracted the bundle, set an environment variable pointing at the unpacked directory. The example below mirrors a common layout such as `/home/ubuntu/hari/hackathon_data`.

```bash
export HACKATHON_DATA=/path/to/hackathon_data
mkdir -p \ \
  "${HACKATHON_DATA}/submission/asos_predictions" \ \
  "${HACKATHON_DATA}/intermediate_files/asos_tmp" \ \
  "${HACKATHON_DATA}/evaluation/asos_results"

python hackathon/predict_hackathon.py \
    --input-jsonl "${HACKATHON_DATA}/datasets/asos_public/asos_public.jsonl" \
    --msa-dir "${HACKATHON_DATA}/datasets/asos_public/msa/" \
    --submission-dir "${HACKATHON_DATA}/submission/asos_predictions/" \
    --intermediate-dir "${HACKATHON_DATA}/intermediate_files/asos_tmp/" \
    --result-folder "${HACKATHON_DATA}/evaluation/asos_results/"
```

Key outputs:

- `${HACKATHON_DATA}/submission/asos_predictions/<datapoint_id>/model_*.pdb` – ranked PDBs emitted for each datapoint
- `${HACKATHON_DATA}/intermediate_files/asos_tmp/` – intermediate YAMLs, raw Boltz outputs, and logs (safe to delete afterwards)
- `${HACKATHON_DATA}/evaluation/asos_results/combined_results.csv` – evaluation summary with ligand RMSDs for the allosteric subset and the full set

## 4. Evaluate existing predictions only
If you already have predictions organised in the submission layout (five ranked models per datapoint), rerun scoring without regenerating structures:

```bash
export HACKATHON_DATA=/path/to/hackathon_data
python hackathon/evaluate_asos.py \
    --dataset-file "${HACKATHON_DATA}/datasets/asos_public/asos_public.jsonl" \
    --submission-folder "${HACKATHON_DATA}/submission/asos_predictions/" \
    --result-folder "${HACKATHON_DATA}/evaluation/asos_results/"
```

The evaluator prints aggregate RMSDs to the console and refreshes the CSV/plots inside `${HACKATHON_DATA}/evaluation/asos_results/`.

## 5. Customising template matching
Advanced users can tweak the supplied matcher and template library:

- Templates live in `hackathon/templates/allosteric_templates.json`. Add or edit entries to extend the library.
- Programmatic access is provided through `boltz.pockets.AllostericTemplateMatcher`. You can import it inside your own experiments or modify the heuristics in `src/boltz/pockets/template_matching.py`.
- When you change the matcher or templates, rerun the prediction command above to test the updated behaviour.

## 6. Troubleshooting tips
- **No confident match logged:** The script falls back to baseline Boltz-2 settings and still writes predictions. Check the console output for confidence scores to diagnose template coverage.
- **Missing MSAs:** Ensure the `--msa-dir` flag points to the extracted `msa/` directory. Each entry must retain its relative path from the JSONL file.
- **Large temporary directory:** Remove `./asos_tmp/` between runs if disk space becomes tight.

You're ready to iterate on allosteric binders! Combine these commands with your own modifications in `prepare_protein_ligand` to explore new pocket steering strategies.
