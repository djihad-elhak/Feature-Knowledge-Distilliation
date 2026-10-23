# Feature Distillation of a Mobile VLM on GQA

**MobileVLM V2 7B (teacher) → MobileVLM V2 3B (student)**

My part of the team project on knowledge distillation for mobile VLMs: **feature (representation) distillation**. The student is trained to answer GQA questions correctly and to reproduce the teacher's hidden states.

## Output (results, analyses and the feature model)

https://www.kaggle.com/datasets/djihadelhakbelkhir/kd-vlm-continue

## Notebooks

| Notebook | Role |
|---|---|
| `GQA_preparation_and_sampling.ipynb` | Builds the GQA sample (20,000 train images, 1,000 dev, 398 testdev, image-disjoint, seed 42) |
| `feature-kd-vlm.ipynb` | Teacher feature cache, student training, evaluation. Stopped at step 4500/5661 (Kaggle 12 h limit) |
| `feature-kd-vlm-continue.ipynb` | Resumes training, evaluates, exports the merged model, runs the teacher-vs-student analysis |

## Method

- Teacher (frozen, fp16) hidden states are cached once; the student is MobileVLM V2 3B + LoRA (r=16, on q/k/v/o).
- Loss = answer cross-entropy + MSE between student and teacher states at layers 16 and 32, on two signals: the **answer position** and the **visual tokens** (144 tokens pooled to a 4×4 grid).
- A small adapter maps student states (2560-d) to teacher space (4096-d). It is used only during training.
- Run `D_qv` (both signals), 1 epoch, 90,561 questions, 5,661 steps, 2×T4.

## What the output contains

| Item | What it is |
|---|---|
| `runs/D_qv/merged_fp16/` | The **feature-distilled student**: standalone fp16 model, loads with MobileVLM's `load_pretrained_model` (no LoRA, adapters or teacher needed), plus a model card |
| `results.json`, `train_log.csv`, `env.json` | Final scores, training curves and the environment record for `D_qv` |
| `baselines/` | Predictions, confidences and scores of the teacher and the untouched student on `dev_mini` and `dev` |
| `comparison_dev.csv`, `per_type_dev.png` | Accuracy overall and per question type |
| `breakdown_*.csv`, `breakdown.png` | Accuracy by answer type, answer frequency, question length, semantic type |
| `agreement.csv` | How often the student gives the same answer as the teacher, and where errors overlap |
| `calibration.csv`, `calibration.png` | Confidence quality (ECE, reliability diagram, accuracy-coverage) |
| `cka.csv`, `cka.png` | Similarity of student and teacher representations, layer by layer |
| `efficiency.csv` | Parameters, FLOPs, memory, latency |
| `bootstrap_dev.csv` | Significance tests (paired bootstrap) |
| `learning_curves.png` | Training loss curves |
| `qualitative.png` | Examples where teacher and student disagree |
| `scorecard.csv` | One summary table |

## Main results (`dev`, 13,174 questions)

| Model | Accuracy |
|---|---|
| Teacher (7B) | 71.67% |
| Student 3B, zero-shot | 69.33% |
| Student 3B, feature KD (`D_qv`) | 69.41% |

- The student keeps 96.8% of the teacher's accuracy with 2.3× fewer parameters, 2.5× fewer FLOPs and about 1.9× lower latency.
- Feature KD clearly aligned the representations (CKA with the teacher: visual tokens 0.48 → 0.92, answer position 0.74 → 0.82).
- Accuracy did not change significantly (+0.08 pts, 95% CI [−0.50, +0.72]). Calibration got slightly worse (ECE 0.034 → 0.042).
- The CE-only control (`A_ce`) has not been run yet, so the effect of the teacher signal itself is not isolated.
