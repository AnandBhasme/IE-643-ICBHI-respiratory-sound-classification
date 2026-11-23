# What was tested, and what was not (23 Nov 2026)

Tested on PRAJNA (one NVIDIA L40S per job, PyTorch 2.0.1, Python 3.11):

| Check | Result |
|---|---|
| Unit tests (`tests/`, CPU) — split counts, prior weights, shipped probabilities reproduce 58.46 ± 0.96 / 59.57 ± 0.64 / 8 of 10 wins, U-Net + DDIM sampler on CPU, log-mel → audio round trip (correlation > 0.9) | 5 passed |
| Same tests plus weight tests (`tests/test_demo_weights.py`): packaged fp16 classifier reproduces the stored test probabilities (max difference < 0.05, argmax agreement ≥ 90 %), generation + audio | 7 passed |
| Clean copy with **no** `artifacts/` and no audio: `python scripts/reproduce.py --stages check evaluate` | tables and figures reproduced exactly, including the bias sweep (−0.1044 [−0.6479, +0.4390], p = 0.674) |
| End-to-end `python scripts/reproduce.py --stages all --quick` (1 seed, 1 epoch, 600 + 200 generator steps, tiny pools): M0 → generators → pools → quality gates → teacher → M1b / M3 / K3 / T_K3 → tables | completed, 25 min 44 s (`docs/quick_run_log.txt`). Its numbers (≈ 53 Score) are meaningless; it only proves the wiring |
| Demo CLI: `classify`, `generate` (GPU and CPU), `check` | works; GPU 50 steps in seconds, CPU 10 steps ≈ 13 s, ≈ 1 min for 50 steps. `check --n 12` with the M0 probe recognised normal 100 %, crackle 83 %, wheeze 92 %, both 92 % of generated cycles |
| Gradio app: built, launched, three endpoints (`/classify_upload`, `/generate`, `/batch_check`) listed through `gradio_client` | works; the page was **not** inspected in a browser |
| `--backend slurm --dry-run` | job graph printed correctly (dependencies, one GPU per job, no `--exclusive` / `--nodelist`) |

A bug found and fixed during testing: `SpecDiff/model.py` attention raised "query: last dimension must be contiguous" for small
batches on CUDA (torch 2.0.1); `.contiguous()` added, values unchanged.

**Not verified / open**
* The full reproduction (≈ 30 classifier runs, two generators; days of GPU time) was not re-run. The shipped probabilities are the
  original runs; a fresh run will differ by seed noise.
* `scripts/download_data.py`: the ICBHI and AST URLs are the official ones, but the cluster could not reach the ICBHI server in a quick
  check, so the automatic download is untested (manual placement is documented).
* `weights/weights_manifest.json` has empty URLs until the three files are uploaded to a GitHub release.
* `AUDIO`: the demo audio is an approximate Griffin-Lim inversion, not a vocoder.
* Nothing was committed or pushed to GitHub.
