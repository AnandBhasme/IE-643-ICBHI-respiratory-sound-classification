**What this demo does.** It classifies one respiratory cycle into normal, crackle, wheeze or both, using an Audio Spectrogram
Transformer (AST) fine-tuned on the ICBHI 2017 training split. It also generates synthetic cycles with SpecDiff, a diffusion model
that works on the classifier's own log-mel input, and compares them with real cycles from the same class and stethoscope.

**Scores.** Every score is the official ICBHI Score, (Se + Sp) / 2, on the official test split (2,756 cycles), last training epoch.
Final models: T_K3 59.57 ± 0.64 with the prior correction, M0 58.46 ± 0.96 (10 seeds each). Intensive-phase candidates and
ensembles: 3 seeds each, from the intensive assessment. The score table in the About tab says which models are installed here.

**Which models are not here.** The earlier demo also offered phase-1 and phase-2 checkpoints (E1, E2, E3, E7a). Their scores were
selected on the test set (best epoch), so they are left out of this demo.

**Ensembles.** An ensemble averages the softmax outputs of its members with equal weight. Members are loaded one at a time, so the
memory use is that of one model. Ens-AST has 15 members: C0 to C4, seeds 1 to 3. Its C3 members duplicate C1 (a scheduler bug erased
C3's layer-wise learning rates), so C1 counts twice.

**Prior correction.** Models trained with a balanced synthetic mix (T_K3) predict as if the classes were balanced. The correction
multiplies each probability by the real training prior divided by the mix prior, with no tuned parameter. It moves the operating
point towards specificity.

**The generator.** SpecDiff generates the 128-bin log-mel the AST reads, conditioned on class, stethoscope and cycle duration, with
classifier-free guidance on the class. The audio player inverts the log-mel with Griffin-Lim: it is approximate and for listening
only. The nearest real cycle is taken from a small set of real training cycles shipped with the demo.

**Limits.** The gain of T_K3 over the real-only model is a specificity gain (wheeze and "both" recall fall). It equals the gain from
shifting the real-only model's threshold to the same specificity (see docs/POSTHOC_DIAGNOSTICS_LOOK10.md). ICBHI has few patients
and the split is recording-level. Do not use this tool for diagnosis.
