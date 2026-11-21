# Requirement 9 — LSTM and Transformer, latent space, attention, step / sigmoid / tanh

## Why this module exists

Everything so far looked at **one frame at a time**. A lesson is a sequence: attention drifts, recovers, drops again.
This module tests the two standard sequence models on **real classroom video**:

- **LSTM**: reads the frames one after another and keeps a memory (hidden state).
- **Transformer**: looks at all frames at once and decides, by **attention**, which ones matter.

It also opens the models up: what attention looks at, what the **latent space** (the model's internal numbers) looks
like, and what happens when the LSTM's inner non-linearity is **tanh, sigmoid or a step function**.

Code: `src/smartattend/sequence/classroom_seq.py` (this step), `lstm_trend.py`, `transformer_agg.py`, `latent_viz.py` ·
tests: `tests/test_classroom_seq.py` (7) · Colab: `notebooks/09_sequence_models_colab.ipynb` (CPU is enough).

## Run it

```bash
python -m smartattend sequence          # about 5 minutes on a laptop CPU
python -m pytest tests/test_classroom_seq.py
```

Results: `reports/eval/sequence_models.json`. Figures: `reports/figures/sequence/`.

## The task and the data

**Input per frame** = what the app produces: fine-tuned YOLOv8n finds the students, ResNet18 gives each one
probabilities for attentive / distracted / inactive (Requirement 7). Seven numbers summarise a frame (mean
probabilities, share predicted attentive / distracted, number of students, spread).
**Target** = the **labelled** share of attentive students in that frame.
**History** = the last 8 frames.

| role | videos | frames | labelled attentive share |
|---|---|---|---|
| training | taimoor-class-2 | 118 | 0.94 |
| training | taimoor-class-3 | 20 | 0.23 |
| training | taimoor-class-4 | 19 | 0.85 |
| early stopping | taimoor-class-5 | 22 | 0.87 |
| **test** | taimoor-zaeem (other classroom) | 85 | 0.36 |

**Why the training videos come from the *validation* classroom:** the ResNet18 was trained on the training classrooms,
so its outputs there are unrealistically good. On the validation classroom its outputs look like they will in real
use. Hyperparameters were fixed in advance; nothing was tuned on the test video; every learned model ran with 5 seeds.

## 9a. LSTM vs Transformer over time

**The LSTM cell** (`FlexLSTM`, written by hand so the non-linearity can be swapped):
```python
i, f, g, o = (W_x x_t + W_h h).chunk(4)           # four gates from the current frame and the memory
i, f, o = sigmoid(i), sigmoid(f + 1), sigmoid(o)   # input, forget, output gates (values 0..1)
c = f * c + i * act(g)                             # cell state: keep some old memory, add some new
h = o * act(c)                                     # hidden state passed to the next frame
```

**The Transformer block** (`TemporalTransformer`):
```python
h = embed(x) + position                            # 8 frames -> 8 vectors; position says which frame is which
a, weights = attention(h, h, h)                    # every frame looks at every frame; weights sum to 1
h = norm(h + a);  h = norm(h + feed_forward(h))
y = out(h[:, -1])                                  # read the answer from the current frame
```

**Results on the test video** (MAE = average error of the share, lower is better; correlation = does it follow the
ups and downs, higher is better):

```
model                                                MAE    correlation
current frame only (app output)          0.326           +0.61
moving average of last 8 frames          0.324           +0.64
ridge regression on the window           0.218           +0.70
LSTM (tanh cell)                         0.414 ± 0.057   +0.64 ± 0.02
LSTM (sigmoid cell)                      0.503 ± 0.021   +0.43 ± 0.34
LSTM (step cell)                         0.455 ± 0.051   +0.36 ± 0.24
Transformer (1 block, 2 heads)           0.346 ± 0.165   +0.34 ± 0.51
Keras LSTM (TensorFlow, tanh)            0.524 ± 0.043   +0.42 ± 0.26
```

**Check on new videos without touching the test set** (leave one video out, MAE on the held-out video):

```
held-out video (frames, true share)     current frame only    ridge regression    LSTM (tanh)    Transformer
taimoor-class-2 (118, 0.94)                          0.057               0.135          0.184          0.137
taimoor-class-3 (20, 0.23)                           0.678               0.641          0.750          0.658
taimoor-class-4 (19, 0.85)                           0.115               0.221          0.202          0.146
taimoor-class-5 (22, 0.87)                           0.110               0.333          0.294          0.251
mean of the 4 videos                                 0.240               0.332          0.357          0.298
```

**What this shows (an honest negative result):**
- **The learned sequence models do not beat the simple baselines.** On new videos the app's raw per-frame output is
  the best (0.240); LSTM, Transformer and ridge are all worse.
- **Ridge won on the test video (0.218) but loses in cross-validation (0.332),** so that win is not reliable.
- **Why:** there are only **three training videos**, and one of them (118 of 157 windows) is 94 % attentive. The
  networks learn "the answer is high" and carry it to a class that is 36 % attentive
  (`classroom_lstm_transformer.png`: the Transformer's output is an almost flat line at 0.83).
- **The LSTM (tanh) is the most stable learned model:** its trend correlation (0.64 ± 0.02) matches the moving
  average, while the Transformer swings between seeds (0.34 ± 0.51). With little data the LSTM's built-in assumption
  ("recent frames matter, in order") helps; the Transformer has to learn that from data it does not have.
- **Sequence models cannot repair wrong inputs.** On taimoor-class-3 (23 % attentive) the per-student classifier
  itself says "attentive" far too often (error 0.68), and every model built on it inherits that.
- **Keras and PyTorch LSTMs behave alike** (both worse than the baselines; differences are within the seed spread).

**Decision for the app:** keep the simple per-frame share with a moving average for the trend chart. A learned
sequence model needs many more labelled videos, from many classes.

**Earlier result on one person's video** (`lstm_trend.py`, 293 training windows, forecasting the next 12 frames):
LSTM tanh 0.155 ± 0.008, sigmoid 0.111 ± 0.013, step 0.174 ± 0.058, against **0.063 for "repeat the last value"**.
The same conclusion on different data: with small datasets, a simple rule beats the LSTM.

## 9b. Attention

**Over time** (`classroom_attention.png`, left): the Transformer's attention from the current frame to the 8 frames is

```
t-7 ... now:  0.124 0.125 0.125 0.125 0.124 0.124 0.124 0.128     (uniform = 0.125)
```

It is **almost exactly uniform**: the model learned no preference for any frame, so it is simply averaging. This is
what attention looks like when there is too little data to learn from, and it explains the flat output.

**Across students** (right): a second Transformer reads the *set* of students in one frame (each student = 3
probabilities + box position and size) and a summary token attends to the students.

```
model                                                MAE    correlation
mean p(attentive)                        0.293           +0.64
share predicted attentive                0.326           +0.61
DeepSets (no attention)                  0.498 ± 0.012   +0.65 ± 0.01
Transformer over students                0.477 ± 0.014   +0.61 ± 0.03
attention per student ranges from 0.78x to 1.01x the uniform share
attention weight vs student feature (correlation): p(attentive) -0.53, p(distracted) +0.54, y +0.37, width +0.30
```

- Attention is **mild** (between 0.78× and 1.01× an equal share) but **not random**: it leans toward students the
  classifier thinks are **distracted** (+0.54) and toward larger, lower boxes (front rows). Those are the students
  who change the class's attentive share.
- It does **not** improve the estimate: both set models follow the trend as well as the plain average (correlation
  0.61–0.65) but are far off in level (0.48–0.50 vs 0.29), for the same reason as above.
- On **simulated** frames, where training data is unlimited, the same Transformer did help (error 0.111 vs 0.174 for
  the plain average, earlier stage). The architecture works; the real data is too small.

## 9c. Latent space

`classroom_latent_space.png`.

- **ResNet18's 512 internal numbers for each test student** (left, squeezed to 2-D with PCA): attentive and distracted
  students **overlap heavily**. A 5-nearest-neighbour vote in this space is right 69.8 % of the time, against 55.7 %
  for always guessing the largest class. So the features carry real but limited information, which is why
  Requirement 7's classifier reaches macro-F1 0.46 and not higher. The "inactive" students are scattered among the
  others: the network has no separate region for them.
- **The LSTM's 16 memory numbers across the test video** (right): **one direction holds 95 % of the variation**, and it
  follows the labelled attentive share (|r| = 0.68). The memory behaves like a single smoothed "attention level"
  dial; the path through the video moves between a low-attention cluster and a high-attention cluster.
- An earlier stage also trained a small variational autoencoder on face frames (`reports/experiments/sequence/vae_faces.png`,
  `latent_space.png`): its 8-number latent space lets you slide smoothly between faces.

## 9d. Step vs sigmoid vs tanh inside the LSTM

`lstm_activations.png`.

| cell non-linearity | classroom MAE | classroom correlation | one-person forecast MAE | gradient reaching 20 frames back |
|---|---|---|---|---|
| tanh (standard) | 0.414 | +0.64 ± 0.02 | 0.155 ± 0.008 | 1.0e-02 |
| sigmoid | 0.503 | +0.43 ± 0.34 | 0.111 ± 0.013 | 1.1e-03 |
| step (trained with a straight-through estimator) | 0.455 | +0.36 ± 0.24 | 0.174 ± 0.058 | 5.5e-03 |

- **tanh** is centred on zero (outputs −1 … 1), so the memory can go up *and* down, and it passes back the most
  gradient: **about 10× more than sigmoid** after 20 frames. It is the most stable across seeds on both tasks.
- **sigmoid** only outputs 0 … 1, so the memory can only be added to, and its slope is at most 0.25, so the learning
  signal fades fastest. It is unstable on the classroom task; it happened to do best on the one-person forecast,
  where the target itself lies between 0 and 1.
- **step** has slope 0 everywhere, so by itself it cannot be trained at all (Requirement 4). It only trains here
  because of a trick, the *straight-through estimator* (pretend the slope is 1 near zero when going backwards). The
  result is the noisiest of the three (largest spread on the one-person task).
- All three lose to the simple baselines on this data, so this is a comparison of **behaviour**, not a claim that
  one of them is good enough.

## Checklist
- [x] LSTM (hand-written cell) and Transformer on real classroom video, against three simple baselines, 5 seeds
- [x] leave-one-video-out check; Keras (TensorFlow) LSTM alongside PyTorch
- [x] attention over time and across students, measured and drawn on a real frame
- [x] latent space of the ResNet18 embedding and of the LSTM memory
- [x] step / sigmoid / tanh: accuracy, stability and gradient flow through time
- [x] negative results reported as such, with the reason and the decision for the app
- [x] 7 tests in `tests/test_classroom_seq.py`
