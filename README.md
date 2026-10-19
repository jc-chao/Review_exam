# Exam Concepts: Folders 00–03

All concepts from the core notebooks, with every answer-sheet Exercise folded in under its notebook.
Optional, Guess and Further Reading material is left out, matching the exam scope.

**How to read the exercises:** each one gives the question, then a revision answer built from your own answer sheet.
Lines marked **⚠ Fix** are places where your written answer was wrong or incomplete; learn the corrected version.

Sources: [02-neural-networks/](02-neural-networks/) NB1–5 + [answer-sheet.md](02-neural-networks/answer-sheet.md), [P1](03-from-notebook-to-prod/P1_config-envvars-logging/P1_config-envvars-logging.ipynb), [P2](03-from-notebook-to-prod/P2_containerisation/P2_containerisation.ipynb), [P3](03-from-notebook-to-prod/P3_data-ops/P3_data-ops.ipynb). (00-introductions is only an environment check: load a CSV with pandas, plot it, and gitignore `data/`.)

---

## The one theme across the whole week

1. **State the problem** as input, output and metric.
2. **Get a baseline before building anything.** That means guessing / majority class, logistic regression, naive or seasonal-naive.
3. Build the model, then **compare it with the baseline on the same test set and the same metric**.
4. Ask whether the gain was **worth the cost** (training time, inference time, size).
5. Find **where it fails**: per-class recall, the confusion matrix, error plots.

**Test set discipline:** the test set never updates weights. But once you change knobs because of its score, it is helping you choose a model and is no longer independent. Tune on a **validation** set, and look at the test set once, at the end.

---

# Part A: Neural Networks (02-neural-networks)

## NB1: Foundations, one line of the training loop at a time

### Concepts
| Concept | One-liner |
|---|---|
| **Model** | A function with weights. One neuron = `w1·x + w2·y + b`, then group 1 if > 0. |
| **One neuron's limit** | Whatever the weights, it can only draw a **straight** boundary. On the two moons it stalls at about ⅓ wrong. |
| **Why a loss, not a count** | A wrong-count is a staircase: nudging a weight usually changes nothing, so there's no "downhill" direction. A smooth loss has a slope everywhere. |
| **Sigmoid** | Squashes any score into 0..1, read as "confidence it's group 1". |
| **Log loss (BCE)** | `−mean(y·log p + (1−y)·log(1−p))`. Tiny when confident and right, **huge when confident and wrong**. |
| **Gradient** | The slope of the loss for each weight. Together these slopes form the gradient. |
| **Gradient descent** | `w ← w − learning_rate × slope`. Repeat. |
| **Learning rate** | Step size. A *hyperparameter* chosen before training and never learned. |
| **Backpropagation** | Computes all the slopes in **one backward pass** with the chain rule, instead of nudging each weight (one data pass per weight, which can't scale to millions of weights). |
| **Activation (ReLU, Tanh)** | A non-linearity between layers. Without it, a weighted sum of weighted sums is still one weighted sum, so the boundary stays straight. ReLU = `max(0, x)` (piecewise-straight bends); Tanh = smooth curves. "Deep" = more layers with activations between them. |
| **Hidden width** | Neurons in the hidden layer, *not* input features. Too few → **underfit** (HIDDEN=3 leaves ~20 wrong). More = more **capacity**, but a bendy boundary alone doesn't prove overfitting; you need held-out data to tell. |
| **Batch / epoch** | Batch = the points used for one update. Epoch = one full pass over the training data. Updates per epoch = ⌈N / batch_size⌉. |
| **Optimiser** | The rule that turns gradients into updates. SGD: `−lr·grad`. **Adam** keeps running averages of gradients and squared gradients and **rescales each weight's step**, so its usual lr is ~10× smaller than SGD's. |
| **Train vs test accuracy** | Train = fit on points used for updates. Test = estimate on new points. A big gap = **overfitting**. |
| `model.train()` / `model.eval()` | Switch the mode for layers like dropout and batchnorm. They don't choose data or update weights. |
| `@torch.no_grad()` | No gradients tracked during evaluation, which saves memory and time. |
| **BCEWithLogitsLoss** | Log loss that takes raw scores ("logits") and applies the sigmoid internally. |

### The training loop (know every line)
```python
model.train()
for x, y in loader:
    optimizer.zero_grad()          # grads ACCUMULATE in PyTorch unless cleared
    loss = criterion(model(x), y)  # forward pass + score
    loss.backward()                # backprop: slope for every weight
    optimizer.step()               # update weights by the optimiser's rule
    total_loss += loss.item() * len(x)   # weight by batch size (last batch may be smaller)
```

### Exercises

**Ex 1.3. Predict the loss curve at `LEARNING_RATE = 20`, then explain any difference.**
Each step is so big it overshoots the minimum, so the loss bounces or spikes instead of falling smoothly. The prediction matched the plot.
For contrast: lr 0.01 is still falling at step 100 (too slow), and lr 5 falls fast with a wobble.

**Ex 1.5. Batch size 1 needs the fewest epochs but takes the most wall-clock time. Why both? Which size for 1M points?**
- *Fewest epochs:* it makes 200 updates per epoch instead of 1, so it moves downhill many more times per pass.
- *Most time:* each update is tiny, sequential work, so there's no vectorisation or parallelism over a batch. Per-update overhead is paid 200×.
- *For 1M points:* use **mini-batches** (e.g. 32–512, often a power of 2). That gives many updates per epoch, efficient vectorised maths, and gradients that are less noisy than single points.

**Ex 1.6. A colleague copies SGD's learning rate into Adam and the loss diverges. What went wrong?**
⚠ Fix: your answer argued that Adam is *faster at tiny learning rates*. The question is about **divergence**. The point is that **the same number means different things under each rule**:
- SGD's step = `lr × gradient`, so it scales with how big the gradient is.
- Adam's step ≈ `lr × m̂/√v̂`. Dividing by the running gradient size normalises it, so **each weight moves roughly `lr` per step regardless of gradient size.**
- So an lr that is fine for SGD (0.1, where small gradients keep steps small) makes Adam take full-size 0.1 steps on every weight. It overshoots and diverges (the Try showed Adam at 0.1 bouncing).
- Lesson: re-tune lr per optimiser. Adam's typical default is 1e-3, roughly 10× smaller than SGD's. Never compare optimisers at the same lr.

**Ex 1.7. At 500 epochs the accuracies separate (train 93%, test 89%). What does the gap mean, why does test count, and what one thing would you change?**
- The gap means the model has started **overfitting**: it is learning noise and quirks of the 200 training points that don't generalise.
- Test counts because it is data the model **never trained on**, so it estimates performance on new data. Training accuracy is optimistic by construction.
- ⚠ Fix: you didn't name the one change. Pick one: **train for fewer epochs / early-stop** on a validation curve, or use a smaller HIDDEN, or add regularisation. Also move repeated tuning onto a **validation set**.

---

## NB2: Computer Vision Basics (images as data, and a number to beat)

### Concepts
- **Problem framing:** input = one 28×28 RGB cell image; output = one of 8 cell types; metric = accuracy.
- **Image as numbers:** `(3, 28, 28)` = 2,352 numbers. A batch is `(64, 3, 28, 28)`. NB1's sample was `(2,)` and its batch `(200, 2)`. **Same DataLoader, different x shape.**
- `permute(0,3,1,2)`: NumPy gives channels-last `(N,H,W,C)` and PyTorch wants channels-first `(N,C,H,W)`. `/255` scales to 0..1 for stable training. Labels use `.long()` (int64, which CrossEntropyLoss requires).
- **Class imbalance:** the biggest class has ~3× the smallest. Remember this for "guessing".
- **Shuffling:** `shuffle=True` on the train loader only. Each image is independent, so order carries no information. The test loader keeps a fixed order (order doesn't change the score).
- **EDA** = you choose a question and investigate it with a plot or calculation. Use **train_set only**; never pick preprocessing because it makes the test set look easier. EDA continues after modelling: when a model confuses classes, look at those images.
- **Two baselines:**
  1. **Guessing** (always the majority class "neutrophil") → **19.5%**. This is the floor.
  2. **Logistic regression on flattened pixels** → **77.2%** (100 iters). It is one neuron per class, with straight boundaries in 2,352-D space. It treats pixels as unrelated columns.
- `MAX_ITER` trade-off: 10 iters loses several points; 300 iters gains a couple. **Time moves far more than score.**
- If the network doesn't beat the baseline: either the cheap model is good enough, or the network's pipeline is broken. Both are useful to know.

### Exercises

**Ex 2.1. Describe the task in one sentence for a client, and name a way accuracy is high but the sorter is useless.**
- "The computer looks at a photo of one blood cell and puts it into one of eight bins by cell type, so the technician checks bins instead of sorting from scratch."
- With imbalance, a model can score high accuracy while almost never finding a **rare but clinically critical** class. E.g. 99% of cells are one type, the model always predicts it, and it gets 99% accuracy with 0% recall on the cells that matter.

**Ex 2.2a. Why is shuffling harmless here? Give a dataset where shuffling before the split misleads you.**
Each cell image is a complete, independent sample, so photo order carries no signal. **Time series** is the counter-example: shuffling before the split puts *future* rows in training and their neighbours in test. That is leakage, and it makes the error look better than real forecasting would.

**Ex 2.2b. EDA write-up (question, what you saw, what you'd check next).**
- *Q1: do same-label images look alike?* Within a class, colour and shape are similar, but crop position and background vary a lot. Distance to own class mean: platelets 97% closest to their own mean; immature granulocytes only 21%. So IG is the likely weak class.
- *Q2: any blank or odd images?* No blanks. Some low contrast, some badly cropped, some odd background colours, and exact duplicates.
- *Next:* check per-class accuracy / confusion matrix for IG. Check whether odd backgrounds cluster in certain classes (a model could learn "background → class" shortcuts).

**Ex 2.3. Is the baseline good? Argue against guessing and against the lab's needs, then set the bar for the network.**
- *Against guessing:* 77% vs 19.5%, about 4× better. It clearly learned from the pixels.
- *Against the lab:* about 780 of 3,421 test images land in the wrong bin. That is a lot of manual re-checking, and one overall number can hide a weak class (basophils are ~7% of training data; IG is suspect from EDA).
- *Bar:* **~90%**. That halves the errors (~342 wrong) and justifies the extra cost. Also look at per-class recall, not just accuracy.

---

## NB3: Computer Vision Classification (a CNN, trained, evaluated, costed)

### Concepts
- **Two facts about images:** (1) a pixel relates to its **neighbours** (edges and blobs are local), and (2) a pattern means the same thing **wherever** it appears.
- **Scramble experiment:** shuffle the pixel positions the same way for every image. Logistic regression scores the **same** (~77%), which proves it never used spatial layout. A CNN would collapse on scrambled images.
- **Conv2d:** a small window of weights slides over every position. Small window = **local connectivity**. Same window everywhere = **weight sharing**. (Formal names: sparse interactions, parameter sharing. Together these are the CNN's **inductive bias**.) PyTorch doesn't flip the window (strictly cross-correlation); with learned weights it makes no difference.
- **Architecture:**
```
(3,28,28) → Conv(3→16,k3,p1) → ReLU → MaxPool2 → (16,14,14)
          → Conv(16→32,k3,p1) → ReLU → MaxPool2 → (32,7,7)
          → Flatten (1568) → Linear(1568→128) → ReLU → Linear(128→8) logits
```
- `padding=1` with a 3×3 kernel keeps H×W the same. Each pool halves H and W.
- **No softmax at the end:** `CrossEntropyLoss` applies it internally to raw logits. CrossEntropy is log loss for many classes.
- **Parameter counts:** CNN 206,952; logistic 18,824 (8×2352 + 8); no-conv MLP 302,216. **Almost all the CNN's params sit in the first Linear (1568×128).** The convolutions hold under 6k, and they *reduce* cost by shrinking what reaches the Linear layer.
- Parameter formulas: conv = `out_ch × (in_ch × k × k + 1)`; linear = `out × (in + 1)`.
- **Training:** Adam, lr 1e-3, 15 epochs. Loss drops steeply, then flattens. Still edging down means more epochs might help. Flat from the start: check data, labels and lr. Bouncing: lower the lr.
- **Evaluate:** pred = `argmax` over class scores (not a threshold at 0). Scoreboard: guess 19.5% → logistic 77.2% → **CNN 88.8%**.
- **Confusion matrix:** rows = true, columns = predicted, diagonal = correct.
  - **Recall** = diagonal / **row** sum ("of the real basophils, how many found?"). Basophil 196/244 = 0.80.
  - **Precision** = diagonal / **column** sum ("of those called basophil, how many were?"). 196/273 = 0.72.
  - **Accuracy** = diagonal / total, so a small class can be missed with little effect on it.
  - **Macro recall** = mean of the per-class recalls; every class counts equally. Here 0.881 vs accuracy 0.888 (close, because the weakest class, IG, is also large).
- **Inference cost:** `predict_one` adds a batch dim with `unsqueeze(0)` → 4 ms/image on CPU, 0.83 MB of weights. Warm up before timing. These numbers are good for **comparing models**, but are *not* the full deployment cost (no loading, preprocessing or serving overhead).

### Exercises

**Ex 3.1a. What do `MaxPool2d` and `Flatten` do to the block of numbers (shape in, shape out, kept vs thrown away)?**
- **MaxPool2d(2):** `(C, H, W) → (C, H/2, W/2)`, e.g. `(16,28,28) → (16,14,14)`. In each 2×2 block it keeps only the **largest** value. **¾ of the numbers are thrown away**, along with the *exact* position: the survivor says "the pattern was somewhere in this block". Benefits: less compute, tolerance to small shifts, and the next conv sees a wider area of the original image. Channels are unchanged.
- **Flatten:** `(32, 7, 7) → (1568,)` per image (batch: `(N,32,7,7) → (N,1568)`). It **only rearranges** into one row so a Linear layer can take it. No value changes and nothing is thrown away. Same as `.flatten(1)` before logistic regression.
- ⚠ Fix: you wrote that Flatten gives "[1,3,4,4]", which is a 4-D shape. The output is **1-D per sample**.

**Ex 3.1b. Why does a conv layer making 32 maps over a 14×14 image need so few parameters compared with a linear layer of the same output size?**
- conv2 = 32 × (16×3×3 + 1) = **4,640** params (conv1 = 448).
- A linear layer producing the same 32×14×14 = 6,272 outputs from its 16×14×14 = 3,136 inputs needs 6,272 × (3,136 + 1) ≈ **19.7 M** params.
- Why: **weight sharing**, because each of the 32 filters (3×3×16 + bias) is reused at all 196 positions. **Local connectivity**, because each output looks at a 3×3 patch, not every input value.
- ⚠ Fix: your comparison used the raw image (2,352 inputs → 14.8 M). The question means a linear layer producing the same 32×14×14 output from that conv's own input (16×14×14 = 3,136 inputs), which gives ≈ 19.7 M. Either way the conclusion is the same: thousands vs millions.

**Ex 3.3. Tell the lab where the model fails and what to do about the worst confusion. Then propose three other metrics.**
- *Where it fails:* **immature granulocytes have the worst recall, 0.72** (28 of 100 real IGs missed), confused mostly with monocytes, basophils and neutrophils. Monocyte (0.71) and basophil (0.72) have the lowest **precision**: many images get called those wrongly. Platelets and eosinophils are near perfect.
- *What to do:* route low-confidence IG/monocyte/basophil predictions to a technician; look at the confused images (EDA again); get more IG examples or use class weights; consider augmentation or higher resolution.
- ⚠ Fix: name, define and justify three metrics, e.g.:
  1. **Per-class recall / macro recall:** fraction of each true class found. It catches a missed rare class that accuracy hides.
  2. **Per-class precision:** of those predicted as class X, the fraction correct. It catches a bin polluted with wrong cells (e.g. monocyte).
  3. **Macro-F1:** harmonic mean of precision and recall, averaged per class. It catches a class that is weak on either side, weighted equally regardless of size.
  4. (Alt) **Log loss / calibration:** whether the confidence is trustworthy. It catches confidently-wrong predictions, which matters if confidence decides what the technician re-checks.

**Ex 3.5. Which tool for which problem?**
| # | Problem | Approach | Why / missing detail |
|---|---|---|---|
| 1 | Circuit boards, missing component | **Computer vision** | Image in, defect yes/no out. A fixed camera and fixed board layout may even allow classical template comparison. Detail: how many defect examples exist. |
| 2 | Discharge summaries → plain language | **Language model** | Free text in, rewritten text out. Detail: privacy (patient data) and the need for clinician review. |
| 3 | Drug orders from 3 years of weekly sales | ⚠ **Classical methods** (time-series forecasting: seasonal-naive baseline, lags + Random Forest) | Numbers in, a number out. No CV or LLM needed. |
| 4 | PDF forms, fixed fields → spreadsheet | ⚠ **Classical / neither** first | Same fields in the same place, so a PDF text/field parser or template extraction is enough. Add **OCR** only if the PDFs are scanned images. An LLM only if layouts actually vary across clinics. Detail: digital or scanned PDF? |
| 5 | Skin photos, flag urgent | **Computer vision** (CNN) | Image in, urgency flag out. Detail: labelled data, class imbalance; optimise **recall** for urgent cases. |

---

## NB4: Time Series Basics (EDA and data cleaning, Beijing PM2.5 hourly)

### Concepts
- **Tidy first:** combine year/month/day/hour into one `DatetimeIndex` with `pd.to_datetime`, `set_index`, `sort_index`, then drop the old columns. This enables `.loc["2013"]`, `resample`, and `.index.hour`.
- **Cadence** = the most common gap between rows (`index.diff().mode()`): hourly here.
- **Order matters:** the same values shuffled lose all structure.
- **Four things to look for:** trend, seasonality, autocorrelation, noise.
- **Decomposition** (`seasonal_decompose(series, period=…)`): trend + seasonal + residual. It can't handle NaN, so fill first. `period=24` (daily) gives the cleaner seasonal and a smaller residual than `24*7`: PM2.5's strongest short cycle is the **day**.
- **Stationary:** roughly stable mean and variance over time (check with rolling mean and std). The ADF test's null = unit root (non-stationary); small p is evidence against it, not proof of stationarity.
- **ACF / PACF:** correlation with lagged copies of itself. ACF decays gradually (short-term persistence dominates). PACF is strongest at the first few lags.
- **Cyclical encoding:** `sin(2π·h/24)`, `cos(2π·h/24)` puts the hours on a circle.
- **Scales:** a single day shows the daily rhythm (high at night, low mid-afternoon); the full record shows the yearly rhythm (high in cold months). Each hides what the other shows. The yearly swing is **bigger**; the daily cycle is **cleaner**.
- **Moving average:** a bigger window is smoother but **flattens the spikes you may care about most**.
- **Resampling:** downsample (hourly→daily→weekly) = less noise and size, but you lose the finer cycles. Upsample = invent points and fill gaps. *How far ahead you forecast decides how you preprocess.*
- **Missing values:** time series can be filled from neighbours because neighbouring hours are similar (you can't do that with shuffled tabular rows).
- **Outliers:** z-score > 3 flags points, but pollution spikes may be the **real events** you want to predict.

### Exercises

**Ex 4.0a. How does time-series data differ from regular data? Examples and analyses?**
- **Temporal dependence:** rows are *not* independent; order is fixed and neighbours are correlated (autocorrelation). Regular tabular ML assumes independent rows.
- Has structure tabular data doesn't: **trend, seasonality, cycles**. It can't be shuffled and needs chronological splits.
- Examples: retail sales (forecasting), stock prices (moving averages, volatility), sensor and pollution readings (anomaly detection), ECG (classifying sequences). Analysis types: describe patterns, classify sequences, forecast.

**Ex 4.0b. Why is `int64` not ideal for year/month/day/hour?**
- Four separate ints aren't a **timestamp**, so you can't slice by date, resample, compute gaps, or sort properly as time. Combine them into a `DatetimeIndex`.
- ⚠ Add: as model features, int64 implies a straight-line magnitude and order that's wrong for **cyclic** fields (hour 23 looks "far" from 0, December far from January). The numbers are labels on a cycle, not amounts to add or average.

**Ex 4.1. Same numbers, but one plot looks like information and the other like noise. Why? What does that say about random splits?**
- The top plot keeps time order, so you see persistence (autocorrelation), build-ups and decays, and daily cycles. The information lives in the **order**, not just the values. Shuffling keeps the distribution but destroys that.
- So a **random train/test split** for forecasting puts each test hour's neighbours in training (**leakage**) and tests on "the past", not the future. It gives an over-optimistic error. Use a **chronological split**.

**Ex 4.2a. Additive vs multiplicative decomposition?**
- **Additive:** `Y = T + S + R`. The seasonal swing has a **constant size** whatever the level. Simple and works with zero or negative values, but wrong if the swings grow with the level.
- **Multiplicative:** `Y = T × S × R` (⚠ you wrote T×S×T). The seasonal swing **scales with the level** (e.g. sales peaks grow as the business grows). Handles growing amplitude; needs strictly positive values. A log transform turns it into additive.
- Use multiplicative when the seasonal amplitude grows or shrinks with the trend. For PM2.5 additive was used; you could argue for multiplicative if the daily swings are larger in high-pollution winter.

**Ex 4.2c. Hour as 0–23 vs one-hot vs sin/cos?**
| Encoding | Wraparound explicit? | Gains | Loses |
|---|---|---|---|
| Single 0–23 | ❌ 23 and 0 look 23 apart | 1 column, simplest | False ordering and distance |
| One-hot (24 cols) | ❌ (no notion of distance at all; every hour equally far) | Flexible, no false order | 24 columns, not smooth, neighbours unrelated |
| sin + cos | ✅ 23:00 (−0.26, 0.97) is next to 00:00 (0, 1) | 2 columns, smooth, circular distance | Less intuitive |
- **Same sine, different cosine:** 0h and 12h (sin = 0, cos = +1 vs −1); also 6h±k pairs like 3h and 9h. Sine alone maps two hours to the same value, so you need **both** to get a unique point on the circle.
- **Link to 2.2:** the seasonal component showed a daily cycle (PM2.5 lowest mid-afternoon, highest around midnight). sin and cos let a model learn that position-in-the-day effect smoothly, e.g. a linear model `PM2.5 ≈ a + b·sin + c·cos` can fit one smooth daily wave.

**Ex 4.3. Symmetric or skewed? What does a long right tail mean, and which days dominate an averaged error?**
- **Right-skewed** (positive). Most hours are low or moderate, with rare smog episodes of 200–600+.
- If you average errors over the year, the **few extreme smog days dominate**. Example: 9 normal days with error 10 and one smog day with error 200 → MAE = 29, ~70% from one day. With **squared** error that day is **~98%** of the total.
- Why it matters: the metric decides what the model tries to get right. **RMSE/MSE** pushes it to fit the peaks (good if missing a smog day is the worst outcome). **MAE** weighs every day more equally, so it reflects typical-day performance. Report more than one.

**Ex 4.4a. ffill vs bfill vs interpolate: pros, cons, and which one looks into the future?**
| Method | How it crosses a gap | Pros | Cons |
|---|---|---|---|
| **ffill** | Flat at the **last** known value | Uses only the past, so it's safe for forecasting | Flat line, stale over long gaps |
| **bfill** | Flat at the **next** known value | Simple | Uses a **future** value, obviously |
| **interpolate** | Straight line from before to after | Looks most realistic and smooth | **Quietly** uses the future endpoint |
- **Interpolate** is the one that quietly looks into the future: it looks plausible, so the leak is easy to miss (bfill does it openly). In forecasting, filled training features then contain information that wouldn't be available at prediction time. That is **leakage**, and it gives optimistic scores that fail in production. Default to ffill.

**Ex 4.4b. Red outlier points: errors or real pollution? How to decide, and the cost of each mistake?**
- *Decide:* check plausibility (physically possible?), **consistency with neighbours** (a gradual build-up = real; a single isolated jump = suspicious), season (winter peaks are expected; the **July** points stand out), and other variables or stations (low wind, etc.). Don't delete just because a point is rare.
- ⚠ Add the **cost in each direction**:
  - Treat **real** spikes as errors and delete them → the model never learns smog events, under-forecasts exactly the dangerous days, and health warnings are missed. Usually the **costlier** mistake.
  - Treat **sensor errors** as real → the model learns false spikes, over-forecasts, and inflates the error metric (especially RMSE). Result: false alarms.

---

## NB5: Time Series Forecasting (Walmart M5, store CA_1, daily)

### Concepts
- **Lag trick:** turn a sequence into a supervised table. Each row's features are past values `X(t−1)…X(t−k)` and its target is `X(t+h)`. Still **univariate** (one measured variable), however many lag columns you build. **Calendar features** (day of week, month, holidays) are allowed because a future date is known in advance.
- **Problem:** input = recent demand + calendar context for the target day; output = units sold **HORIZON** days ahead; metric = **RMSE** (in units, lower is better).
- **Strongest cycle** = weekly → **ACF spikes at lag 7** (and 14, 21, 28).
- `build_supervised`: `lag_0` = today, `lag_l = s.shift(l)`, `roll_mean_7`, dow/month sin and cos, `is_event`, `snap`. **target = `s.shift(-horizon)`** pulls the future value onto today's row. `dropna` removes the first rows (not enough history) and the last `horizon` rows (no future yet).
- **Chronological split:** the first 80% of time is train and the last 20% is test. Demo at a 7-day horizon: chrono RMSE 510.8 vs random 501.5. The random split **flatters** the model.
- Boundary leak: with a large horizon, the last training rows' *targets* fall inside the test period. Fix it by leaving a gap of `HORIZON` days.
- **Baselines:**
  - **Naive / persistence:** tomorrow = today (`lag_0`). RMSE 917.2.
  - **Seasonal-naive:** same weekday last week (`shift(SEASON − HORIZON)`). RMSE **600.4**. This is the bar, because retail is day-of-week driven.
  - **The bar to beat = the *best* simple baseline.**
- **Random Forest** (100 trees averaged) on all features → **RMSE 431.8**, a real win (~28% better than seasonal-naive).
- **Error inspection:** busiest 10% of days under-forecast by ~580; quietest 10% over-forecast by ~211, so it **regresses toward the mean**. Christmas (store closed, ~0 sales) is left out.
- **Horizon curve (optional):** error generally rises with horizon. Seasonal-naive is undefined when h > 7.
- **Sequence models:** RNN/LSTM read the series step by step and carry a hidden state; LLMs (Transformers) predict the next **token** the same way. Not automatically better than a baseline.

### Exercises

**Ex 5.1. With a 4-week (28-day) horizon instead of 1 day, is yesterday still useful? What matters more?**
- Yesterday is much less useful: it's 28 days before the target, not 1. (Note: features can only use data up to the forecast date, so "the value right before the target" isn't available.)
- More important: **day of week** of the target day (Saturday ≫ Tuesday), the **same weekday 4 or 52 weeks earlier** (lag 28, 364), **monthly / yearly seasonality** (e.g. August peak), **holiday/event** and **SNAP** flags for the target date, and longer rolling means (28-day) for the level.
- Change: rebuild the lags relative to the horizon, add longer lags and rolling windows, put calendar features on the **target** date, and leave a horizon-sized gap between train and test.

**Ex 5.2a. Why is the random split's lower error a lie? Explain to a client.**
"With a random split, the model is tested on days whose neighbours, including the days *after* them, were in training. It's like predicting Tuesday after already seeing Monday *and Wednesday*. In real use you only ever have the past, so the random-split number describes a situation that can never happen. The chronological number is the one to quote."

**Ex 5.2b. Sketch walk-forward validation. Why does it respect time?**
```
Fold 1: train [0..k]     → test [k+1]
Fold 2: train [0..k+1]   → test [k+2]
Fold 3: train [0..k+2]   → test [k+3]    (expanding window; average the fold errors)
```
Every test point is strictly **after** all of its training data, so there's no future information in training. It mimics production, where you retrain as new data arrives. Ordinary k-fold shuffles, so some folds train on the future.

**Ex 5.4a. Does `N_LAGS = 7` reach the ACF spike? Should it? What does `roll_mean` add? Why does `snap` matter for this store?**
- **Yes.** At H = 1, `lag_6` is 7 days before the target, the weekly spike. It should, because lag 7 is the strongest signal ("same weekday last week" is the best baseline).
- **roll_mean_7** = one smoothed number for the recent **level/trend**, robust to single-day noise. A 7-day window also averages out the weekday effect. Raw lags give the individual days; the mean gives "is traffic up or down overall".
- ⚠ Add: **SNAP** (food-benefit payment days) is **state-specific**. CA_1 is in California, so the flag reflects California's SNAP calendar and spending surges. A store in another state has different SNAP days or a weaker effect, so the same column could be noise there.

**Ex 5.4b. Did the Random Forest beat the best baseline at H = 1? Where does it fail?**
- **Yes:** RF 431.8 vs seasonal-naive 600.4 vs naive 917.2, about a 28% gain over the best baseline. Here the extra features (lags + calendar + events + SNAP) paid off. The general lesson still holds: always check; on easy short horizons complex models often only *tie* the simple rule.
- ⚠ Fix: your sentence "complex models still win for short-horizon tasks because the future is uncertain" isn't the right reasoning. It won here because it **combines several signals** the single-rule baselines ignore.
- Failures:
  - **Lags the turns:** it reacts to a surge or drop a day late because it leans on recent lags.
  - **Flattens peaks:** a Random Forest **averages** leaf values, so extremes are pulled toward the mean. The busiest 10% of days are ~580 units under.
  - **Misses holiday shifts:** e.g. around 4 July, people shop the day **before** and it's quiet **on** the day. `is_event` only flags the day itself, so the model misses the pre-holiday surge (one concrete failure to name).

---

# Part B: From Notebook to Production (03)

## P1: Config, environment variables and logging

### Config files vs environment variables
| | Config files (YAML/JSON/TOML/INI) | Environment variables |
|---|---|---|
| What | External settings file | Key-value pairs set **outside** the code (shell, container, CI) |
| For | **Complex/nested** settings: hyperparameters, paths, feature lists, metric thresholds | **Secrets** (API keys, passwords, DB URIs) + values that **differ per environment** (dev/test/prod DB URL) |
| Version control | ✅ Yes (never secrets) | ❌ Kept out of git |
| Python | Hydra / PyYAML | `os.environ.get("KEY", default)` |
- **Common pattern:** the config file holds the structure and defaults; env vars **override** per environment and supply secrets.
- **Payoff:** consistency and reproducibility across environments, flexibility to experiment, **security** (no hardcoded secrets), maintainability (tune in the config, debug in the code).
- Production secrets go in a secret manager (e.g. AWS/GCP Secret Manager); `python-dotenv` loads a local `.env` for development.

### Logging
- **print vs logging:** logging adds persistence (files and external sinks), structure (timestamp, level, module), control (filter by level or source), flexible destinations, and automatic context. `print` is unsuitable for production.
- **Levels (a threshold):** `DEBUG < INFO < WARNING < ERROR < CRITICAL`. Setting a level shows that level **and above**. DEBUG while developing, WARNING (or INFO) in production.
  - DEBUG = per-row and diagnostic detail; INFO = step start/end with counts; WARNING = unexpected but recovered (e.g. a bad row skipped, no GPU); ERROR = an operation failed; CRITICAL = the program can't continue.
  - Bad input that is handled correctly is a **WARNING**, not an ERROR: the function didn't fail.
- **Three steps:** `basicConfig(...)` once (`force=True` in Jupyter), `logger = logging.getLogger(name)`, then `logger.info(...)`.
- **Named loggers:** `getLogger(__name__)` gives the module's dotted name, so you know *where* a line came from and can `setLevel` one module (e.g. quiet `data_prep`) without touching others. `__name__` updates automatically if the file moves and follows the package hierarchy.
- **Handlers** decide *where* (StreamHandler → console/stderr; FileHandler → disk). **Formatters** decide *how each line looks*. **Each handler has its own level.** E.g. console at INFO + file at DEBUG lets an overnight job keep full detail in the file without flooding the screen.
- **Two gates:** a message must pass the **logger/root level** first, *then* each handler's level. In `logging.yaml`, root = INFO, so DEBUG never reaches even `debug.log`. That is also why `debug.log` looks identical to `info.log`.
- `delay: True`: a log file is only created when its first message arrives (so `errors.log` is missing after a clean run).
- `logger.exception("msg %s", x)`: logs at ERROR **plus the full traceback**. Use it only inside `except`. `logger.error(e)` gives the message only. (`exc_info=True` is equivalent.)
- **`%s` args instead of f-strings:** the string is built only if the message is actually emitted (lazy formatting).
- **Structured (JSON) logging:** one JSON object per line, so machines can search, filter, count and dashboard it, and adding fields is safe. Libraries: `structlog`, `Loguru`.
- **Never log:** secrets (keys, passwords, tokens) or PII (NRIC, bank, health data). Logs are copied, retained and widely readable, which is a PDPA/compliance risk.
- **Don't log everything:** I/O slowdown, disk and storage cost, hard to find the important lines, and a higher chance of leaking secrets.
- Why logging matters in production: **debugging** (what failed at 3 AM), **reproducibility** (log config, versions, seeds), **monitoring** (batch timing, drift warnings), **audit/compliance** (who did what).

### Hydra
- `@hydra.main(version_base=None, config_path="conf", config_name="config")` loads `conf/config.yaml` and passes it as `cfg`. Read values with `cfg.model.lr`.
- **CLI override:** `python app.py model.lr=0.1 model.epochs=10`. A typo in a key (e.g. `model.learning_rate`) **errors** rather than being silently ignored. Use `+model.dropout=0.2` to **add** a new key.
- `--cfg job` prints the final config and exits without running.
- Each run gets an `outputs/YYYY-MM-DD/HH-MM-SS/` folder with `.hydra/` (config.yaml, overrides.yaml, hydra.yaml) plus a log file. You can see exactly which settings produced a result weeks later.
- **Multirun `-m`:** runs every combination and writes to `multirun/`. `-m model.lr=0.1,0.01 model.epochs=5,10` → **4 runs**.
- **Env var in config:** `raw_dir: ${oc.env:RAW_DATA_DIR,./data/raw}` reads the env var or falls back to the default. `RAW_DATA_DIR=/x python app.py` sets it for one command only. Use it for paths that differ between your workspace, a teammate's laptop, and prod.
- In `process_data.py`, `setup_logging()` replaces Hydra's logging with `conf/logging.yaml`, so Hydra's `process_data.log` in `outputs/` stays **empty** and the logs go to `./logs/`.

### `if __name__ == "__main__":`
Code inside runs only when the file is **executed directly** (`python file.py`, where `__name__ == "__main__"`), **not when imported** (then `__name__` is the module name). It lets a file be both a runnable script and an importable module without side effects.

---

## P2: Containerisation (Docker)

### Concepts
- **Containerisation:** package an app with all its dependencies into a standard unit. It solves **"works on my machine"**: the same behaviour in dev, staging and prod.
  - Limit: it doesn't solve hardware or architecture differences (a CUDA image can't use an Apple Silicon GPU).
- **VM vs container:**
| VM | Container |
|---|---|
| Full guest OS **with its own kernel**; virtualises hardware | **Shares the host kernel**; virtualises at the OS level |
| GBs, boots in seconds to minutes | MBs, starts in about a second |
| Strong isolation, high overhead | Good-enough isolation (weaker than a VM), low overhead |
On macOS/Windows, containers run inside a hidden Linux VM.
- **Why containerise:** reproducibility, isolation + per-container resource limits, fast start + horizontal scaling + density, immutable versioned images that fit CI/CD.
- **Dockerfile → image → container:**
  - **Dockerfile** = the recipe (instructions).
  - **Image** = a read-only, **immutable**, layered template, versioned with tags and stored in registries.
  - **Container** = a running instance of an image. **Ephemeral**: data written inside is lost when it's removed, so save outputs to volumes or external storage.
- **Registries:** public (Docker Hub, GitHub) and private (**GAR** on GCP, ACR, **Harbor** on-prem/air-gapped). Private ones add access control, CI/CD integration and vulnerability scanning.
- **Runtimes:** Docker (client + **daemon** `dockerd`, a background API service that does the build/run work) vs **Podman** (**daemonless**, **rootless**, OCI-compatible, same commands). **Orchestration:** Docker Swarm (simple, built-in) vs Kubernetes (industry standard, scalable, declarative).
- **Otherside:** devbox (where you type) → Docker runs on **cpubox/gpubox**. Mounts must be under `/data`. The daemon is **shared**, so suffix image and container names with your username, use unique ports (base + PORT_OFFSET), and only stop your own containers.

### Commands and flags
| Command | Purpose |
|---|---|
| `docker build -t name:tag .` | Build. `-t` = name:tag; **`.` = build context** (the files sent to the daemon; `COPY` can only reach inside it); `-f` = which Dockerfile; `--platform` for cross-architecture builds |
| `docker run --rm img` | Run. `--rm` deletes the container when it exits; `-it` = interactive shell; `-d` = detached/background; `--name`; `-p host:container`; `-v host:container[:ro]`; `-e KEY=val`; `--user 2222:2222`; `--memory/--cpus` limits; `--restart`; `--gpus all` |
| `docker ps` / `-a` / `--filter "status=exited"` | List running / all / stopped containers |
| `docker stop` / `docker rm` / `rm -f` | Stop a running container / remove a stopped one / force both |
| `docker tag src registry/path/name:tag` | Add a **second name** for the same image (same IMAGE ID, nothing copied or uploaded). Needed because a push goes to the registry named in the image name |
| `docker push …` | Upload to the registry (after `gcloud auth configure-docker asia-southeast1-docker.pkg.dev`) |
| `docker logs [--tail 20] [-f]` | A container's stdout/stderr, your main debugging tool |
| `docker exec [-it] ctr cmd` | Run a command **inside a running** container. Exiting an exec'd bash doesn't stop the container: only the main process (e.g. `sleep infinity`) keeps it alive |

GAR image name: `asia-southeast1-docker.pkg.dev/<project>/<repo>/<image>:<tag>`.

- **Build context vs WORKDIR vs cwd:** cwd = where your shell is; build context = the directory sent to the daemon (it bounds `COPY`); WORKDIR = a directory *inside the image*.
- **Ports:** `-p 8830:8501` = host 8830 → container 8501. `EXPOSE` only **documents** a port; `-p` actually publishes it. On Otherside: browser → devbox:8501 → `socat TCP-LISTEN:8501,fork TCP:cpubox:8830 &` → cpubox:8830 → container:8501.
- **Bind to `0.0.0.0`, not `127.0.0.1`:** `127.0.0.1` inside a container = only accept connections from inside that container. Traffic from `-p` arrives on `eth0` and is refused. `0.0.0.0` = listen on all interfaces.
- **Port conflicts** ("port is already allocated"): pick another port in your +1000 range; `docker ps`; `lsof -i :8000`.
- **Volumes / bind mounts:** `-v /host/path:/container/path`. What's written in the container lands on the host and survives `--rm`. A **new** container starts fresh from the image, so files written inside an old container are gone. `:ro` = read-only: mount **inputs** read-only and outputs writable.
- **File ownership:** containers default to **root (UID 0)**, so files on the host can end up root-owned and uneditable by you (coder = UID 2222). Fix with `--user 2222:2222` or a `USER` line.

### Dockerfile instructions
`FROM` (base image) · `WORKDIR` (cd inside the image; created if missing) · `COPY`/`ADD` (host → image) · `RUN` (at **build** time) · `ENV` · `EXPOSE` (documentation only) · `VOLUME` · `CMD`/`ENTRYPOINT` (at **run** time). **Each instruction = one layer.**

```dockerfile
FROM python:3.12-slim                 # slim, pinned tag (never :latest)
WORKDIR /app
COPY requirements.txt .               # dependencies first...
RUN pip install --no-cache-dir -r requirements.txt   # ...cached layer
COPY . .                              # code last (changes often)
RUN adduser --disabled-password --gecos '' appuser
USER appuser                          # don't run as root
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

### Best practices (memorise)
1. **Layer caching:** copy `requirements.txt` and install **before** `COPY . .`. Docker reuses cached layers up to the first changed instruction, so a code change doesn't reinstall packages.
2. **Combine RUNs** + clean up in the **same** layer: `apt-get update && apt-get install -y … && rm -rf /var/lib/apt/lists/*`. Deleting in a later layer doesn't shrink the earlier one.
3. **`.dockerignore`:** exclude `.git`, `.env`, `__pycache__`, data. That gives a smaller context, faster builds and no leaked secrets.
4. **Minimal base images** (`-slim`, alpine, **distroless** = only the app + runtime, no shell or package manager) → smaller, fewer vulnerabilities.
5. **Pinned tags** (`python:3.12-slim`, not `latest`) → reproducible.
6. **Non-root user** (`USER appuser`).
7. **No secrets in images** (`ENV API_KEY=…` or `COPY .env` bake them into a layer forever). Pass at runtime (`-e`, acceptable for dev) or use Compose/secret managers.
8. **Resource limits** on shared hosts: `--memory=512m --cpus=0.5`.
9. **Multi-stage builds:** a **builder** stage has the compilers, build tools and full dependency install; the **runtime** stage starts from a clean slim image and uses `COPY --from=builder` to take **only the artefacts** (e.g. `/opt/venv`, the app). The final image is small and has no build tools.

```dockerfile
FROM python:3.12-slim AS builder
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r requirements.txt
FROM python:3.12-slim
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
```

### 5.2 Debug-a-Dockerfile (the four bugs and how each showed up)
| Bug | Symptom / where found |
|---|---|
| Misspelled filename in `COPY` (requirements file) | **`docker build` fails** at the COPY step; compare names character by character |
| `CMD ["python", "sever.py"]` typo | Container **exits immediately**: `docker ps -a` shows Exited, `docker logs` shows `can't open file` |
| `PyYAML` missing from requirements (`import yaml` ≠ pip name) | Exits: `docker logs` shows `ModuleNotFoundError` |
| Flask bound to `127.0.0.1` | Container **running but unreachable**: `docker logs` shows `127.0.0.1:5000`. A request from **inside** (`docker exec … urlopen('http://127.0.0.1:5000')`) works; from outside it fails. Fix: `app.run(host="0.0.0.0", port=5000)` |
Debug order: **build output → `docker ps -a` (exit status) → `docker logs` → `docker exec`**.

### 5.5 Streamlit questions
- **Use your own model:** `COPY assets/my_model.pkl assets/model.pkl` (the app skips the download if the file exists), or bind-mount it at runtime.
- **Does every `docker run` re-download?** Yes, because the download lands inside an ephemeral container. Persist it by baking it into the image or bind-mounting `-v /host/assets:/app/assets`.

### Extras (skim): Docker Compose
Define services, ports, volumes, env and secrets in `compose.yml`, then use `docker compose build / up -d / ps / logs / exec / down`. It's declarative, multi-container, and supports runtime secrets. Cleanup: dangling images, stopped containers, unused volumes and networks (`prune`, but only your own on a shared daemon).

---

## P3: DataOps

### Data shapes
| Shape | Meaning | Examples |
|---|---|---|
| **Structured** | Fixed schema, rows × columns, typed; easiest to query | CSV, Parquet, SQL tables |
| **Semi-structured** | No rigid schema but **self-describing** (keys/tags inline) | JSON, JSONL, XML, YAML, logs |
| **Unstructured** | No schema; you must extract structure first | Text, PDF, images, audio, video |
Storage choice is driven by **shape, size and query pattern** (and access frequency).

### Data access methods
Direct DB connection (SQL / NoSQL) · direct download (`wget`/`curl`) · dataset-hub SDKs · **REST/GraphQL API** · cloud object storage (S3/GCS) · **event streams** (real-time pushes).

### REST
- **Endpoint** = URL of a resource. **Methods:** GET (read), POST (create), PUT/PATCH (update), DELETE. POST-for-complex-reads is a frowned-upon workaround.
- **Response** = JSON + **status code**: `200` OK, `404` not found, `429` too many requests (4xx = client error, 5xx = server error).
- **Pagination:** the server returns pages (`?limit=100&offset=0`) plus a `next` link; loop until `next` is `None`, with a `MAX_PAGES` safety net.
- **Rate limits:** pause between requests (`time.sleep`) and reuse a `requests.Session`.
- **Failure handling:** always set a `timeout`. 404 → log a WARNING and return `None`; any other 4xx/5xx → `response.raise_for_status()` raises. A pipeline should handle expected failures gracefully, not crash halfway.
- **GraphQL vs REST:** GraphQL lets the client name the exact fields it wants (no over-fetching); with REST the server decides what each endpoint returns.
- **Bronze landing:** save responses **exactly as received**, one JSON per line (`.jsonl`). Write to a `.tmp` file, then rename, so a crash never leaves a half-written file. Skip the download if the file already exists (idempotent, kind to free APIs).

### Training data origins + legal
- Origins: public hubs (Kaggle, UCI, HF) · web crawls (Common Crawl, huge and noisy) · licensed/commercial · **synthetic** (simulators, GAN/LLM, SMOTE) · **internal** (most valuable and differentiated, but often full of PII).
- **Legal gates:** licence/copyright ("public" ≠ free to use or train on; scraping can breach terms of service) · **privacy**: Singapore **PDPA** (EU GDPR) requires consent or another lawful basis and security · **sharing**: you need redistribution rights + true **anonymisation**. Removing names isn't enough; quasi-identifiers like postal code + birth date + gender can re-identify people.
- Obligations **increase** when you share the data or a model trained on it.

### Medallion architecture (data quality in stages)
| Layer | Contents | Quality |
|---|---|---|
| 🥉 **Bronze** | Raw, untouched, as ingested | Lowest |
| 🥈 **Silver** | Cleaned and conformed: deduped, types fixed, bad rows dropped, units standardised, schema enforced | Medium |
| 🥇 **Gold** | Curated, aggregated, feature-engineered, business/ML-ready | Highest |
- **Always keep Bronze:** if the cleaning logic has a bug, fix `to_silver` and **reprocess from bronze** without calling the API again.
- Each hop is a transformation (ETL/ELT). Backups ≈ bronze standard; analysis, dashboards and LLM/RAG knowledge bases need silver/gold.
- **Write the validation checks first** (`validate_silver`): define what "clean" means, collect **every** problem, raise, and **only save if it passes**, so bad data never reaches gold.
- Bronze → silver steps: `drop_duplicates(subset="id")` → `dropna(subset=["name"])` → `pd.to_numeric(errors="coerce")` → filter values > 0 → unit conversion (dm→m, hg→kg) → rename.
- Silver → gold features: `bst` = sum of stats, `bmi`, `is_dual_type`, one-hot `pd.get_dummies(prefix="type")`. Top 3 bst: Mewtwo, Dragonite, Mew.
- **Stop vs drop-and-continue:** stop (fail fast) when essential columns or the schema are broken, because continuing corrupts downstream joins, models and DBs. Drop + log a WARNING for a few bad rows in non-essential data. A middle ground is to **quarantine** bad rows for later review.

### Row vs column storage
- **Row-based** (CSV, JSONL, OLTP DBs): a whole record is stored together. Good for writing and reading entire records.
- **Columnar** (**Parquet**, warehouses): each column is stored together. Reading only the needed columns is fast for analytics, **compression** is excellent (same-typed values sit together), and **dtypes and schema are preserved** (CSV is all text).
- So Parquet suits the silver/gold layers.

### Databases by query pattern
| Query by… | DB type | Examples | Good for |
|---|---|---|---|
| Name/key | **Object/blob storage** | S3, GCS | Whole files of any type (images, PDFs, datasets) |
| Name/key | **Key-value** | Redis, DynamoDB | Near-instant lookup by key; can't query inside the value (caches, sessions) |
| Structured tables | **Relational (SQL)** | Postgres, MySQL | Fixed schema, joins, indexes, **ACID** transactions, one-to-many relationships |
| Structured tables | **Wide-column** | Cassandra, HBase | Huge tables, very high write volume, partitioned across machines |
| Structured tables | **Data warehouse** (columnar) | BigQuery, Snowflake | Analytics: scan and aggregate a few columns over millions of rows |
| Flexible fields | **Document** | MongoDB | JSON-like records that can each have a different shape; query any field |
| Relationships | **Graph** | Neo4j | Nodes + edges; friends-of-friends and shortest path without costly repeated joins |
| Time | **Time-series** | InfluxDB, TimescaleDB | Append-heavy timestamped readings, range queries, roll-ups and retention |
| Similarity (optional) | **Vector** | pgvector, Pinecone | Embeddings + approximate nearest-neighbour search ("find things like this") |
**SQL vs NoSQL:** SQL = relational tables, a fixed schema, one standard language. NoSQL = a *family* of non-relational stores, each with its own model and API.

### Loading data: disk → RAM → CPU
Disk = large, cheap, persistent. **RAM** is ~100,000× faster but small, and the OS and other apps already use part of it. The CPU only works on data in RAM. pandas loads everything into RAM, so a large file → **OOM**. Mitigate with `usecols=`, `chunksize=`, or push the work down to a DB/warehouse.

### ETL vs ELT
| | **ETL** (Extract → Transform → Load) | **ELT** (Extract → Load → Transform) |
|---|---|---|
| Where it transforms | Its own staging/processing engine, *before* loading | Inside the warehouse, *after* loading |
| Medallion | Extract = bronze, transform = silver, load = clean/gold | Load = **bronze lands in the warehouse**, transform → silver/gold |
| Raw data kept? | ❌ Only the cleaned result is loaded, so a logic bug means re-extracting everything | ✅ Raw stays, so you can re-run the transform |
| Scale/cost | The separate engine becomes a bottleneck | Uses elastic warehouse compute |
| PII/compliance | ✅ Mask or drop PII **before** it ever lands (e.g. a bank strips NRIC before the analytics warehouse) | Raw PII lands first and must be protected afterwards with access controls |
**Choose ETL** for sensitive or regulated data. **Choose ELT** for large volumes, fast iteration and keeping the raw data, when PII isn't the concern.

---

## P4: Hands-on (where P1 and P2 meet)
See the [P4 README](03-from-notebook-to-prod/P4_hands-on/README.md) and [PLAN.md](03-from-notebook-to-prod/P4_hands-on/PLAN.md). Know: which **env vars** the script reads, its **log format and levels**, any **rounding** rules, and exactly **how the image is built, tagged (GAR path) and run**.

---

## Last-minute rapid-fire
- `zero_grad` → because grads accumulate. `backward` → backprop. `step` → apply the update.
- Activation → non-linear boundary. Hidden width → capacity. Test gap → overfitting.
- Adam lr ≈ 10× smaller than SGD's; never copy lrs between optimisers.
- Baseline first: guess 19.5% → logistic 77.2% → CNN 88.8%. TS: naive 917 → seasonal-naive 600 → RF 432.
- Recall = row, precision = column. Macro recall = equal weight per class.
- Conv params = `out × (in × k² + 1)`; pooling halves H and W; Flatten only reshapes.
- Can't shuffle time → chronological split + walk-forward. bfill and interpolate leak the future; use ffill.
- sin + cos together for cyclic features. Right skew → RMSE dominated by peaks; MAE = typical day.
- Config file = structure and defaults (in git); env var = secrets and per-environment values (not in git).
- Levels DEBUG<INFO<WARNING<ERROR<CRITICAL; logger level is the first gate, handler level the second. `logger.exception` inside `except`.
- Docker: deps before code, combine RUNs, `.dockerignore`, slim + pinned, non-root, no secrets, multi-stage, bind `0.0.0.0`.
- Bronze raw → silver clean (validate first) → gold features. Parquet = columnar. ETL protects PII; ELT keeps the raw data.
