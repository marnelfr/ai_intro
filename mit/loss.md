## Loss
Definitions:
- **Loss** is a number that measures how bad the predictions are.  
- **Loss** is the cost incurred from incorrect predictions.

Both mean: the more wrong the prediction, the **higher the penalty (cost/loss)**; the more accurate the prediction, the **lower** the penalty.

---

### 1) What exactly is “loss”?
- For **one example** \((x, y)\), the **loss function** \(\ell(\hat y, y)\) outputs a **single number**.  
  - \(\hat y\): the model’s prediction for \(x\)
  - \(y\): the true/target value
- For a **batch** (size \(B\)) we usually **average** the per-example losses:
  
  \[
  \text{batch\_loss} \;=\; \frac{1}{B}\sum_{i=1}^B \ell(\hat y^{(i)}, y^{(i)})
  \]
  
- Training tries to **minimize** this loss (also called **cost**, **error**, or **objective**).

> “Cost incurred from incorrect predictions” = “penalty assigned by \(\ell\)” when \(\hat y\) deviates from \(y\).

---

### 2) Why do we use loss instead of just “correct/incorrect”?
A strict “0 if correct / 1 if incorrect” (**0–1 loss**) is not **differentiable**, so it’s not useful for gradient-based learning.  
Instead, we use **smooth, differentiable** losses that:
- give **small penalties** for small mistakes,
- give **large penalties** for big or confident mistakes,
- provide **gradients** to update weights.

---

### 3) Common loss functions (with intuition & tiny numbers)

###### A) **Regression** (predict real values)
- **MSE** (Mean Squared Error):
  
  \[
  \ell(\hat y, y)=\frac{1}{2}(\hat y - y)^2 \quad \text{(the } \tfrac{1}{2}\text{ is optional)}
  \]
  
  - **Small miss** → small penalty; **big miss** → much bigger penalty (square grows fast).
- **MAE** (Mean Absolute Error):
  
  \[
  \ell(\hat y, y)=|\hat y-y|
  \]
  
  - Penalizes **linearly** with distance; more robust to outliers than MSE.

**Numeric feel (MSE)**, true \(y=3.0\)  
- \(\hat y=2.6 \Rightarrow \ell = (2.6-3)^2/2 = 0.08\)  
- \(\hat y=5.0 \Rightarrow \ell = (5-3)^2/2 = 2.0\)  
Second prediction is **more wrong**, so **much higher cost**.

---

###### B) **Binary classification** \((y\in\{0,1\})\)  
Model outputs a probability \(p=\Pr(y{=}1\mid x)\).

- **Binary Cross-Entropy (Log Loss)**:
  
  \[
  \ell(p, y) = -\big[\,y\log p + (1-y)\log(1-p)\,\big]
  \]
  
  Examples (true label \(y=1\)):  
  - \(p=0.9 \Rightarrow \ell\approx 0.105\) (small cost: right and confident)  
  - \(p=0.6 \Rightarrow \ell\approx 0.511\) (higher cost: less confident)  
  - \(p=0.1 \Rightarrow \ell=2.302\) (**huge cost**: confidently wrong)

This captures “cost from incorrect predictions”: **the more confidently wrong you are, the bigger the penalty**.

---

###### C) **Multi-class classification** (one correct class among \(K\))
Model outputs a probability vector \(\hat{\boldsymbol p}\) via **softmax**.  

- **Categorical Cross-Entropy**:
  
  \[
  \ell(\hat{\boldsymbol p}, \boldsymbol y) \;=\; -\sum_{c=1}^{K} y_c \log \hat p_c
  \]
  
  where \(\boldsymbol y\) is **one-hot** (1 for the true class \(c^\*\), 0 otherwise).  
  This simplifies to \( -\log(\hat p_{c^\*}) \).  
  **Low \(\hat p_{c^\*}\)** = **high cost**.

---

### 4) Loss vs. “incorrectness”
- Loss is **graded**, not just “0 or 1”.  
- It reflects **how wrong** you are and **how confident** the model is.  
- Minimizing loss usually yields better generalization than optimizing raw accuracy early on.

---

### 5) Per-example loss → dataset loss → training
- **Per-example**: \(\ell(\hat y^{(i)}, y^{(i)})\)  
- **Batch loss**: average over a mini-batch  
- **Epoch loss**: average over all batches in an epoch  
- The optimizer uses the **gradient of the loss** to nudge weights to reduce future loss.

---

### 6) “Cost” can also mean real-world costs (optional)
Sometimes we **weight** the loss to reflect real costs:
- False negatives in medical diagnosis might be **more costly** than false positives.  
- With class imbalance, weight classes differently in the loss to avoid bias.

---

### 7) Takeaway
- **Loss / cost / objective** are near-synonyms here.  
- “Cost from incorrect predictions” = a plain-English summary of:  
  > *Compute a numeric penalty for each prediction; the more wrong (and confident) it is, the bigger the penalty; train to minimize the total penalty.*

## Multi-class case, step by step.

### Setup
- You have **K classes** (e.g., 10 digits, 100 object categories, etc.).
- The last layer of your network produces either:
  - **logits** $begin:math:text$z \\in \\mathbb{R}^K$end:math:text$ (raw scores, no activation), or
  - **probabilities** $begin:math:text$\\hat{\\boldsymbol p}$end:math:text$ after a **softmax** activation.

###### Softmax (turn logits into probabilities)
$begin:math:display$
\\hat p_i \\;=\\; \\frac{e^{z_i}}{\\sum_{j=1}^{K} e^{z_j}}\\,,\\quad i=1,\\dots,K
$end:math:display$
- Each $begin:math:text$\\hat p_i \\in (0,1)$end:math:text$ and $begin:math:text$\\sum_i \\hat p_i = 1$end:math:text$.

###### Categorical cross-entropy loss
If the true class is $begin:math:text$c^\\*$end:math:text$ (one-hot vector $begin:math:text$\\boldsymbol y$end:math:text$ with $begin:math:text$y_{c^\\*}=1$end:math:text$), the **per-example** loss is:
$begin:math:display$
\\ell(\\hat{\\boldsymbol p}, \\boldsymbol y) \\;=\\; -\\sum_{i=1}^{K} y_i \\log \\hat p_i \\;=\\; -\\log \\hat p_{c^\\*}.
$end:math:display$
For a batch, we average over examples.

######### What it means
- If the model assigns **high probability** to the true class, the loss is **small**.  
- If it assigns **low probability** (especially if it’s confident about a wrong class), the loss is **large**.  
This is exactly the “cost incurred from incorrect predictions.”

---

### Tiny numeric example (by hand)
Take **3 classes** $begin:math:text$(K=3)$end:math:text$ with logits:
$begin:math:display$
z = [\\,2.0,\\; 0.5,\\; -1.0\\,].
$end:math:display$
Exponential terms:  
$begin:math:text$e^{2.0}\\approx 7.389$end:math:text$, $begin:math:text$e^{0.5}\\approx 1.6487$end:math:text$, $begin:math:text$e^{-1.0}\\approx 0.3679$end:math:text$.  
Sum $begin:math:text$S \\approx 7.389 + 1.6487 + 0.3679 = 9.4056$end:math:text$.

Softmax probabilities:
- $begin:math:text$\\hat p_1 = 7.389 / 9.4056 \\approx 0.7858$end:math:text$
- $begin:math:text$\\hat p_2 = 1.6487 / 9.4056 \\approx 0.1753$end:math:text$
- $begin:math:text$\\hat p_3 = 0.3679 / 9.4056 \\approx 0.0391$end:math:text$

**Case A**: true class = 1 → loss $begin:math:text$= -\\log(0.7858) \\approx 0.241$end:math:text$ (good prediction).  
**Case B**: true class = 3 → loss $begin:math:text$= -\\log(0.0391) \\approx 3.240$end:math:text$ (confidently wrong → big penalty).

---

### How to use it in Keras/TensorFlow

###### Option 1 — Last layer has softmax
```python
from tensorflow import keras
from tensorflow.keras import layers

K = 10  ### number of classes

model = keras.Sequential([
    layers.Input(shape=(m,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(K, activation="softmax"),  ### outputs probabilities
])

### If your labels are integers 0..K-1:
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

### If your labels are one-hot vectors:
### model.compile(optimizer="adam",
###               loss="categorical_crossentropy",
###               metrics=["accuracy"])
```

###### Option 2 — Last layer returns logits (no activation)
This is numerically stable if you use the “from_logits” loss:
```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model = keras.Sequential([
    layers.Input(shape=(m,)),
    layers.Dense(128, activation="relu"),
    layers.Dense(K)  ### logits
])

model.compile(optimizer="adam",
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
```
**Do not** apply softmax yourself if you use `from_logits=True`; the loss will handle it internally.

---

### Labels: sparse vs categorical
- **Sparse**: labels are **integers** (e.g., `3`, `7`). Use `sparse_categorical_crossentropy`.
- **Categorical**: labels are **one-hot vectors** (e.g., `[0,0,1,0,...]`). Use `categorical_crossentropy`.

They compute the same math; the input label format is the only difference.

---

### Helpful extras (common multi-class tweaks)

######### Label smoothing
Reduces overconfidence and can improve generalization:
- Replace the hard one-hot target with a slightly smoothed version:
  $begin:math:display$
  y' = (1-\\varepsilon)\\, \\text{one\\_hot} + \\frac{\\varepsilon}{K}\\,\\mathbf{1}
  $end:math:display$
- In Keras:
```python
from tensorflow.keras.losses import CategoricalCrossentropy
loss = CategoricalCrossentropy(label_smoothing=0.1)
### Or for sparse labels:
### loss = SparseCategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
```

######### Class imbalance
Give more weight to rare classes:
```python
model.fit(X, y, epochs=..., class_weight={0:1.0, 1:2.5, 2:4.0})
```
(or provide `sample_weight` per example).

######### Metrics
- `accuracy` for overall correctness.
- `top_k_categorical_accuracy` if you care whether the true class is in the top-k predictions:
```python
from tensorflow.keras.metrics import TopKCategoricalAccuracy
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=[TopKCategororicalAccuracy(k=5)])
```

---

### Quick checklist for multi-class
- Use **softmax** (or logits + `from_logits=True`).
- Pick **sparse** vs **categorical** loss to match your label format.
- Consider **label smoothing** and **class weights** if you overfit or face imbalance.
- Don’t apply softmax **twice** (either in the last layer **or** in the loss).
- Monitor **training vs validation** loss/accuracy across epochs to catch over/underfitting.