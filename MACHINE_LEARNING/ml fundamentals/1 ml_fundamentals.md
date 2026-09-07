## What is Machine Learning?

Machine Learning is a way of building programs that **learn from data** 

### Traditional Programming vs Machine Learning

In **traditional programming**, a human writes the rules:

```
Input + Rules → Output

You write:
  IF price < 5000 AND size > 1000 THEN label = "good deal"
```

In **Machine Learning**, the program figures out the rules itself:

```
Input + Output → Rules (learned automatically)

You give the model:
  - 10,000 house listings
  - Whether each was a good deal or not
  → Model learns its own rules from the patterns
```
---

## ML vs Statistics vs Rule-Based Systems

These three approaches all work with data but have very different goals and philosophies.

| | Rule-Based | Statistics | Machine Learning |
|--|-----------|-----------|-----------------|
| **How it works** | Human-written if/else logic | Mathematical analysis of data | Learns patterns automatically from data |
| **Who writes the logic** | Developer | Analyst (interprets formulas) | The model itself |
| **Goal** | Follow fixed rules | Understand and explain data | Make accurate predictions |
| **Handles complexity** | Poorly | Moderately | Very well |
| **Interpretability** | Very easy | Easy to moderate | Often hard (black box) |
| **Example** | Tax calculation | Finding correlation between age and income | Recommending products |

### Key Distinctions

**Rule-Based vs ML:**
Rules are written by humans and break when reality changes. ML adapts as long as it sees new training data.

**Statistics vs ML:**
Statistics asks *"why does this happen?"* — it focuses on understanding and explaining. ML asks *"what will happen next?"* — it focuses on prediction accuracy, often sacrificing interpretability.

They overlap more than people think. Many ML algorithms (linear regression, logistic regression) are rooted in statistics. The difference is mostly in **intent and application**.

---

## When NOT to Use ML

ML is powerful, but it is not always the right tool. Reaching for it by default is a common mistake.

**Don't use ML when:**

- **The problem has clear, stable rules** — Tax calculations, unit conversions, sorting a list. A simple `if/else` or formula is faster, cheaper, and more reliable.

- **You have very little data** — ML needs examples to learn from. With 50 rows of data, a statistical formula or domain expert will outperform any model.

- **You need full explainability** — In medical diagnosis or legal decisions, you may need to explain exactly *why* a decision was made. Many ML models can't do this reliably.

- **The cost of being wrong is too high** — A spam filter getting it wrong is annoying. An ML model approving a bridge design getting it wrong is catastrophic. High-stakes decisions need human judgment and verified systems.

- **The relationship is straightforward** — If the pattern is simple and linear, a basic formula works fine. ML adds complexity without adding value.

---
# Types of Machine Learning 

## 1. Supervised Learning
The model learns from **labelled data** — meaning every training example comes with the correct answer already attached.

```
Input (x)  +  Correct Answer (y)  →  Model learns the mapping x → y
```

The word "supervised" means the labels is guiding the learning. The model keeps comparing its predictions to the correct answers and adjusts until it gets good at predicting.

### How It Works

```
Training Data (x, y pairs)
         ↓
    Model makes prediction ŷ
         ↓
    Compare ŷ with actual y
         ↓
    Calculate error
         ↓
    Adjust parameters
         ↓
    Repeat → Model improves
```

### Types of Supervised Learning

#### A) Regression — Predicting a Number

The output is a **continuous numerical value**.

| Algorithm | Example Use Case |
|-----------|-----------------|
| Linear Regression | Predicting house prices |
| Polynomial Regression | Predicting population growth |
| Ridge / Lasso Regression | Predicting salary with many features |

**Example:** Predict a house price based on size, location, and number of rooms. The output could be ₹45,00,000 or ₹82,00,000 — any number on a continuous scale.

#### B) Classification — Predicting a Category

The output is one of a **fixed set of classes**.

| Type | Classes | Example |
|------|---------|---------|
| Binary Classification | 2 (Yes/No) | Spam or Not Spam |
| Multi-Class Classification | 3+ | Cat, Dog, or Bird |
| Multi-Label Classification | Multiple at once | A photo tagged: "beach", "sunset", "people" |

**Example:** Given an email, predict whether it is Spam (1) or Not Spam (0). The model outputs a category, not a number.

**Common algorithms:** Logistic Regression, Decision Trees, Random Forest, SVM, Neural Networks.

### When to Use Supervised Learning

- You have labelled data (even a few thousand examples helps)
- The problem is prediction or classification
- You want the model to generalise to new, unseen inputs

---

## 2. Unsupervised Learning
The model learns from **unlabelled data** — no correct answers are provided. The model must find **hidden structure, patterns, or groupings** in the data on its own.

```
Input (x) only  →  Model finds patterns by itself
```
### Types of Unsupervised Learning

#### A) Clustering — Grouping Similar Things Together

The model groups data points so that items **within a group are similar** and items **across groups are different**.

| Algorithm | How It Works |
|-----------|-------------|
| K-Means | Divides data into K groups based on distance from centre points |
| DBSCAN | Groups points that are densely packed together, ignores outliers |
| Hierarchical Clustering | Builds a tree of clusters from bottom up |

**Example:** A supermarket has 10,000 customers. No labels exist. K-Means groups them into:
- Group A: Young, frequent buyers, mostly snacks
- Group B: Families, large monthly purchases, bulk items
- Group C: Elderly, small weekly purchases, specific brands

The model discovered these groups

#### B) Dimensionality Reduction — Compressing Information

Reduces the number of features while keeping the most important information. Used when data has hundreds of columns and most are redundant or correlated.

| Algorithm | Use Case |
|-----------|----------|
| PCA (Principal Component Analysis) | Compress 100 features into 5 key ones |
| t-SNE | Visualise high-dimensional data in 2D |
| Autoencoders | Compress images for storage or denoising |

**Example:** A dataset of face images has 10,000 pixels per image. PCA compresses each image to 50 key values while retaining the essential structure — making it faster to process without losing meaningful information.

#### C) Anomaly Detection — Finding the Unusual

The model learns what "normal" looks like, then flags anything that deviates significantly.

**Example:** A bank trains a model on millions of normal transactions. When a transaction looks unlike everything it has seen — unusual amount, unusual location, unusual time — it flags it as potential fraud.

#### D) Association Rule Learning — Finding What Goes Together

Discovers which items or events frequently appear together.

**Example:** Supermarket basket analysis:
```
People who buy bread and eggs → also buy butter (78% of the time)
People who buy diapers → also buy beer on weekends (famous retail finding)
```
Used for product recommendations and store shelf layout.

### When to Use Unsupervised Learning

- You have a lot of data but no labels
- You want to explore and understand the structure of your data
- You want to reduce data complexity before applying supervised learning

---

## 3. Semi-Supervised Learning

### What Is It?

A middle ground between supervised and unsupervised learning. The model is trained on a **small amount of labelled data** combined with a **large amount of unlabelled data**.

```
Small labelled dataset  +  Large unlabelled dataset  →  Better model
```

### Why Does This Exist?

Labelling data is **expensive and time-consuming**. It requires human experts. Unlabelled data, on the other hand, is often abundant and cheap to collect.

Semi-supervised learning gets most of the benefit of labelled training without needing to label everything.

### How It Works (Intuition)

1. Train on the small labelled set first
2. Use the model to **pseudo-label** the unlabelled data (make predictions and treat confident ones as labels)
3. Retrain on the combined dataset
4. Repeat — the model gradually improves as it labels more data confidently

### Real-World Example

**Google Photos Face Recognition:**
- You manually label 20 photos of your friend Priya
- Google then scans 5,000 unlabelled photos, finds faces that look like Priya's, and clusters them together
- You confirm a few more → the model now recognises Priya in all 5,000 photos

A human labelled 20. The model effectively handled 5,000.

### When to Use Semi-Supervised Learning

- Collecting raw data is easy but labelling is expensive or slow
- You have domain experts who can label a small but high-quality subset
- You want significantly better performance than unsupervised alone

---

## 4. Self-Supervised Learning

The model **creates its own labels from the raw data** — no human labelling needed at all. It learns by solving a **pretext task**: a puzzle designed so the model must understand the data deeply to solve it.

```
Raw Data  →  Model generates its own labels  →  Learns rich representations
```
### How It Works

A pretext task is designed so that the answer is hidden inside the data itself:

**Example 1 — Next Word Prediction (Language Models)**
```
Input:  "The cat sat on the ___"
Label:  "mat"  (hidden from model during training, already in the text)
```
The model reads billions of sentences, predicts the next word each time, and learns the structure of language deeply. This is exactly how GPT models (like the one you're talking to) are trained.

**Example 2 — Masked Image Modelling**
```
Take an image → Hide 75% of the pixels
→ Model must reconstruct the hidden pixels
→ To do this well, it must understand what objects look like
```

**Example 3 — Contrastive Learning (SimCLR, CLIP)**
```
Take a photo of a cat
→ Create two different crops/augmentations of it
→ Train the model to recognise they are the same image
→ Train the model to distinguish them from random other images
→ Model learns what makes a cat a cat
```

### Why It Matters

Self-supervised learning has enabled the biggest breakthroughs in modern AI:
- **GPT, BERT, LLaMA** — trained self-supervisedly on text
- **CLIP** — understands images and text together
- **AlphaFold** — learned protein structures

It removes the bottleneck of human labelling and scales to internet-sized data.

### When to Use Self-Supervised Learning

- You have massive amounts of raw, unlabelled data
- You want to pre-train a model that can later be fine-tuned on a small labelled set
- Building foundation models or embeddings

---

## 5. Reinforcement Learning
An agent learns by **interacting with an environment**, taking actions, and receiving **rewards or penalties**. There is no dataset — the model learns by trial and error.

```
Agent  →  takes Action  →  Environment changes  →  Agent gets Reward/Penalty
                                                          ↓
                                              Agent learns to maximise reward
```

### Key Components

| Component | Meaning | Example (Chess) |
|-----------|---------|----------------|
| **Agent** | The learner/decision-maker | The chess-playing AI |
| **Environment** | The world the agent interacts with | The chessboard |
| **State** | Current situation | Current positions of all pieces |
| **Action** | What the agent can do | Moving a piece |
| **Reward** | Feedback signal | +1 for winning, -1 for losing, 0 otherwise |
| **Policy** | Strategy the agent follows | Which move to make in each position |

### How It Learns

The agent starts knowing nothing. It tries random actions, observes what happens, and gradually learns which actions lead to more reward over time. This is called **exploration vs exploitation**:

- **Exploration:** Try new actions to discover what works
- **Exploitation:** Use what you already know works

Early in training, the agent explores a lot. As it learns, it exploits more.

### Real-World Examples

- **Game playing:** AlphaGo (beat world Go champion), OpenAI Five (beat pro Dota 2 players)
- **Robotics:** Teaching a robot arm to grasp objects by rewarding successful picks
- **Self-driving cars:** Rewarding staying in lane, penalising collisions
- **Recommendation systems:** Rewarding clicks and watch time

### Why It's Different from Other Types

In supervised learning, you need a labelled dataset. In RL, there is no dataset — the agent **generates its own experience** by acting in the world. This makes it powerful for problems where the right answer is unknown in advance but success or failure can be measured.

### When to Use Reinforcement Learning

- The problem involves sequential decisions over time
- You can define a clear reward signal
- A simulation or environment exists to train in
- Examples: games, robotics, optimisation problems

---

## 6. Batch Learning vs Online Learning

This is not a different type of ML in terms of algorithm — it is about **how and when** the model is trained on new data.

---

### Batch Learning (Offline Learning)

The model is trained on the **entire dataset at once**, then deployed. After deployment, it does not update itself. To improve it, you retrain from scratch on new data.

```
Collect all data → Train model → Deploy → (static until retrained)
```
**Characteristics:**
- Training happens offline, before deployment
- Computationally expensive (processes all data each time)
- Simple and stable — predictions are consistent
- Must be periodically retrained if the world changes

**Example:** A house price prediction model trained on last year's data. Every 6 months, you retrain it on the latest data and redeploy.

**Best for:**
- Data that doesn't change rapidly
- When training can be done overnight or on a schedule
- When consistency and stability matter more than freshness

---

### Online Learning (Incremental Learning)

The model is **continuously updated** as new data arrives — one example (or small batch) at a time. The model never stops learning.

```
New data arrives → Model updates immediately → Keeps improving in real-time
```

**Analogy:** A student who learns something new every day and keeps updating their knowledge, rather than studying everything at once before an exam.

**Characteristics:**
- Model adapts to new patterns in real-time
- Efficient — doesn't need to reprocess old data
- Risk: bad data or sudden shifts can quickly degrade the model (requires monitoring)
- Uses a **learning rate** to control how fast it adapts to new data

**Example:** Stock price prediction — the market changes every second. An online learning model updates with each new tick of data, staying current rather than relying on yesterday's patterns.

**Best for:**
- Fast-changing environments (stock markets, social media trends)
- Data arriving continuously in a stream
- When storage of all historical data is impractical

---

### Batch vs Online — Side by Side

| | Batch Learning | Online Learning |
|--|---------------|----------------|
| **When it trains** | Once, on full dataset | Continuously, on new data |
| **Adapts to change** | Only after retraining | Immediately |
| **Compute cost** | High (processes all data) | Low per update |
| **Stability** | High | Can be unstable if data quality drops |
| **Use case** | Monthly sales forecasting | Real-time fraud detection |
| **Risk** | Stale model | Corrupted model from bad data |

---

## Summary — All Types at a Glance

| Type | Data Needed | Learns From | Best For |
|------|-------------|-------------|----------|
| **Supervised** | Labelled (x, y) | Correct answers | Prediction, classification |
| **Unsupervised** | Unlabelled (x only) | Structure in data | Clustering, compression, anomaly detection |
| **Semi-Supervised** | Mostly unlabelled + some labelled | Both | When labelling is expensive |
| **Self-Supervised** | Raw data only | Self-generated labels | Language models, vision models |
| **Reinforcement** | No dataset — trial and error | Rewards from environment | Games, robotics, sequential decisions |
| **Batch** | Full dataset upfront | All data at once | Stable, infrequent retraining |
| **Online** | Stream of new data | One example at a time | Real-time, fast-changing environments |
