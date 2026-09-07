# Neural Networks
A neural network is a machine learning model loosely inspired by how the human brain works. The brain contains billions of neurons connected to each other, passing signals back and forth to process information. A neural network borrows this idea and builds it in software using numbers and math.

The key thing that makes neural networks special is that they can **learn complex patterns automatically** from data without you manually engineering features. You feed in raw inputs, and the network figures out on its own what combinations of those inputs are meaningful.

They power almost everything impressive in modern AI: image recognition, language models, speech recognition, game playing, drug discovery, and much more.

---

## The Biological Inspiration

A biological neuron receives signals through dendrites, processes them in the cell body, and fires an output signal through the axon if the combined input is strong enough.

An artificial neuron mirrors this:

```
Inputs          Weights         Neuron          Output

x1 ----w1---->  |           |
x2 ----w2---->  |  sum + b  | --> activation --> ŷ
x3 ----w3---->  |           |
```

Each input $x$ is multiplied by a weight $w$, all the weighted inputs are summed together with a bias $b$, and the result is passed through an activation function to produce an output. That output becomes the input to the next layer.

---

## A Single Neuron - The Building Block
A single neuron computes:

$$
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

$$
a = g(z)
$$

Where:
- $z$ is the weighted sum (also called the pre-activation)
- $g$ is the activation function
- $a$ is the output of the neuron (also called activation)

This is exactly logistic regression if you use the sigmoid as $g$. In fact, logistic regression is just a single neuron with a sigmoid activation. A neural network is many of these neurons stacked together.

---

## Common Terms in Neural Networks
**Neuron (Node):** The basic unit. Takes inputs, computes a weighted sum, applies an activation function, outputs a value.

**Weight ($w$):** A learnable parameter that scales each input. The network adjusts weights during training to improve predictions.

**Bias ($b$):** A learnable offset added to the weighted sum. It allows the neuron to shift its activation even when all inputs are zero.

**Activation ($a$):** The output of a neuron after applying the activation function. It is what gets passed to the next layer.

**Activation Function ($g$):** A mathematical function applied to the weighted sum to introduce non-linearity. Without it, the entire network collapses to a single linear equation regardless of depth.

**Layer:** A group of neurons that all process inputs at the same level and pass their outputs to the next group.

**Parameters:** All the weights and biases in the network. These are what get learned during training.

**Hyperparameters:** Settings you choose before training — number of layers, number of neurons per layer, learning rate, activation function. They are not learned from data.

**Forward Pass:** The process of passing input through the network layer by layer to get a prediction.

**Loss:** How wrong a single prediction is.

**Cost:** Average loss across the entire training set.

**Epoch:** One full pass through the entire training dataset during training.

**Batch:** A subset of the training data used in one gradient descent update. Mini-batch training uses small batches (e.g. 32 or 64 examples) rather than the whole dataset at once.

---

## Layers — The Structure of a Neural Network

A neural network is organised into layers. Each layer transforms its inputs and passes the result to the next one.

### Input Layer

This is not really a layer that computes anything. It just holds the raw input features and passes them into the first real layer.

If your input has 4 features (house size, rooms, age, location score), the input layer has 4 nodes, one per feature.

### Hidden Layers

These are the layers between the input and output. They are called "hidden" because their values are not directly visible in the data  you do not observe them, the network creates them internally.

Hidden layers are where the magic happens. Each neuron in a hidden layer learns to detect some pattern or combination of the input features. Early hidden layers typically detect simple patterns. Deeper layers combine those simple patterns into more complex ones.

**Example in image recognition:**
- Layer 1 neurons learn to detect edges and corners
- Layer 2 neurons combine edges into shapes (circles, rectangles)
- Layer 3 neurons combine shapes into object parts (eyes, wheels, doors)
- Layer 4 neurons combine parts into full objects (face, car)

You did not tell the network to do this. It learned this hierarchy automatically from data.

### Output Layer

The final layer produces the network's prediction. The number of neurons and the activation function here depend on the task.

| Task | Output Neurons | Activation |
|------|---------------|-----------|
| Binary classification | 1 | Sigmoid (outputs probability 0 to 1) |
| Multi-class classification | One per class | Softmax (outputs probabilities summing to 1) |
| Regression | 1 | None or linear (outputs any real number) |

### Visualising the Structure

```
Input         Hidden         Hidden         Output
Layer         Layer 1        Layer 2        Layer

  x1  o                                        
       \       o   o                           
  x2  o  ---  o   o  ---   o   o  ---  o  -->  ŷ
       /       o   o                           
  x3  o                                        

4 inputs    4 neurons      3 neurons      1 output
```

Each line represents a weight connection. Every neuron in one layer connects to every neuron in the next layer. This is called a **fully connected** or **dense** layer.

---
# Activation Functions
An activation function is a mathematical function applied to the output of a neuron after computing the weighted sum. It decides whether and how strongly a neuron should "fire"  that is, what value it passes forward to the next layer.

Without an activation function, every layer in a neural network would just compute a linear transformation of its inputs. A linear transformation of a linear transformation is still just a linear transformation. No matter how many layers you stack, the entire network would behave like a single linear equation  no more powerful than logistic regression. Activation functions break this limitation by introducing **non-linearity**, which is what allows deep networks to learn complex patterns.

There are two broad categories of activation functions. Some are used in hidden layers to help the network learn rich internal representations. Others are used in the output layer to shape the final prediction into the right format for the task.
<img width="1226" height="604" alt="Screenshot 2026-07-06 215926" src="https://github.com/user-attachments/assets/0dbfb090-3e70-43b5-aadd-7a9405b7218d" />

## 1. Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Output range: 0 to 1

The sigmoid takes any real number and squashes it into a value between 0 and 1. When $z$ is a large positive number, the output approaches 1. When $z$ is a large negative number, the output approaches 0. At $z = 0$, the output is exactly 0.5.

### Where It Is Used

The sigmoid is most naturally used in the **output layer for binary classification**, where you want the network to output a probability between 0 and 1. For example, predicting whether an email is spam (1) or not spam (0).

### Problems with Sigmoid in Hidden Layers

**Vanishing gradient:** When $z$ is very large or very small, the sigmoid curve becomes almost flat. The gradient at those regions is nearly zero. During backpropagation, gradients are multiplied layer by layer, so passing near-zero gradients back through many layers causes them to shrink exponentially. By the time they reach the early layers, the update is so small that those layers barely learn anything. This is called the vanishing gradient problem and it makes training deep networks with sigmoid activations very slow.

**Not zero-centred:** Sigmoid outputs are always positive (between 0 and 1), never negative. This can cause gradients during backpropagation to always be the same sign, which leads to inefficient zigzag updates during optimisation.

For these reasons, sigmoid is mostly avoided in hidden layers today and reserved for the output layer of binary classifiers.

---

## 2. Tanh (Hyperbolic Tangent)
<img width="894" height="516" alt="image" src="https://github.com/user-attachments/assets/95ff2932-b125-417e-bdd1-3d245f72dab2" />

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

Output range: -1 to 1

Tanh is similar to sigmoid in shape but stretched to output values between -1 and 1 instead of 0 and 1. The key improvement is that tanh is **zero-centred** — its output ranges symmetrically around zero. This makes gradients during backpropagation more balanced and generally leads to faster convergence than sigmoid.

### Where It Is Used

Tanh is sometimes used in **hidden layers** as an alternative to ReLU, particularly in recurrent neural networks (RNNs) where the zero-centred property is important. It also appears as the output activation in tasks where predictions should range between -1 and 1.

### Problems with Tanh

Tanh still suffers from the **vanishing gradient problem** for large positive or large negative values of $z$, where the curve flattens out. The gradients at those regions approach zero, slowing down learning in deep networks. It is better than sigmoid in this regard but not fully immune.

---

## 3. ReLU (Rectified Linear Unit)
<img width="910" height="387" alt="image" src="https://github.com/user-attachments/assets/d45d975a-bac7-40ac-ac4c-b23ecc4bb7a7" />

$$
g(z) = \max(0, z)
$$

Output range: 0 to infinity

ReLU is the most widely used activation function in hidden layers today. Its rule is simple: if $z$ is positive, output $z$ unchanged. If $z$ is negative, output 0.

### Why ReLU Became the Default

**No vanishing gradient for positive values.** For any positive $z$, the gradient of ReLU is exactly 1. This means gradients flow back through the network without shrinking, allowing deep networks to train effectively.

**Computationally cheap.** ReLU is just a comparison and a clamp. No exponentials, no divisions. It is extremely fast to compute, which matters when you have millions of neurons.

**Sparsity.** For any neuron where $z$ is negative, ReLU outputs exactly 0. In practice, roughly half the neurons in a ReLU network output zero at any given time. This sparsity makes the network more efficient and can act as a mild regularizer.

### Where It Is Used

ReLU is the default choice for **hidden layers** in most feedforward and convolutional neural networks. When in doubt about which activation to use in a hidden layer, start with ReLU.

### Problems with ReLU

**Dying ReLU.** If a neuron's weights get pushed into a state where $z$ is always negative for every training example, the neuron will always output 0. Its gradient is also 0, so it never receives a useful update and never recovers. The neuron is effectively dead for the rest of training. This can happen with high learning rates or poorly initialised weights.

**Not zero-centred.** Like sigmoid, ReLU outputs are always non-negative, which can cause the same zigzag gradient update issue.

---

## 4. Leaky ReLU
<img width="900" height="433" alt="image" src="https://github.com/user-attachments/assets/f69ae1a4-1076-405a-9aa5-20f739d7973f" />

$$
g(z) = \max(0.01z, z)
$$

Output range: negative infinity to positive infinity (but with a very small slope for negative inputs)

Leaky ReLU is a direct fix for the dying ReLU problem. Instead of outputting exactly zero for negative $z$, it outputs a small negative value proportional to $z$ — typically $0.01z$. This small slope keeps the gradient alive even for negative inputs, so the neuron can still receive updates and potentially recover.

### Where It Is Used

Leaky ReLU is used in **hidden layers** as a drop-in replacement for ReLU, especially in situations where dying neurons are observed or suspected.

### Parametric ReLU (PReLU)
<img width="607" height="484" alt="image" src="https://github.com/user-attachments/assets/d4a0f769-f881-4bfa-a8e8-058627095292" />

A variation where the slope for negative values is not fixed at 0.01 but is learned during training. This gives the network one more learnable parameter per neuron, allowing it to determine the optimal slope for negative inputs automatically.

$$
g(z) = \max(\alpha z, z)
$$

Where $\alpha$ is learned, not set manually.

---

## 5. ELU (Exponential Linear Unit)
<img width="910" height="465" alt="image" src="https://github.com/user-attachments/assets/4105772f-4f3e-4e3d-90cd-c78965c0aad2" />

$$
g(z) =
\begin{cases}
z & \text{if } z > 0 \\
\alpha(e^z - 1) & \text{if } z \leq 0
\end{cases}
$$

Output range: $-\alpha$ to positive infinity (typically $\alpha = 1$)

ELU is similar to Leaky ReLU but uses a smooth exponential curve for negative inputs instead of a straight line. This smooth transition through zero has two advantages over ReLU: it reduces the dying neuron problem and it pushes the mean activation closer to zero, which speeds up learning.

### Where It Is Used

ELU is used in **hidden layers** when you want the benefits of ReLU but with smoother negative-side behaviour. It is more computationally expensive than ReLU (because of the exponential) but can produce better results on some tasks.

---

## 6. Swish
<img width="532" height="457" alt="image" src="https://github.com/user-attachments/assets/332725aa-45bd-4ff2-8224-fa963bebde45" />

$$
g(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
$$

Output range: roughly -0.28 to positive infinity
Swish was proposed by Google in 2017. It is the product of the input and its own sigmoid value. Unlike ReLU which has a hard zero cutoff, Swish has a smooth, slightly non-monotonic shape — it dips just below zero for small negative inputs before approaching zero from below as $z$ becomes more negative.

In practice, Swish has been shown to outperform ReLU on some deep networks, particularly very deep ones. The intuition is that the smooth non-monotonic shape provides a richer gradient signal than the hard cutoff of ReLU.

### Where It Is Used

Swish is used in **hidden layers** of very deep networks, notably in EfficientNet and some Transformer-based architectures. It is not yet as universally adopted as ReLU but is worth knowing.

---

## 7. GELU (Gaussian Error Linear Unit)
<img width="553" height="478" alt="image" src="https://github.com/user-attachments/assets/ad0d21dc-b083-421d-9ab0-41e11899306a" />

$$
g(z) = z \cdot \Phi(z)
$$

Where $\Phi(z)$ is the cumulative distribution function of the standard normal distribution. In practice it is approximated as:

$$
g(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(z + 0.044715z^3\right)\right]\right)
$$

Output range: roughly -0.17 to positive infinity

GELU is similar in shape to Swish — smooth, slightly non-monotonic, and allowing small negative outputs. The key difference is that GELU weights the input by the probability of it being positive under a Gaussian distribution, which gives it a principled probabilistic interpretation.

### Where It Is Used

GELU is the activation function used in **Transformer architectures** including BERT, GPT, and most modern large language models. It has become the standard in natural language processing (NLP) and is increasingly used in vision transformers as well.

---

## 8. Softmax

$$
\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

Output range: 0 to 1 for each output, and all outputs sum to exactly 1.

Softmax is different from all the others. It does not operate on a single neuron in isolation. It operates on the **entire output layer at once**, converting a vector of raw scores into a probability distribution.

### How It Works

Say you are classifying an image into one of three classes: cat, dog, or bird. The output layer has 3 neurons with raw scores:

$$
z = [2.5, \; 1.0, \; 0.1]
$$

Softmax converts these into probabilities:

$$
\text{softmax}([2.5, 1.0, 0.1]) = \left[\frac{e^{2.5}}{e^{2.5}+e^{1.0}+e^{0.1}}, \; \frac{e^{1.0}}{e^{2.5}+e^{1.0}+e^{0.1}}, \; \frac{e^{0.1}}{e^{2.5}+e^{1.0}+e^{0.1}}\right]
$$

$$
\approx [0.70, \; 0.21, \; 0.09]
$$

The model predicts 70% cat, 21% dog, 9% bird. The predicted class is cat. All probabilities sum to 1.

### Where It Is Used

Softmax is exclusively used in the **output layer for multi-class classification** problems where the classes are mutually exclusive (each input belongs to exactly one class).

---

## 9. Linear (No Activation)

$$
g(z) = z
$$

This is just the identity function — the output equals the input, no transformation applied.

### Where It Is Used

The linear activation (or no activation at all) is used in the **output layer for regression tasks**, where the network needs to predict any real number without range restriction. Predicting house prices, temperature, or stock returns all require unconstrained output values.

---

## Choosing the Right Activation Function

| Location | Task | Recommended Activation |
|----------|------|------------------------|
| Hidden layers | Most feedforward networks | ReLU |
| Hidden layers | When dying ReLU is a problem | Leaky ReLU or ELU |
| Hidden layers | Very deep networks | Swish or GELU |
| Hidden layers | RNNs and some special cases | Tanh |
| Hidden layers | Transformer models (NLP) | GELU |
| Output layer | Binary classification | Sigmoid |
| Output layer | Multi-class classification | Softmax |
| Output layer | Regression | Linear (none) |

---

## Side by Side Comparison

| Activation | Output Range | Zero-Centred | Vanishing Gradient | Dying Neurons | Main Use |
|------------|-------------|--------------|-------------------|--------------|----------|
| Sigmoid | 0 to 1 | No | Yes | No | Output: binary classification |
| Tanh | -1 to 1 | Yes | Yes (less severe) | No | Hidden: RNNs |
| ReLU | 0 to inf | No | No (positive side) | Yes | Hidden: default choice |
| Leaky ReLU | -inf to inf | No | No | No | Hidden: ReLU fix |
| ELU | -α to inf | Approximately | No | No | Hidden: smooth alternative |
| Swish | ~-0.28 to inf | No | No | No | Hidden: deep networks |
| GELU | ~-0.17 to inf | No | No | No | Hidden: Transformers |
| Softmax | 0 to 1 (sum = 1) | No | No | No | Output: multi-class |
| Linear | -inf to inf | Yes | No | No | Output: regression |


| Concept | What It Means |
|---------|--------------|
| Activation function | Non-linear transformation applied after the weighted sum in a neuron |
| Non-linearity | What makes deep networks capable of learning complex patterns |
| Vanishing gradient | Gradients shrinking to near zero as they flow back through deep layers |
| Dying ReLU | Neurons permanently stuck at zero output, receiving no gradient updates |
| Zero-centred | Output values balanced around zero, helping gradient flow stay symmetric |
| Softmax | Converts raw output scores into a probability distribution summing to 1 |
| Sigmoid | Squashes output to 0 to 1, used for binary probability output |
| ReLU | Default hidden layer activation, fast and effective for most tasks |
| GELU | Smooth probabilistic activation, standard in Transformer models |

## How a Neural Network Works - Forward Propagation

Forward propagation is the process of computing a prediction by passing the input through every layer one at a time from left to right.
Forward propagation is simply:

Taking the input, passing it through every layer of the neural network, and producing the final prediction.

Nothing is learned here. No weights are updated. No gradient descent happens.

It is only:
```
Input
   ↓
Multiply by weights
   ↓
Add bias
   ↓
Activation Function
   ↓
Repeat for every layer
   ↓
Prediction
```
### Notation

For a network with multiple layers, use superscripts in square brackets to denote the layer:

- $a^{[l]}$ is the activation of layer $l$
- $W^{[l]}$ is the weight matrix of layer $l$
- $b^{[l]}$ is the bias vector of layer $l$

### Step by Step

For a 3-layer network (2 hidden layers, 1 output layer):

**Layer 1:**

$$
Z^{[1]} = W^{[1]} X + b^{[1]}
$$

$$
A^{[1]} = g^{[1]}(Z^{[1]})
$$

**Layer 2:**

$$
Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}
$$

$$
A^{[2]} = g^{[2]}(Z^{[2]})
$$

**Output Layer:**
$$
Z^{[3]} = W^{[3]} A^{[2]} + b^{[3]}
$$

$$
\hat{Y} = g^{[3]}(Z^{[3]})
$$

Each layer takes the previous layer's activations as input, computes a weighted sum, and applies an activation function. The final output $\hat{Y}$ is the network's prediction.

### A Concrete Example

A network predicting whether a customer will churn (1) or not (0).

Input features: account age, monthly spend, number of complaints, last login days ago.

```
Input:  [24, 85.5, 2, 7]   (account age, spend, complaints, days since login)

Layer 1: 4 inputs  →  3 neurons  →  ReLU  →  3 activations
Layer 2: 3 inputs  →  2 neurons  →  ReLU  →  2 activations
Output:  2 inputs  →  1 neuron   →  Sigmoid  →  probability of churn
```

Final output: 0.83 — the model predicts 83% probability the customer will churn.

---

## How the Network Learns 

Training a neural network involves three steps repeated many times:

**Step 1 — Forward pass:** Pass training examples through the network to get predictions.

**Step 2 — Compute cost:** Compare predictions to actual labels using a cost function (log loss for classification, MSE for regression).

**Step 3 — Backpropagation:** Compute how much each weight contributed to the error, then update all weights using gradient descent.

Backpropagation works by applying the chain rule of calculus to propagate error signals backwards through the network, from the output layer all the way back to the first hidden layer. This tells each weight: "you made the error larger" or "you made the error smaller" and by how much.

The weights are then nudged in the direction that reduces the cost. This whole cycle repeats for many epochs until the cost is minimised.

---

## Inference - Using a Trained Network

Inference is the term for using a trained neural network to make predictions on new data. It is just a forward pass — you feed in the input and the network outputs a prediction. No learning happens during inference, the weights stay fixed.

```
New input arrives
      |
      v
Forward pass through trained network
(weights are frozen, no updates)
      |
      v
Output: prediction or probability
      |
      v
Apply threshold or take argmax
      |
      v
Final class label or value
```

### Training vs Inference

| | Training | Inference |
|--|---------|---------|
| Data | Labelled training set | New, unseen data |
| Weights | Updated on every step | Frozen, no updates |
| Direction | Forward pass + backward pass | Forward pass only |
| Cost | Computed and minimised | Not needed |
| Speed | Slower (needs backprop) | Fast (forward only) |
| Hardware | GPU strongly preferred | CPU often sufficient |

Inference is much faster than training because there is no backpropagation happening. A model that took hours to train can make predictions in milliseconds.

---

## Architecture — Putting It All Together

The term "architecture" refers to the overall design of the network:
- How many layers
- How many neurons per layer
- What activation functions to use
- How layers are connected

There is no single right answer. Architecture is chosen based on the problem, the data, and experimentation.

**Shallow network:** One or two hidden layers. Works well for structured tabular data (spreadsheet-style data like house prices or customer records).

**Deep network:** Many hidden layers (3 or more). Needed for complex data like images, audio, and text. More depth allows the network to learn increasingly abstract representations.

**Wide network:** Many neurons per layer. More capacity per layer.

In practice, you start with a reasonable architecture and adjust based on whether the model is underfitting or overfitting.

---

## Quick Concept Summary

| Term | What It Means |
|------|--------------|
| Neuron | A single computing unit: weighted sum plus activation |
| Weight | Learnable parameter scaling each input connection |
| Bias | Learnable offset allowing the neuron to shift its output |
| Activation function | Non-linear function applied after the weighted sum |
| Input layer | Holds raw input features, no computation |
| Hidden layer | Intermediate layers that learn internal representations |
| Output layer | Produces the final prediction |
| Forward propagation | Passing input through layers to get a prediction |
| Backpropagation | Propagating error backwards to compute gradients |
| Epoch | One full pass through the training dataset |
| Batch | A subset of data used for one gradient update |
| Inference | Using a trained model to predict on new data |
| Architecture | The overall design: number of layers, neurons, activations |
