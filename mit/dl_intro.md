## [MIT - Introduction to Deep Learning - 2025](https://www.youtube.com/watch?v=alfdI7S6wCY)

### Definition
**Intelligence** is the ability to process information in order to inform some 
future decisions or actions.
> ...meaning take decisions or make actions based on information we've got.

👇👇👇\
**Artificial Intelligence** is the practice of building (artificial) algorithm 
to do the same process: use information, use data, to inform future decisions/actions.\
👇👇\
**Machine learning** is a subset of artificial intelligence that focuses on not 
explicitly programming the computer on how to use/process that data or information 
to inform that decision but just try to learn some patterns within the data to make 
those decisions.\
👇\
**Deep learning** is a subset of machine learning which focuses on doing that exact 
process with _deep neural networks_.
> Meaning teach computers how to learn a task directly from raw data

### Why deep learning?
Traditional **machine learning** typically defines a set of features.\
A **feature** is a set of rules which define how to do a task step by step; which is not
that easy, and even more complicated when it comes to robust features.

With **deep learning**, instead of defining these features, the key idea here is to let the system
get to these features automatically by observing the data.\
For example, when looking at faces, it can figure out step by step in a hierarchical way which 
patterns to detect first, without us explicitly telling it what those features are.

> So deep learning is a type of machine learning that uses artificial neural networks to learn from
data. These networks are inspired by the structure of the human brain and are able to learn complex 
patterns from large amounts of data. Deep learning is used in a wide range of applications, including
image recognition, natural language processing, and self-driving carts.


### Why now?
Even if deep learning techniques are decade olds, we're now experiencing an explosion of those 
techniques because of today's availability of big data (larger datasets, easier to collect and 
store,...), more powerful pieces of hardware to compute (GPUs advent, massively parallelizable,...), and the evolution
of how we _deal_ with software (opensource toolboxes such as PyTorch and Tensorflow, improved 
techniques,...) that makes it very easy to get hands on deep learning. 


### The perceptron—The structural (fundamental) building block of deep learning
A **perceptron (single neuron)** is the simplest type of artificial neuron: it takes inputs, 
multiplies them by weights, adds a bias, and then passes the result through an activation 
function to decide the output.
<p align="center"><img src="./images/perceptron_image.png" width="330px"/></p>

> Basically: inputs → weighted sum + bias → activation → output.

What leads us to the formula:  
<p align="center"><img src="./images/perceptron_formula.png" width="250px"/></p>

Using vectors where the vector X is the collection of our inputs and the vector W, the
collection of our weights, we can simplify the formula of a perceptron to:
<p align="center">y = g(X.W + b)</p>

> Meaning that to pass information through a neuron, we take a dot product, apply a bias, and then 
apply a non-linearity

With `z = X.W + b`, we'll have `y = g(z)`.


#### Bias
It's a free number we add to the output of the weighted inputs in a perceptron.\
Without bias, `z = X.W` which is a line that must always pass through the origin (0,0).
That means no matter what weight `W` is, the line is stuck crossing that point. And this is a problem
because in real data, relationships between inputs and outputs don’t always pass through zero, so
our neuron can't always rotate around the origin.

**E.g.:**
Let's imagine we want to model how much coffee we drink based on how many hours of sleep we missed.\
Here, `Xs || Ws` can be how many hours of sleep we missed while the second one will be how many coffees
that will be drunk.

Then, the Bias here will be equal to how many coffees we naturally drink even in case of zero hour of 
missed sleep. 

> So in general: **bias** is the baseline output when all inputs are zero.


#### Activation Function G
The activation function G (like sigmoid, ReLU, tanh, etc.) decides the final output.\
it's the function that decides the output of a neuron by transforming its input into a useful 
signal by adding non-linearity, so the network can learn complex patterns.

> Activation functions introduce non-linearity, which lets neural nets model things like language,
images, and decision boundaries.

**E.g.:** Let's imagine we want to build a neural network to distinguish red points from green ones,
meaning we want, using a line, separate these red points from green ones:
<p align="center"><img src="./images/red_green_points.png" width="300px"/></p>


With a linear function, we can only produce linear decisions no matter the size of the network,
meaning we can only draw a strict line (1st image). But non-linearity allows us to approximate 
arbitrarily complex function, so end up with such of line in the second image.
<p align="center"><img src="./images/red_green_points_2.png"/></p>

**Common activation functions:**
- Sigmoid function `g(z) = 1/(1+e^(-z))`, produces output between 0 and 1 so is good for probabilities. 
- Hyperbolic Tangent (tanh) function `g(z) = (e^z - e^(-z))/(e^z + e^(-z))`, produces output between -1 and 1, Historically used in RNNs (recurrent neural networks).
- Rectified Linear Unit (ReLU) function `g(z) = max(0, z)`, can suffer from “dying ReLU” (neurons stuck outputting 0)
- Leaky ReLU function `g(z) = z > 0 ? z : 0.01z`, fixes “dying ReLU” by allowing small negative outputs.


### Building Neural Networks with Perceptrons
First, let's simplify the schema of a neuron:
<p align="center"><img src="./images/simplified_neuron.png" width="330px"/></p>

where as always:
<p align="center"><img src="./images/z_formula.png" width="200px"/></p>

If we want to get two outputs neron network, we only have to create a second perceptron, 
so we'll have two neurons instead of one. They will use the same inputs but **because their weights 
are different**, they'll produce two different outputs:
<p align="center"><img src="./images/multi_output_perceptron.png" width="330px"/></p>

Where:
<p align="center"><img src="./images/zi_formula.png" width="200px"/></p>

#### Layer
When you group multiple perceptrons working in parallel side by side on the same inputs, but each 
learning different weights, we get a **layer**. \
We talk about "**learning**" here because when we first create the neural network, each connection 
(weight) is just a random number. Then we show the network many examples (inputs and their correct 
outputs):
- The network makes a prediction with its current weights,
- compare the prediction to the correct answer, what gives error,
- then the network adjusts its weights a little bit to reduce that error, and again, and again,...
until it gets the correct output.

#### Dense Layer
When all inputs are densely connected to all outputs, these layers are called **Dense layers**.\
If you take your inputs and connect them to several neurons in parallel (each neuron producing 
one output), that whole thing is a layer.\
If there are 2 output neurons then, that’s a multi-output perceptron. But when generalized, 
we just call it a **dense layer with 2 units**.

#### Input layer
In big systems, the “inputs” might indeed be outputs of a previous model, so **input layer**.\
But even when they’re just raw features, many diagrams still label that column as the “input 
layer” for clarity of depth counting.

Inputs are often called _raw features_ because we actually have: 
> Raw data → (preprocess/encode) → Input features `X`

So when we say “inputs (features)”, it means the numeric input vector `X` the model 
consumes, which may be raw data (already numeric) or a numeric encoding of the raw data.

### Single hidden layer neural network
<p align="center"><img src="./images/single_hidden_neural_network.png"/></p>

To clearly pull out everything hidden here, let have a look at the formula of z<sub>i</sub> and y<sub>i</sub>:
<p align="center"><img src="./images/shlnn_zi_yi_formula.png"/></p>

As shown by the formula, we have the first layer "**Inputs -> Hidden**" where:
- Each hidden neuron does a weighted sum + bias,
- Then we apply the activation function g(z),

So the hidden layer outputs are g(z<sub>1</sub>), g(z<sub>2</sub>),... 
Then we have the second layer "**Hidden → Outputs**"
which:
- start with those activated values g(z<sub>j</sub>),
- do, using them, another weighted sum + bias,
- then we apply another activation function g to produce the final output y<sub>1</sub>, y<sub>2</sub>.

> We talk about **hidden layer** because we don't directly observe the data that happens in this 
layer. 

With the **dense layers** simplified, our neural network will become:
<p align="center"><img src="./images/simplified_dense_layer.png"/></p>

Such of network can be implemented in python using 
- Tensorflow
```python
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n), 
    tf.keras.layers.Dense(2)
])
```
- Pytorch
```python
import torch as nn
model = nn.Sequential([
    nn.Linear(m, n), 
    nn.ReLU(), 
    nn.Linear(n, 2)
])
```

### Deep Neural Network
<p align="center"><img src="./images/deep_neural_network.png"/></p>

Here is a deep (multi-layer) dense network where:
<p align="center"><img src="./images/deep_neural_network_zki.png"/></p>

Earlier layers learn simple features; deeper layers combine them into more abstract features.
The column labeled z<sub>k,1</sub>, z<sub>k,2</sub>, ..., z<sub>k,n_k</sub> is here, one hidden 
layer’s feature vector.

Data flows through a chain of blocks, each doing ``(W.X + b) → activation``, turning
inputs into increasingly useful features until the last block outputs y<sub>i</sub>.

**E.g.:** [testing/functional.py](./testing/functional.py).\
Let say that 
- input size is n<sub>0</sub>=m;
- Hidden sizes are respectively n<sub>1</sub>,n<sub>2</sub>, ...,n<sub>L-1</sub>;
- And output size n<sub>L</sub>.

Example of values:
```python
m = 5   # input features (n0)
n1 = 8  # neurons in first hidden layer
n2 = 4  # neurons in second hidden layer
k = 2   # output classes
```
Using these values, here is the exact network will get:
<p align="center"><img src="./images/network_example.png"/></p>

Here is the functional API for the model training:
```python
from tensorflow import keras
from tensorflow.keras import layers

inputs  = keras.Input(shape=(m,), name="x")                       # input layer
h1      = layers.Dense(n1, activation="relu", name="L1")(inputs)  # hidden layer 1
h2      = layers.Dense(n2, activation="relu", name="L2")(h1)      # hidden layer 2
outputs = layers.Dense(k, activation="softmax", name="Out")(h2)   # output layer

model = keras.Model(inputs, outputs)
model.summary()
```

Sequential API (same thing, more compact):
```python
from tensorflow.keras import Sequential, layers

model = Sequential([
    layers.Input(shape=(m,)),               # x
    layers.Dense(n1, activation="relu"),    # h^(1)
    layers.Dense(n2, activation="relu"),    # h^(2) (z_{k,i})
    layers.Dense(k,  activation="softmax")  # ŷ
])
model.summary()
```

- We use ReLU in hidden layers because fast, simple, and helps gradients flow (good default for 
dense nets).
- Softmax in the last layer: turns the final scores into a probability distribution across classes 
(sums to 1) — perfect for multi-class classification.

#### Hidden sizes
Hidden sizes control the model capacity:
- Bigger = more capacity (risk of overfitting, more compute).
- Smaller = less capacity (risk of underfitting).

Common (but optional) patterns:
- A "funnel" (e.g., 8 → 4) to compress information.
- Start small, increase until ``validation accuracy/loss`` stops improving.
- Set the last layer size to the number of outputs (here k=2).

#### [Loss](./loss.md)
It's a number that measures how bad the model’s predictions are.
- Higher loss = worse predictions.
- Lower loss = better predictions.

<p align="center"><i>L</i>(<i>f</i>(x<sup>i</sup>; <b>W</b>), y<sup>(i)</sup>)</p>

Where <i>f</i>(x<sup>i</sup>; <b>W</b>)</b> is the current prediction and <b>y<sup>(i)</sup></b>, 
the expected output.

There two kinds of loss
- Training loss: computed on the training data (the examples the model learns from).
- Validation loss: computed on separate data the model never sees during training (to test if it 
generalizes).

The **empirical loss** measures the total loss over our entire dataset:
<p align="center">J(W)=1/n(​∑<sup>n</sup><sub>i=1</sub>(​L(f(x(i);W),y(i)))</p>

#### Epoch, Batch
Epoch is one full pass through the entire training dataset during training.\
**E.g.:**\
Giving 10,000 training examples. One epoch means the model has seen all 10,000 examples once 
(possibly in mini-batches).

Training typically runs for multiple epochs, so the model sees the same data many times, gradually 
improving its weights.
However, training doesn’t usually feed the entire dataset at once (that wouldn’t fit in memory). 
Instead, we split it into **mini-batches**, a small subset of the data processed in one go (e.g., 
32 samples).

**Iteration/Step:** one parameter/weight update after processing a single batch.\

##### Concrete example
- N = 1,000 samples, batch size B = 100.
- Steps per epoch = N/B = 1000/100 = 10.
- If you train for 5 epochs, total steps = 10 x 5 = 50 total steps = 50 parameter updates.

#### Overfitting and Underfitting
- **Overfitting**: as epochs go by, the training loss goes down, and the validation loss goes up
(low training loss, high validation loss) → meaning that the model is memorizing the training data 
while failing on new data.\
How to fix it:
    1.	Simplify the model (fewer layers/units).
	2.	Regularize:
          - Dropout (e.g., 0.2–0.5 between dense layers).
          - L2 weight decay (a small value like 1e-4).
	3.	Early stop when validation loss stops improving.
	4.	More/augmented data (flip/rotate images, etc.).

- **Underfitting**: high training and validation loss → meaning that the model is too simple/not 
trained enough, not learning enough patterns.
	1.	Bigger model (more units/layers).
	2.	Train longer (more epochs).
	3.	Reduce regularization (lower dropout/L2).
	4.	Slightly higher learning rate (so it can learn faster), if it’s too small.





































