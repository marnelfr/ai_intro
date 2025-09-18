## [MIT Introduction to Deep Learning - 2025](https://www.youtube.com/watch?v=alfdI7S6wCY)

### Definition
**Intelligence** is the ability to process information in order to inform some 
future decisions or actions.\
> Meaning take decisions or make actions based on information we've got.

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
process with _deep neural networks_.\
> Meaning teach computers how to learn a task directly from raw data

### Why deep learning?
Traditional **machine learning** typically defines a set of features.\
A **feature** is a set of rules which define how to do a task step by step; which is not
that easy, and even more complicated when it comes to robust features.

With **deep leaarning**, instead of defining these features, the key idea here is to let the system
get to these features automatically by observing the data.\
For example, when looking at faces, it can figure out step by step in a hierarchical way which 
patterns to detect first, without us explicitly telling it what those features are.

### Why now?
Even if the deep learning techniques are decade olds, we're now experiencing an explosion of those 
techniques because of today's availability of big data (larger datasets, easier to collect and 
store,...), more powerful hardwares (GPUs, massively parallelizable,...), and the evolution of 
how we _deal_ with software (opensource toolboxes such as PyTorch and Tensorflow, improved 
techniques,...)


### The perceptron—The structural (fundamental) building block of deep learning
A **perceptron or single neuron**

we are also going to add this one number called a bias term which effectively a way for us to shift left and right along our activation function G

👉 In general: bias = baseline output when all inputs are zero.

How to you passe information through a neuron: take a dot product, apply a bias, and you apply a
non-linearity
> y = g(b + X^T.W)
























