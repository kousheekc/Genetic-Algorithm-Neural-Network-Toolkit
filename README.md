# Genetic-Algorithm-Neural-Network-Toolkit
Genetic Algorithm and Neural Network Toolkit for Reinforcement Learning applications

## Requirements:
* [Matplotlib](https://matplotlib.org/)
* [Numpy](https://numpy.org/)
* [OpenAI gym](https://gym.openai.com/)

## Neural Networks:
A **neural network** is a network of neurons connected to one another using synapses that carry weights. The responsibilty of the synapse is to scale the input by the weight its carrying (usually a number between 0 and 1). The neurons on the other hand sum the inputs together and pass this value through an activation function. A neural network can be comprised of multiple layers and is called a **deep neural network**. A single layer of a neural network can be mathematically represented as ![\overrightarrow{z}=f(W\cdot\overrightarrow{x})](https://render.githubusercontent.com/render/math?math=%5Coverrightarrow%7Bz%7D%3Df(W%5Ccdot%5Coverrightarrow%7Bx%7D)) where ![\overrightarrow{z}](https://render.githubusercontent.com/render/math?math=%5Coverrightarrow%7Bz%7D) is the output, ![\overrightarrow{x}](https://render.githubusercontent.com/render/math?math=%5Coverrightarrow%7Bx%7D) is the input, ![W](https://render.githubusercontent.com/render/math?math=W) is the weight matrix of that layer and ![f](https://render.githubusercontent.com/render/math?math=f) represents the activation function. Therefore the aim of machine learning, is to determine the weight matrix such that an unknown input produces a valid output. An example would be **back propagation**, where the weights are initally random, but are adjusted iteratively by comparing the output with known data.    

## Genetic Algorithm:
Another approach for finding the optimal weights in a neural network for a certain task is by using a **genetic algorithm**. A population of agents are created each associated with a unique neural network initially having random weights.  
