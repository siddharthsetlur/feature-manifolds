<!-- ---
date: 2025-10-19
authors: Siddharth Setlur
affiliations: University of Edinburgh
hide_search: false
hide_title_block: false
numbering:
  code: true
  math: true
  headings: true
export:
  - format: pdf
    template: lapreprint

--- -->
# An Introduction to Mechanistic Interpretability of Neural Networks 
Last week, we saw that transformer models (such as LLMs) are relatively simple architectures primarily built on simple linear algebra. Despite their simple mathematical building blocks, when scaled up (large amount of data, training for long periods of time, and increasing the number of layers) these models produce impressive results on a large range of tasks. Although scaling up in this manner delivers impressive results, they also exhibit a wide range of harmful behaviors such as hallucination (@hallucination2023) or harmful content among others. Moreover, the models perform in unpredictable manner making deployment in sensitive domains such as medicine or law where decisions can significantly impact human lives challenging. 

*Mechanistic interpretability (MI)* attempts to address these issues by *reverse engineering* transformer models by understanding small pieces of a model and how they interact with each other. By doing so, we hope to understand how these models represent features of interest and the internal computations by which they produce their outputs. If we were able to do this we could perhaps understand and prevent harmful behavior, e.g. by tuning weights appropriately. 

:::{note}
There are other interpretability techniques (@transparentAI2023), a couple of which we will explore later on in the reading course. One related approach is *explainable AI (xAI)*, which aims to produce human interpretable explanations for model behavior, e.g. by using tools from causal inference to intervene on weights and examine outputs. The focus here is on explaining model predictions, in contrast to MI which aims to analyze the inner workings of a model that produce these predictions. MI is sort of like a car mechanic who knows the individual components of a car (the gas pedal, levers, pistons, wheels, etc) and how they're connected to each other. They can use this knowledge to tell you that stepping on the gas speeds up the car. In contrast, xAI gives you an explanation that any (good) driver would - that intervening on the system by hitting the gas makes the car speed up (but it can't tell you the inner working that cause this). 
:::

# Transformers (a quick recap)
Last week, we saw that transformers are built from some very simple mathematics - primarily linear algebra. The following section primarily follows @elhage2021mathematical, which is an excellent reference for the mathematics of a transformer with many pretty pictures. Recall that given a token, a transformer first embeds this token, passes this embedding through a series of *residual blocks*, followed by a finally unembedding layer that yields a probability distribution on the following token. Each residual block consists of an attention layer and a multilayer perceptron (MLP). 

```{figure} ./images/transformer-architecture.png
:label: transformer-archi
:alt: A transfomer architecture
:align: center

A transfomer architecture (@elhage2021mathematical).
```
Rather than overwriting the results from previous layers, each layer of the residual block “reads” its input from the residual stream (by performing a linear projection), and then “writes” its result to the residual stream by adding a linear projection back in. Note that each attention layer consists of multiple heads which each act independently (adding their results to the residual stream). The residual stream is simply the sum of the output of the previous layers and the original embedding and should be thought of as a communication channel between layers.

Note that the residual stream has a very simply linear, additive structure -  each layer performs an arbitrary linear transformation to "read in" information from the residual stream at the start, 4 and performs another arbitrary linear transformation before adding to "write" its output back into the residual stream. One basic consequence is that the residual stream doesn't have a *privileged basis*; we could rotate it by rotating all the matrices interacting with it, without changing model behavior.

:::{hint}
The key insight for our purposes is that examining the residual stream of a token at some intermediary stage in the model should tell us how the model perceives the token at that stage of computation. For example, one could examine the *features* that the model believes describe a token. 
:::

# Features as the basic unit of neural networks
As stated above MI seeks to break up models into components, understand how these components work, and how they fit together. The first question we need to address therefore is: *what is the fundamental/basic unit of a neural network?* Intuitively, one might expect neurons to be this basic unit, and early research in mechanistic interpretability did indicate that this was true. For example, one of the first attempts in MI was @olah_zoom_2020, where the authors examined InceptionV1 (@inceptionv1) which was a CNN introduced by Google for image recognition. The authors find neurons that activate on specific features. For example, they find neurons that respond to curved lines and boundaries with a radius of around 60 pixels. 
```{figure} ./images/curves-detection.png
:label: curves-detection
:alt: Examples of images on which curvev detection neurons have high activation. 
:align: center

Examples of images on which curve detection neurons have high activation (@olah_zoom_2020).
```
There were other examples such as neurons that detected dog heads. These neurons are called *monosemantic* since they encode a single, well-defined concept. Often however, neurons appear to be *polysemantic*, i.e. a single neuron encodes multiple features. 

```{figure} ./images/polysemantic-neuron-inception.png
:label: polysemantic-car-cat
:alt: Examples of images on which a particular neuron has high activation. 
:align: center

Examples of images which a single neuron responds strongly to (an example of polysemanticity) (@olah_zoom_2020).
``` 
## The principle of superposition
It turns out that a large portion of neurons are polysemantic, even in simple models trained on data with few features. One explanation for this is the *principle of superposition*, which posits that when tasked with representing data with more independent "features" than it has neurons, it assigns each feature a linear combination of neurons. Intuitively, an LLM is attempting to encode trillions of "features" such as Python code, DNA, algebraic topology, sheaf theory, cellular biology etc with much fewer neurons. Essentially it is trying to encode a high dimensional vector space into a much lower dimensional one. @elhage2022superposition point to classical results that indicate that such a task is possible 



- **[Johnson-lindenstrauss Lemma](https://en.wikipedia.org/wiki/Johnson–Lindenstrauss_lemma)**: Suppose our model is attempting to represent data with $n$ underlying features but only has $k$ neurons at its disposal, such that $n<\exp (k)$. Faithfully approximating this data, amounts to finding low-distortion embeddings of $\mathbb{R}^n$ into $\mathbb{R}^k$. The lemma guarantees that a linear map $f$ exists such that given orthogonal vectors $u,v \in \mathbb{R}^n$ the projections are quasi-orthogonal (i.e. $\langle u,v\rangle < \epsilon$). In other words, it's possible to have $\exp(k)$ *quasi-orthogonal* vectors in $\mathbb{R}^k$.
- **Compressed sensing**: The result allows us to project a high-dimensional vector onto a lower dimensional space, but how do we recover the original vector? In general, it is impossible to reconstruct the original vector from a low-dimensional projection, but results from the field of compressed sensing tell us that this is possible if the original vectors were sparse. This is the case for feature vectors, since each concept (e.g. text relating to enriched categories) only occurs relatively rarely in the overall dataset. 

```{figure} ./images/superposition-hypothesis.png
:label: superposition-hypothesis
:alt: superposition-hypothesis 
:align: center

We can think of a neural network as approximating a larger sparse model (@elhage2022superposition).
```

In other words, we can think of a neural network as approximating some larger model by superposition. In the larger model (where we have as many neurons as features), neurons are monosemantic. The models we can build in practice use superposition to project these features onto polysemantic neurons. 

## What is a feature?

Having seen that neurons  

# Sparse Autoencoders 

# Feature Manifolds 