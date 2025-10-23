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

The main takeaway here is that neurons exhibit polysemanticity and cannot be used as the fundamental unit of a neural network in the context of MI. 

## What is a feature?

Thus far, we've been talking about "features" without defining precisely what we mean by them. Intuitively these should be thought of as the intrinsic characteristics of the dataset the model is attempting to learn and that the model uses to represent the data. For example, InceptionV1 might have features such as curves, car doors, and dog faces among others that it uses to represent the dataset of images. An LLM might use features like category theory, politics, and Scotland among others as features that it believes represent the dataset of text. There are multiple approaches that one could take to define features such as (@elhage2022superposition): 

(1) *features as arbitrary functions of the input*. 

(2) *features as interpretable properties* i.e. things like curves or dog faces that are human interpretable. 

(3) *Features are neurons in sufficiently large models*, i.e. the neurons in the large model that the observed model is attempting to approximate using superposition. Intituively, each neuron in the upper model depicted in @superposition-hypothesis forms a feature of the observed model downstairs. 

Although (3) might seem circular, this interpretation yields features the model might actually learn (assuming the superposition hypothesis). This is not true of (1). Using (3) gives us a shot of extracting features the model cares about, while (2) restricts to concepts that we understand. Finally, as we will see below (3) is easy to operationalize. 

Formally, this definition uses the superposition hypothesis to state the following. Consider a single block transformer model, i.e. a transformer with a single attention layer and a single MLP layer. Let $m = d_{\text{MLP}}$ denote the number of neurons in the MLP layer. Then the *activation space* of the model is $\mathbb{R}^m$ (each of the $m$ neurons can take on a single real value). Each token $j$ yields a vector $x^j\in \mathbb{R}^m$ when fed through the MLP layer. The superposition hypothesis states that this activation vector is an approximation as below
```{math}
:label: linear-rep-hyp
x^j \approx b + \sum_i f_i(x^j)d_i
```
where $f_i(x^j)\in \mathbb{R}$ is the *activation* of feature $i$ and $d_i\in\mathbb{R}^m $ is a unit vector in activation space called the *direction* of feature $i$. Note that there are $n>>m$ pairwise quasi-orthogonal feature directions $d_1,\dots, d_n$. The fact that features are *sparse* means that for a given $x^j$, $f_i(x^j) = 0$ for most $i$. 
:::{note}
@linear-rep-hyp is only stating that the map from features to activation vectors is linear **not** the map from an input token to features (this is almost always non-linear). 
::: 
## Using features to determine the mechanics of a model
Suppose that we can express MLP activations as a linear combination of feature diections as above. @bricken2023monosemanticity put forward desirable characteristics of such a decomposition: 

1. *Interpretatable conditions for feature activation*: We can find a collection of tokens $j$ that cause feature $i$ to activate, i.e. $f_i(x^j)$ is high and we can describe this collection of tokens.

2. *Interpretable downstream effects*: Tuning $f_i$ (forcing it to be high or low) results in predictable and interpretable effects on the token as it passes through to subsequent layers. 

3. *The feature faithfully approximates the function of the MLP layer*: The right hand side of [](#linear-rep-hyp) approximates the left hand side well (as measured by a loss function). 

A feature decomposition satisfying these criteria would allow us to (@bricken2023monosemanticity):

1. Determine the contribution of a feature to the layer’s output, and the next layer’s activations, on a specific example.
2. Monitor the network for the activation of a specific feature (see e.g. speculation about safety-relevant features).
3. Change the behavior of the network in predictable ways by changing the values of some features. In multilayer models this could look like predictably influencing one layer by changing the feature activations in an earlier layer.
4. Demonstrate that the network has learned certain properties of the data.
5. Demonstrate that the network is using a given property of the data in producing its output on a specific example.
6. Design inputs meant to activate a given feature and elicit certain outputs.
# Sparse Autoencoders 
```{figure} ./images/sae_big_picture.png
:label: sae-architecture
:alt: SAE 
:align: center

Decomposing MLP activations into features using a sparse, overcomplete autoencoder (@bricken2023monosemanticity).
```
We learn feature directions and activations by training a sparse autoencoder (SAE). This is a simple neural network with two layers. The first layer is an encoder that takes in an activation vector $x\in \mathbb{R}^m$ and maps it to a higher-dimensional (say $N$) latent space using a linear transformation followed by a ReLU activation function. Denote by $W^\text{enc}$ the learned encoder weights and by $b^\text{enc}$ the biases. The second layer maps the latent space back down to $m$-dimensions using a linear transformation. Denote by $W^\text{dec}$ the learned decoder weights and by $b^\text{dec}$ the biases. Applying the encoder and then the decoder to an activation vector $x$ yields
```{math}
:label: sae-linear-hyp
\hat{x} = b^\text{dec} + \sum_{i=1}^N f_i(x)W_{\dot, i}^\text{dec}
```
where 
$$
f_i(x) = ReLU(W^\text{enc}_{i,.} x + b_i^\text{enc})
$$
If $x\approx \hat{x}$ and the features are sparse, we have an implementation of @linear-rep-hyp. The loss function we train the SAE on accomplishes exactly this 
```{math}
:label: sae-loss-func
\mathcal{L}(x) &= \|x-\hat{x}\|_2^2 + \lambda\|f_i(x)\|_0, &x\in \mathbb{R}^m
```
The first term ensures that the activation vectors are reconstructed faithfully while the second term ensures sparsity. 
## Experimental results
in @bricken2023monosemanticity, the authors train SAEs with increasingly higher latent dimensions on a simple single layer transformer. The results they obtain indicate that SAEs are capable of extracting interpretable features that 

1. Activate with high specificity to a certain hypothesized context (by context we mean a description of the tokens that activate it like DNA features or Arabic script): whenever $f_i(x^j)$ is high the token $x^j$ can be described by the hypothesized context.
2. Activate with high sensitivity to a certain hypothesized context: whenever a token $x^j$ is described by the context, $f_i(x^j)$ is high. 
3. Cause appropriate downstream behavior: Tuning $f_i$ (e.g. by setting $f_i(x^j)$ to always be high regardless of the input token) and replacing $x^j$ by with $\hat{x}^j$ results in outputs that reflect $f_i$. Note that $\hat{x}^j$ is obtained in the following way: for a given $j$, run the model as is until we obtain the output of the MLP layer $x^j$. Replace $x^j$ in the residual stream by running it through the SAE *with the tuned version of $f_i$* and obtaining the reconstruction $\hat{x}^j$ and let the following layers of the transformer proceed as is. A particularly impressive example of this is Golden Gate Claude (@templeton2024scaling), where the authors are able to scale up the SAE machinery developed in @bricken2023monosemanticity to a full-blown version of Claude. They are able to pin point a feature that corresponds to the Golden Gate bridge and upon tuning that feature up (i.e. making $f_\text{Golden}(x^j)$ high for all $j$), Claude outputs text related to the Golden Gate bridge regardless of the input. 
4. Do not correspond to any single neuron.
```{figure} ./images/feature-clustering.png
:label: feature-clustering
:alt: feature-clustering 
:align: center

2D UMAP projection of the columns of the decoder matrix (feature directions) of sparse autoencoders with  varying latent space dimensions (@bricken2023monosemanticity).
```
I strongly recommend going through @bricken2023monosemanticity, as the visualizations of the features they find are very cool! The interactive dashboard lets you explore features, the tokens that activate them, the effects of ablating (tuning) features, and descriptions of features among other things. One particularly striking observation is that feature directions (the columns of the decoder matrix of the sparse autoencoder) appear to cluster as seen in @feature-clustering. The image shows feature directions of sparse autoencoders with different latent dimensions (the gray is a sparse autoencoder with $512$ latent dimensions while the light green points are feature directions of an encoder with $16,384$ latent dimensions). Similar concepts such as those corresponding to Arabic script or base64 appear to cluster together. To me this is rather counterintuitive as I would have expected a more uniform distribution in order to minimize interference. 
# Feature Manifolds 
Recent work has explored the idea that the internal representations of LLMs (i.e. the way internal activations are represented in terms of features) lie on higher dimensional manifolds called *feature manifolds*. Geometric properties of these manifolds might tell us something about the properties of feature representations of a model. For instance @engels_not_2024, the authors find features that lie on circles. 

, @modell_origins_2025