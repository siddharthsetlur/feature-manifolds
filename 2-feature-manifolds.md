# Geometry of Features Learned by Language Models 
Recent work has explored the idea that the internal representations of LLMs (i.e. the way internal activations are represented in terms of features) lie on higher dimensional manifolds called *feature manifolds*. Geometric properties of these manifolds might tell us something about the properties of feature representations of a model. 
```{figure} ./images/days-week-circ.png
:label: cyclic-features
:alt: cyclic-features
:align: center
Features corresponding to the days of the week, months of the year, and years of the 20th century in layer 7 of GPT2-small appear to lie on a circle. The representations are obtained using an SAE and examing reconstructions after ablating all other SAE features, e.g. the points for days of the week are obtained by examining the representation of weekdays after passing through an SAE where all features are set to $0$ except for the feature corresponding to weekdays (and a cluster that are closely related). Taken from @engels_not_2024. 
```
For instance @engels_not_2024,discover that representations of certain concepts lie on higher dimensional manifolds inclusing features corresponding to the days of the week as depicted in @cyclic-features. Mathematically, we need to update what we mean by features in order to discuss these multi-dimensional features. 
## Multidimensional Features
:::{prf:definition} Multidimensional features (@engels_not_2024)
:label: multid-feature
Let $\mathcal{T}$ denote the input space and $U\subset \mathcal{T}$ be a subset. A $d_f$-dimensional feature is a map $f: U\to \mathbb{R}^{d_f}$. We say that a $f$ is active on $U$.
:::
:::{prf:example} Days of the week
:label: weekdays-feature
Consider the feature 'days of the week'. In this case $\mathcal{T}$ is the space of all tokens in the corpus and $U$ is a subset containing all tokens that correspond to days of the week and related concepts like 'Monday', Tuesday or even abbreviations like 'Mon'. Looking at @cyclic-features, $f: U\to\mathbb{R}^2$ might be a map that sends $U$ to a circle in $\mathbb{R}^2$ with each day of the week getting particular polar coordinates along that circle. 
:::
A higher dimensional feature, might just be the sum of independent lower-dimensional features. We need a means of excluding such features. 

:::{prf:definition} Reducible features (@engels_not_2024)
:label: reducible-feature
A feature $f$ is reducible into features $a$ and $b$ if there exists an affine tranformation 
$$
f\mapsto Rf +c \equiv \begin{pmatrix}
a \\
b 
\end{pmatrix}
$$
where $R$ is a $d_f$ by $d_f$ orthonormal matrix and $c$ is a constant, such that the probability distribution $p(a,b)$ satisfies *one* of the following conditions:
1. Separability: We can express the joint distribution as a product of its marginals - $p(a,b) = p(a)p(b)$
2. Mixture: The joint is a sum of disjoint distributions, at least one of which is lower dimensional $p(a,b) = wp(a)\delta_b + (1-w)p(a,b)$
:::
An example of a separable distribution is geographical coordinates, which can be separated as latitude and longititude. A typical example of a mixture is a one-hot encoding which can be clearly expressed as a sum of lower dimensional features, where we get a Dirac measure on the points that are "hot". @engels_not_2024 operationalize both these criteria as the *separability index* and *$\epsilon$-mixture index* respectively. Rather than going into the details here, the reader is referred to the paper. Instead, we sketch how one can update the [superposition hypothesis](#linear-rep-hyp) to accomodate for multidimensional features and how the authors operationalize this using SAEs. 
:::{prf:definition} Multidimensional superposition hypothesis (@engels_not_2024)
:label: multid-feature
Let $j$ be a token, $x^j\in \mathbb{R}^d$ be the activation vector of an MLP layer of a transformer, and $\{f_i\}_i$ be a collection of *irreducible multidimensional features*. Then,
```{math}
:label: multid-rep-hyp
x^j \approx \sum_i V_i f_i(x^j)
```
where $V_i\in \mathbb{R}^{d\times d_{f_i}}$ are pairwise quasi-orthogonal and $f_i(x^j)=0$ for all $x_j$ that are not in the subset where $f_i$ is active. 
:::
## Discovering Multidimensional Features using SAEs
Can we use the machinery of SAEs that we built up to discover multidimensional features? Yes, by clustering the feature direction! Recall that features live in some higher dimensional spaqce and the model projects these down to a lower dimensional space (superposition hypothesis). Column $i$ of the decoder matrix $D$ of an SAE can be thought of as the projection of the basis vector of $f_i$ down to the lower dimensional space. 

Consider a complete weighted graph whose nodes are features (i.e. one node for each column of the decoder matrix) and the weight of the edge connecting nodes $i$ and $j$ is simply the cosine similarity between the corresponding columns of the decoder matrix (i.e. a high edge weight says that the two features encode similar concepts). Set a threshold $T$ and prune edges with weight below $T$. We now cluster the columns of $D$ by creating a cluster for each connected component of the graph. Note that the spaces spanned by each cluster are approximately $T$-orthogonal (since all of the vectors in the cluster have cosine-similarity above $T$). 

The claim is that if the SAE is large enough and $f$ is active enough (it activates on sufficiently large proportion of tokens), one of the clusters is likely to span $f$. Consider an irreducible $2$d feature and suppose $D$ includes just two columns $g_i$, $g_j$ spanning $f$, then these elements both must have nonzero activations (i.e. $g_i(t)>0$ and $g_j(t)>0$ for all $t$ in the subset of $\mathcal{T}$ where $f$ is active) to reconstruct $f$ (otherwise $f$ is a mixture). Because of the sparsity penalty in @sae-loss-func, this two-vector solution to reconstruct $f$ is disincentivized, so instead the dictionary is likely to learn many elements that span $f$ . These dictionary elements will then have a high cosine similarity, and so the edges between them will not be pruned away during the clustering process; hence, they will be in a cluster. 

While @weekdays-feature provide examples that vailidate this framework, it is unclear why we do not discover more irreducible multidimensional features. 

## Feature Manifolds

@modell_origins_2025 takes this formalism one step further and formally frames representations of features as manifolds. 