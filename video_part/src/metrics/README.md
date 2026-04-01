## Evaluation Metrics

This directory contains the metric code used by `src/evaluate.py`. In the current implementation, the main evaluation pipeline relies on MCC-based matching from `mcc.py` and nonlinear $R^2$ computation from `r2.py`.

To evaluate both component-wise and block-wise identifiability in CausalVerse, we adopt three metrics: **Mean Correlation Coefficient (MCC)**, **coefficient of determination** ($R^2$), and **over-complete MCC**.

Let $Z \in \mathbb{R}^D$ denote the ground-truth latent vector, and let $\hat{Z} \in \mathbb{R}^{\hat{D}}$ denote the estimated latent vector.

### Mean Correlation Coefficient (MCC)

The MCC evaluates component-wise identifiability by measuring how well each ground-truth latent dimension aligns with one estimated latent dimension.

In the current codebase, this role is implemented in `mcc.py`. The evaluation entry point can use either full representations or aggregated representations.

We first compute the Pearson correlation matrix $R$, where the elements are defined as:
$R_{i,j} = \text{corr}(Z_i, \hat{Z}_j)$. 

Then, we find an injective matching $\pi : \{1, \dots, D\} \to \{1, \dots, \hat{D}\}$ that maximizes the term $\sum_{i=1}^D |R_{i,\pi(i)}|$.

Finally, MCC is defined as:

$$
\text{MCC} = \frac{1}{D} \sum_{i=1}^D |R_{i,\pi(i)}|
$$

A higher MCC indicates better component-wise identifiability.

### Coefficient of Determination ($R^2$)

The $R^2$ metric evaluates block-wise identifiability by measuring how much variance in a ground-truth latent block $\mathbf{z}_b$ can be explained by the estimated block $\hat{\mathbf{z}}_b$.

In the current codebase, this role is implemented in `r2.py`, where the reported score is obtained through a regression model.

Formally,

$$
R^2 = 1 - \frac{\text{Var}(\mathbf{z}_b - f(\hat{\mathbf{z}}_b))}{\text{Var}(\mathbf{z}_b)}
$$

Here, $f$ is a regression function, either linear or non-linear, that best predicts $\mathbf{z}_b$ from $\hat{\mathbf{z}}_b$. An $R^2$ value of 1 indicates perfect block-wise identifiability.

### Over-complete MCC

In real-world applications, the true number of latent variables is usually unknown. Without explicit model selection, learned representations may become over-complete, containing both informative and redundant dimensions, i.e., $\hat{D} > D$.

To account for this, we refine the MCC metric by selecting the top $D$ estimated dimensions that best match the ground-truth variables. Standard MCC is then computed on this selected subset.