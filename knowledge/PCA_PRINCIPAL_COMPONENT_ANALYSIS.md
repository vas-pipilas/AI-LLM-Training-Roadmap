# PCA — Principal Component Analysis: detailed theory and lab walkthrough

Primary source: user-supplied "PCA - An example on Exploratory Data Analysis" notebook, 2026-10-31. These notes distinguish notebook results from external theory and corrections. No graded code was supplied or independently re-executed. Follow GUIDED_CODING_GOLDEN_RULE.md.

## 1. Mental model

PCA finds a new coordinate system that captures the directions where centered data varies the most. Imagine dots on paper forming a long narrow cloud: rotate a ruler to follow the longest spread, then record each dot's projection on the ruler. That ruler is PC1. PC2 is perpendicular to PC1 and captures the largest remaining variance. For many features, PCA gives orthogonal weighted combinations of original features, not merely a selection of original columns. PCA is unsupervised and linear; it does not itself classify or cluster data.

**Explained variance** means fraction of the dataset's total variation preserved in the selected directions. It does NOT guarantee preservation of prediction accuracy, semantic meaning, rare events, or other notions of "information."

## 2. Core mathematics and shapes

For m observations and n features, X has shape (m,n). Center each feature: mu = mean(X,axis=0), and Xc = X - mu. Learn principal directions via eigenvectors of covariance or singular value decomposition (SVD). Retain k directions in V_k, shape (n,k). Compressed data Z = (X-mu) @ V_k, shape (m,k). Approximate reconstruction Xhat = Z @ V_k.T + mu, shape (m,n).

The first principal component maximizes projected variance; each later component is orthogonal to earlier components and maximizes remaining variance. Sklearn PCA implements centering, but **not scaling to unit variance**. For features with different units or magnitudes, consider training-fitted StandardScaler before PCA. The source says sklearn handles scaling, which is imprecise: it handles centering only.

Explained variance ratio of component q = variance along q / total feature variance. The sum across retained components is the fraction of variance retained. PC directions have arbitrary sign; sign flips do not change variance explained.

## 3. Six-point Andrew Ng lecture example

~~~python
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[99,-1],[98,-1],[97,-2],[101,1],[102,1],[103,2]])
pca_2 = PCA(n_components=2)
pca_2.fit(X)
print(pca_2.explained_variance_ratio_)   # [0.99244289 0.00755711]
X_trans_2 = pca_2.transform(X)           # (6, 2)

pca_1 = PCA(n_components=1)
X_trans_1 = pca_1.fit_transform(X)       # (6, 1)
X_reduced_1 = pca_1.inverse_transform(X_trans_1) # (6, 2)
~~~

PC1 accounts for 99.244289% variance; PC2 for 0.755711%. A one-dimensional projection preserves nearly all the variance, but loses the second-direction residual. Inverse-transforming with both PCs nearly exactly reconstructs the six points (floating-point precision). Inverse-transforming from one component reconstructs points on one line. In the notebook, the first coordinate from pca_2.transform matches the single-component transformed coordinate.

fit learns mean and principal axes; transform applies the previously learned projection; fit_transform combines these; inverse_transform reconstructs in original coordinates, approximately when components were discarded. Do not refit the PCA object on new observations.

Plotting: plt.plot(X[:,0], X[:,1], 'ro') plots first and second original columns as red circular markers. A transformed PCA coordinate is different from an original feature coordinate.

## 4. The projection widget and the 3D example

The notebook creates 10 2D sample points and calls plot_widget() from pca_utils. Rotating the visualization line shows how projected point positions change; some lines collapse different points together, while the PCA-optimal direction retains the greatest projected variation. pca_utils is course code and its internals are not contained in the pasted notebook.

Then random_point_circle(n=150) and plot_3d_2d_graphs(X) illustrate 3D points approximately contained in a 2D subspace. Two PCs can represent an approximately planar cloud with little variance loss. Exact helper details are not established by the supplied text.

## 5. 500 observations and 1000 features — exploratory analysis

The source loads toy_dataset.csv into a pandas DataFrame with 500 samples and 1000 features. It displays randomly selected original feature-pair scatter plots, then computes df.corr() and screens for large pairwise correlations, reporting maximum magnitude around 0.63. Pairwise scatter/correlation may fail to expose global multivariate structure; 100 plotted pairs are not exhaustive.

~~~python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('toy_dataset.csv')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)  # 500 rows, 2 coordinates per row
df_pca = pd.DataFrame(X_pca, columns=['principal_component_1','principal_component_2'])
plt.scatter(df_pca['principal_component_1'], df_pca['principal_component_2'])
print(sum(pca.explained_variance_ratio_))  # 0.14572843555106266
~~~

The 2D PCA plot in the notebook visually reveals groups despite only **14.5728% explained variance**. Visible groups are exploratory evidence, not proof of a specific number of underlying clusters or preservation of predictive meaning.

~~~python
pca_3 = PCA(n_components=3).fit(df)
X_t = pca_3.transform(df)   # shape (500, 3)
print(sum(pca_3.explained_variance_ratio_)) # 0.20806257816093243
~~~

**Notebook numerical correction:** 0.208062578 = **20.806%**, not 19% as the notebook prose states. Its claim of 10 visible clusters is a visual impression, not a validated cluster count.

## 6. Python/scikit-learn coding reference

| Expression | Meaning |
|---|---|
| X.shape | Number of observations and features |
| X[:,0] | All rows, first feature |
| PCA(n_components=2) | Keep two new principal directions |
| pca.fit(X) | Learn training mean and component directions |
| pca.transform(X) | Project onto learned directions |
| pca.fit_transform(X) | Fit then transform same data |
| pca.inverse_transform(Z) | Approximate original coordinates |
| pca.explained_variance_ratio_ | Fraction of total variance captured by each PC |
| pca.components_ | Learned axes: shape (k, n_features) |
| pca.mean_ | Training feature mean |
| np.cumsum(ratios) | Cumulative explained variance |
| pd.DataFrame(..., columns=...) | Label new PCA coordinates |
| df.corr() | Original columns' pairwise correlation |
| plt.scatter(x,y) | Draw point cloud |

Independent runnable shape/transform example (not graded work):
~~~python
import numpy as np
from sklearn.decomposition import PCA
train = np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
reducer = PCA(n_components=1).fit(train)
compressed = reducer.transform(train)
approx = reducer.inverse_transform(compressed)
new = reducer.transform(np.array([[4.,4.]]))
print(train.shape, compressed.shape, approx.shape, new.shape)
print(reducer.explained_variance_ratio_)  # [1.]
~~~

This example lies on a line: one component retains essentially all its variance. PCA is a deterministic linear projection once fitted.

## 7. Choosing number of components and production safeguards

For 2D/3D visualization use 2 or 3 PCs, understanding that low retained variance can hide structure. For compression choose k using cumulative explained variance (e.g. np.cumsum(pca.explained_variance_ratio_)), then verify actual task metrics such as accuracy or retrieval precision. The number of nonzero components is bounded by min(n_samples-1, n_features) after centering.

For predictive workflows: split train/test first; fit scaling and PCA on training data only; transform test and future observations using saved preprocessing. PCA is scale-sensitive, sensitive to outliers and linear. Large variance is not always predictive; compare to validation performance. PCA is not K-means: it rotates/reduces coordinates; clustering methods assign groups.

## 8. ML and LLM engineering relevance

PCA is useful for inspecting high-dimensional model features or embeddings, generating 2D/3D plots, finding dominant variation, exploring redundancy and, when validated, compressing stored vectors for retrieval. But **PCA is not a mandatory/default step** in training transformers or building RAG pipelines. A 2D PCA plot often distorts original embedding cosine distances and neighbor rankings. Retaining high variance does not automatically retain semantic or predictive information. For nonlinear visualization alternatives include UMAP/t-SNE with their own distortions and assumptions.

Compare your content-based recommender's learned 32-number movie embeddings: PCA could visualize those 32-D representations in 2D, but that visualization is only an exploratory projection. Keep using full trained vectors for actual similarity unless compressed retrieval quality has been tested.

## 9. Notebook corrections and traps

- "sklearn PCA automatically scales" is false: PCA centers, not variance-scales.
- 0.208062578 is 20.81%, not 19%.
- Explained variance is not percentage of useful information, semantic retention, or model accuracy.
- Apparent visual clusters are not independently validated labels.
- Components are linear combinations of original variables, not selected original feature columns.
- New observations use fitted transform; never refit to each new point.
- Keep source notebook code unchanged for grading; treat production safeguards as separate knowledge.

Related notes: PYTHON_ML_CODING_CHEAT_SHEET.md, NUMPY_FOR_ML.md, UNSUPERVISED_LEARNING_AND_KMEANS.md, GUIDED_CODING_GOLDEN_RULE.md.
