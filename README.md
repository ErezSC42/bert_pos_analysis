# Linguistic Information in Deep Language Models
The purpose of this project is to explore the linguistic knowledge embedded in Neural Networks, specifically transformres. in this context, the basic linguistic unit is a word and it's part of speech (POS) label.

## Data
we have used the Universal Dependencies English POS dataset.



## BERT embedding distribution
[Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426) (UMAP) is a manifold based dimension reduction algorithm, useful for visualization of high dimensional data.
I used  it to to visualize the intermediate word representations (as mentioned earlier) in the vanilla BERT model. UMAP has both a supervised and an unsupervised implementations (the labels can be used to improve the dimenstion reduction process). In the following example, the unsupervised implementation was used as our goal is to observe the POS distribution in the embedding space 
### 12th layer 
![umap visualization](images/BERT%20Contextual%20Embedding%20Visualization%20of%20the%2012th%20Layer.png) ![umap visualization](images/BERT%20Contextual%20Embedding%20Visualization%20of%20the%204th%20Layer.png)

### 4th layer
