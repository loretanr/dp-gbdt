Privacy-Preserving Gradient Boosting Decision Trees

Sensitivity and privacy budget are two key design aspects for the effectiveness of differential private models. 

Loose sensitivity bounds lead to more noise to obtain a fixed pri- vacy level. Ineffective privacy budget allocations worsen the accuracy loss especially when the number of trees is large. 

Therefore, we propose a new GBDT training algorithm that achieves tighter sensitivity bounds and more effective noise allocations. Specifically, by investigating the property of gra- dient and the contribution of each tree in GBDTs, we propose to adaptively control the gradients of training data for each it- eration and leaf node clipping in order to tighten the sensitiv- ity bounds. Furthermore, we design a novel boosting frame- work to allocate the privacy budget between trees so that the accuracy loss can be further reduced. 

GBDT: The algorithm builds a number of decision trees one by one, where each tree tries to fit the residual of the previous trees. 

In short, a com- putation is differentially private if the probability of pro- ducing a given output does not depend much on whether a particular record is included in the input dataset. 

Privacy budget allocations: There have been some pre- vious studies on privacy budget allocations among different trees (Liu et al. 2018; Xiang et al. 2018; Zhao et al. 2018). We can basically divide them into two kinds. 1) The first kind is to allocate the budget equally to each tree using the sequential composition (Liu et al. 2018; Xiang et al. 2018). When the number of trees is large, the given budget allocated to each tree is very small. The scale of the noises can be pro- portional to the number of trees, which causes huge accuracy loss. 2) The second kind is to give disjoint inputs to different trees (Zhao et al. 2018). Then, each tree only needs to satisfy ε-differential privacy using the parallel composition. When the number of trees is large, since the inputs to the trees can- not be overlapped, the number of instances assigned to a tree can be quite small. As a result, the tree can be too weak to achieve meaningful learnt models. 

Based on a popular library called LightGBM  
The GBDT is an ensemble model which trains a number of decision trees in a sequential manner.

GBDT traverses all the feature values to find the split that maximizes the gain. If the current node does not meet the requirements of splitting (e.g., achieve the max depth or the gain is smaller than zero), it becomes a leaf node.

We design a novel two-level boosting framework to exploit both sequential composition and parallel composition. 
Inside an ensemble, a number of trees are trained using the disjoint subsets of data sampled from the dataset. Then, multiple such rounds are trained in a sequential manner. 

Also used for regression, with significant improvements compared to other stuff.


Questions:
- In (1), what’s the leaf weight?
- Below (2), how does this work for the root node?
- (8) Exponential mechanism?
- l4: Update gradients of all training instances on loss l.
- l13-l14: Star?