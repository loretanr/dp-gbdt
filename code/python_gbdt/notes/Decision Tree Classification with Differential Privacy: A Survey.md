Decision Tree Classification with Differential Privacy: A Survey

For each component, whether it is leaf nodes, non- leaf nodes, termination criteria, pruning, or multiple trees, the decision to query the data comes with a privacy cost. These privacy costs add up, and ultimately dictate the utility of the decision tree. Great care must therefore be taken when deciding which components are data-based, and which components can be navigated without querying the data. 

The main factors that need to be considered when designing a differentially private machine learning algorithm are: 

* How large of a privacy budget β the data curator is providing the user with. The total budget dictates the overall constraints put on the data mining algorithm. We discuss the range of sizes β can have in real-world scenarios in Section 4.1.  
* The number of times the data needs to be queried. The more queries that the algorithm needs, the more the total privacy budget β needs to be divided up to pay for them all. This reduces how large ǫ can be for each query, and the smaller ǫ is, the noisier the outputs of the queries will be. Limiting the number of required queries is discussed in Section 3.2, Section 3.4 and Section 3.8.  
* The sensitivity ∆ of the queries, which is influenced by the range and distribution (i.e. shape) of the data [Hay et al. 2016]. Sometimes a query that performs well in a non-private setting becomes unusable due to how sensitive it is to individual records, leading to overwhelming noise being added to the output. Instead, traditionally sub- optimal queries in non-private settings can be preferable in the private setting if they have low sensitivity. We explore this phenomena in Section 3.2.  
* The size (or scale) n of dataset x also plays an important role. The amount of noise that must be added to enforce differential privacy is independent of n, so the larger n is, the smaller the relative amount of noise becomes.  
The main factors to consider when designing a decision tree algorithm are: 

* What kinds of dataset properties the algorithm is catered towards, such as if it can handle discrete attributes, continuous attributes, or both. The dimensionality of the dataset also plays a large role, mostly in terms of how the number of attributes m affects tree depth, discussed in Section 3.4. The number of records n also plays a role in defining termination criteria, but predominately due to the requirements of differential privacy, as mentioned in Section 2.1.1. The overall role of the data is discussed in Section 3.1.  
* What splitting function to use (including random selection). We explore the effect of this decision in Section 3.2.  
* What termination criteria to use. We discuss four different types of terminations, and several examples of each, in Section 3.4.  
* Whether to include a pruning step, and what pruning would be most appropriate if so. We discuss pruning in Section 3.5.  
* Whether to build multiple trees and use each as part of a larger ensemble, and how many trees to build if so. We explore these ideas in Section 3.6. 

At its most basic level a decision tree algorithm is deciding which attribute to split each node with (e.g. with a splitting function; see Equation 4), and this decision is dictated by the data in the node. Once the tree has finished being built, the leaf nodes can output some information about the class counts, which is also dictated by the data in the nodes (see Section 2.2). Since these decisions and outputs are directly based on the data, differential privacy states that releasing the information can be a breach of privacy. These potential breaches are what a differentially private decision tree algorithm aims to prevent.  
Privacy budget spent:
- When making the tree if using a greedy approach (i.e. to find best split)
- When querying the leaf nodes for majority count

Gini Index also achieved competitive results, having much lower sensitivity than Information Gain. Max score was found to perform the best.

Essentially, smooth sensitivity finds the maximum amount that a function f ’s output can change when k records are added or removed, given the spe- cific input data x. 

In fact, using smooth sensitivity would have allowed all four splitting func- tions (including Gini Index) to add less noise than was added by Friedman and Schus- ter, provided that the functions are applied to the data using the Exponential mecha- nism. This would be an interesting direction to explore in future research; using the smooth sensitivity of each of the splitting functions instead of the global sensitivity. 

(Continuous attributes) Friedman and Schuster [2010] also conducted a prelimi- nary exploration into using splitting functions on continuous attributes. Using an attribute value from among the records in a node is a breach of privacy, so another solution is needed. Friedman and Schuster propose using the Exponential mechanism to select the best range of attribute values for splitting, where all the values in the range output the same score from the splitting function. A data-independent value can then be uniformly randomly selected from the chosen range. 

TL;DR: you really want to avoid continuous attributes

Q: What kind of prediction accuracy are they looking for? This will impact privacy budget

Despite the average random node being less discriminatory than the average greedy node, the overall prediction accuracy of an ensemble of random trees has been shown to be very similar to the accuracy of an ensemble of greedy trees [Geurts et al. 2006]. 

For smooth sensitivity: Similarly to how Friedman and Schuster [2010] improved upon Blum et al. [2005]’s original work by using the Exponential mechanism instead of the Laplace mechanism in the non-leaf nodes (see Section 3.2), 
Fletcher and Islam [2017] do the same with the leaf nodes, outputting the majority class label with the Exponential mechanism. They observed that if the intent of the leaf nodes is only to output a majority label, then the actual counts of the labels are largely irrelevant. To output the majority (i.e., most frequent) label in a leaf node with the Exponential mechanism, a scoring function (see Theorem2.5) is required; Fletcher and Islam [2017] proposed a piecewise linear function to achieve this.

Unfortunately in the private scenario, each tree added to a forest comes with a cost; if each tree is built using the full dataset, the trees are not disjoint and cannot be queried in parallel. If the records are divided into disjoint subsets for each tree, the leaf nodes will have much lower support, and therefore less reli- able majority class labels. The smaller a class count is, the larger the relative ef- fect of adding Lap(1/ǫ) to it is. 

Takeaway from table:
- smooth sensitivity will be good only if we have enough data (or a high privacy budget)
- Random trees are better because you don’t spend privacy budget when building the tree. Other random construction with high accuracy that might use all data for each tree is https://cspri.seas.gwu.edu/sites/g/files/zaxdzs1446/f/downloads/semi-supervised_0.pdf. Uses only 5 trees for small datasets. Seem to be around 80% accuracy