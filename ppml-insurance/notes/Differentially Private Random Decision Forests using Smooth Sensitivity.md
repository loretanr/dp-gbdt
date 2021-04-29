Differentially Private Random Decision Forests using Smooth Sensitivity

- Minimizes the number of queries required and the sensitivity of these queries, by avoiding querying the private data except to find the majority class label in the leaf nodes: Rather than using a count query to return the class counts like the current state-of-the-art, we use the Exponential Mechanism to only output the class label itself. This drastically reduces the sensitivity of the query – often by several orders of magnitude – which in turn reduces the amount of noise that must be added to preserve privacy. 

In this paper, we focus on a system that takes large amounts of data as input, and outputs a classification model

The main factors that dictate the impact of differential privacy on a classifier are: (a) the size of the privacy budget; (b) the number of queries that are required to build the classifier; and (c) the sensitivity of those queries to small changes in the data.

Given a reasonably large dataset, we achieve high accuracy with a budget as small as epsilon = 0.1.

Assumption: features and their respective values are publicly available

In this paper, we propose a differentially-private decision forest algorithm that makes very efficient use of the privacy budget ǫ to output a classifier with high prediction accuracy. We achieve this by proposing a query in Section 4.1 that outputs the most frequent label in some subset xi of the data with high probability, and using this query in each leaf node of all the trees in a forest. We prove that this query has low sensitivity, making it reliable even without a large privacy budget. This proof is generalized to the non-binary case, and tested on several datasets that have more than two class labels. 

Smooth sensitivity allows for much less noise to be added while still achieving differential privacy, by analyzing the actual dataset x instead of just assuming the worst-case scenario.

We also demonstrate in Section 5.1 how the larger the size n of a dataset, the smaller epsilon can be without injecting too much noise into the resulting trees

We generate n = 30000 records for each dataset, with different numbers of continuous features m, and use a balanced binary class feature (for synthetic data)

Several differentially-private decision tree algorithms have been proposed in recent years (Friedman & Schuster, 2010; Jagannathan et al., 2012; Fletcher & Islam, 2015a,b; Rana et al., 2016). Of these, two of them took a similar approach to our paper and used random decision trees to construct a decision forest (Jagannathan et al., 2012; Fletcher & Islam, 2015b). The other three make more traditional decision trees, using greedy heuristics in each node to construct non-random trees (Friedman & Schuster, 2010; Fletcher & Islam, 2015a; Rana et al., 2016). All of them achieve differential privacy by adding Laplace noise (Dwork et al., 2006) to the counts of the labels in the nodes.1 While this approach makes good use of parallel composition (Definition 3), it scales poorly with multiple labels, since noise needs to be independently added to each label count. 

One of the disadvantages of using a splitting criteria in each node is that the user must query the data to do so. This is an expenditure of the privacy budget that random decision trees avoid entirely.

In fact, the amount of Laplace noise that needs to be added to a frequency query (such as label counts) cannot be reduced by using smooth sensitivity instead of global sensitivity; adding or removing one record can always change a count by 1, even when considering a specific dataset x. What this means is that submitting a frequency query to the dataset is an inherently expensive query, and should be avoided in favor of less expensive queries if possible. We demonstrate that this is indeed possible by devising a less sensitivity query that outputs a similar answer, resulting in less noise overall.

Main contribution: The novel parts of our algorithm are the following: how we output the majority (i.e., most frequent) label of each leaf node (Section 4.1); our efficient utilization of the privacy budget (Section 4.2); our proposed tree depth, extending the non-private work of Fan et al. (2003) to handle numerical features (Section 4.3); and the number of trees we build (Section 4.4).

Differential privacy is achieved by our algorithm only outputting the following: the structure of the trees (which does not use the data); and the most frequent label in each leaf node, which is done using the Exponential Mechanism. The user can then use the outputted labels from the leaf nodes in whatever way they wish; differential privacy is immune from post-processing, and differentially-private outputs can never incur additional privacy costs after the fact (Dwork & Roth, 2013). More specifically, the user is free to use majority voting; they can predict the label of a new record using the most common predicted label from all the trees in the ensemble.

Definition 4 describes how the Exponential Mechanism is capable of returning the discrete output of the most frequent class label in a leaf x

When building multiple trees (say τ trees), there are two fundamentally different ways we can use our privacy budget ǫ. One is to use composition (Definition 2), where all the records in x (n = |x|) are used in every tree and we divide ǫ evenly amongst the trees, ǫ ′ = ǫ/τ. The other way is to use parallel composition (Definition 3), dividing x into disjoint subsets2 evenly amongst the trees (n = |x|/τ) and using the entire ǫ budget in each tree.

good-scoring labels have a higher chance of being outputted by the Exponential Mechanism when disjoint subsets of x are used

Due to the above factors, we propose using parallel composition, and using disjoint data in each tree with the full privacy budget

In our experiments, using both synthetic and real-world data, over 85% of the leaf nodes in any non-trivial tree are usually empty. however, these empty leaf nodes are unlikely to be visited by future records. Any records that do finish at an empty leaf node will be predicted to have a class label that is randomly chosen with uniform probability

Uses quite a lot of records (see table 3)

In other words, a sample of 3,000,000 records with ǫ = 0.1 can achieve comparable results to a sample of 30,000 records when ǫ = 1.0

For example, a large public project was able to use a privacy budget of ǫ = 8.6

How much privacy do we want???

very good results with epsilon close to 2

Fig. 9 demonstrates that with a budget of ǫ = 1, 30,000 records is all that is needed to make a classifier with over 85% predictive accuracy.