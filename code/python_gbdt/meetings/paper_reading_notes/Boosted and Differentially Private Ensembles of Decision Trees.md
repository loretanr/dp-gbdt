Boosted and Differentially Private Ensembles of Decision Trees

DP: essentially proceeds by randomizing parts of the whole process to reduce the output’s sensitivity to local changes in the input

- Fletcher is cited as source
- show that the sensitivity of the splitting criterion has the same dependence on the curvature: in few words, faster rate goes along with putting more noise to pick the split
- They use some kind of new loss, which tunes the boosting convergence vs privacy budget tradeof

Differential privacy (DP) essentially relies on randomized mechanisms to guarantee that neighbor inputs to an algorithm M should not change too much its distribution of outputs (Dwork et al., 2006). In our context, M is a learning algorithm and its input is a training sample (omitting additional inputs for simplicity) and two training samples S and S 0 are neighbors, noted S ≈ S 0 iff they differ by at most one example. The output of M is a classifier h. Definition 1 Fix ε ≥ 0. M gives ε-DP if p[M(S) = h] ≤ exp(ε) · p[M(S 0 ) = h], ∀S ≈ S 0 , ∀h, where the probabilities are taken over the coin flips of M.

A fundamental quantity that allows to finely calibrate noise to the privacy parameters relies on the sensitivity of a function f(.), which is just the maximal possible difference of f among two neighbor inputs

- Boosting start by formulating a Weak Learning Assumption (WLA) which gives a weak form of correlation with labels for the elementary block of a classifier. In the case of a DT, such a block is a split

Main points where you spend DP budget:
- Node splitting in trees
- Leaf predictions in trees

To split nodes they use the exponential mechanism to choose the splits (Friedman & Schuster 2010)

So far, all recorded approaches consider uniform budget spending (Fletcher & Islam, 2019) but such a strategy is clearly oblivious to the accuracy vs privacy dilemma. We now introduce a more sophisticated approach exploiting our result, allowing to bring strong probabilistic guarantees on boosting while being private. The intuition behind is simple: the "support" (total unnormalized weight) of a node is monotonic decreasing on any root-to-leaf path. Therefore, we should typically increase the budget spent in low-depth splits because (i) it impacts more examples and (ii) it increases the likelihood of picking the splits that meet the WLA in the exponential mechanism. 

We have performed 10-folds stratified CV experiments on 19 UCI domains, detailed in App., Section 17.3, ranging from m · n < 3 000 to m · n > 200 000

We have compared our approach, bdpeα, to two state of the art implementation of RFs based on Fletcher & Islam (2017) but replacing the smooth sensitivity by the global sensitivity (Definition 1)

The picture that seems to emerge is that objective calibration is the best technique for high privacy demand

Questions:
- In (1), what's `u`
- How do you define the weights for the leaves? E.g. in (6)