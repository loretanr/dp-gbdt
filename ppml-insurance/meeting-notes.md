# Meeting notes

## 2020-12-10

Next steps:
- Understand why 3-trees does not work on the synthetic dataset
- Understand in which situation our compositions of 3-trees is suboptimal
- How can we rescale the gains in a differentially private manner?

## 2020-11-19

- Introduction
- Background
  - DPGBDT
- Related Work
  - Pros and cons of current approach
- Section 4
  - Better stuff
- Motivation / Problem statement
  - Threat model in there
	- Privacy attack (adversary is the insurance company)
    - Security attack (customer tries to corrupt the system)
    - Enclave attack (again insurance company is the adversary)
 - Delete section 6
- Future work

## 2020-10-05

- Change std from 50% to ... so that baseline is around 10% error
- Increase complexity of features
  - Antivirus, firewall, monitoring, IDS/IPS...
  - Blue teaming, red teaming, IOCs, ....
  - Generate Bayesian network graph

## 2020-10-27

- Make plots on same dataset
- Change y-axis to percentage wise (mean)
  - Log absolute error stuff (predicted/real)
- Cannot add probs
  - Apply Bayes theorem
  - Conditional probabilities - Bayesian networks
    - Assume a conditional independence of antivirus, training and patching: we do not have any data on how they are 
      dependent; so, we model them as conditionally independent for the sake of having a maximally data-based synthetic 
      data generation process.
- log normal distribution
- Use cost number to create distribution (stereotype) and then draw random number from that
- std 10%
- 2^12 = 4096
- Add chaos with more features (e.g. 90% compliant with PCI and all)

## 2020-10-22

- Comments on Zurich's slides:
  - Translating privacy budget to business people
  - Model stays in enclave or nah?
    - outside is easier to evaluate
    - inside is better privacy wise
    - What comes in what comes out
    - max(privacy_customer) vs. accuracy
  - Next prez: dumb some parts down
- PII separation
  - Number of records per kind (multiply that by average price of each PII?)
- How much does it cost them if they are out of business (same for ransomware)
- How did Zurich come up with loss numbers?
- Enisa
- Find more features to add to synthetic dataset

## 2020-10-15

- Next steps:
  - Synthetic data generation:
    - What about features that are correlated? Identify such features
  - Zurich slides:
    - Address comments

## 2020-10-08

- Next steps:
  - Work on synthetic data
  - Work on Zurich presentation

## 2020-10-01

- Next steps:
  - Model improvement: less splitting
    - Cap maximum number of leaf nodes
    - Min. number of samples required in node before splitting
  - Evaluate model on smaller number of samples: 100
  - Brainstorm ways to reduce splitting

## 2020-09-22

- Plots with budget from 0. to 2
- Keep our own implementation
- Budget allocation for gain:
  - Counting queries
  - Expo. mechanism assumptions?
- Next steps:
  - Take a look at papers citing the authors
  - Evaluate other baselines
  - Evaluate other dataset (CANCER, ADULT)
  - https://scholar.google.de/scholar?cites=2769682308598133424&as_sdt=2005&sciodt=0,5&hl=de
  
## 2020-09-04

- Reference Dataset: https://www.kaggle.com/lucasgreenwell/depression-anxiety-stress-scales-responses
- No seeding for laplace (author's implementation seed it...)
- Take Gradient Boosted trees (or best of the non private method, for trees) for reference implementation
- Pick the 20 questions that separates best the dataset for the reference implementation

## 2020-08-28

- Added value if we can release which questions are the most relevant
  - This should be fairly easy to release, however we need to be careful with the privacy budget
- No risk score for training data, rather a severity rating and a loss
- probability * severity = risk
- Model outputting a severity distribution could prove useful
- 1 dimensional value outputs for now is fine
- End goal is a regression model
- Omitting questions when we work with all questionnaires so that we have some sort of baseline across questionnaires
- More detailed timeline for milestones as we go
  - gtheo@ TODO
- Project worker NDA to be signed
  - gtheo@ TODO
- Output a severity distribution: It would be useful for an insurance engineer to obtain as an output a severity 
distribution for each customer: "What is the confidence of the model for which risk severity estimation?"
- Future research: As the insurance market has a wide variety of questionnaires with different questions, it would be 
useful to learn an abstract representation for these questions and to learn how to categorize different questions. 
Then, we could train a regression model on this code space and could also capture unseen questions that cover various 
aspects of questions that we have already seen.
- Next meeting Tuesday 22nd 1pm

## 2020-08-18

- Tightness: you can the best amount of privacy with respect to the noise that you add
- Exponential mechanism: does better than randomly choosing a feature for the split
- Let's go with https://www.aaai.org/ojs/index.php/AAAI/article/view/5422
  - Version without splitting the data, then version with splitting the data
  - Talk to Philip for outputting of the Gi 
  - Reach out to authors for implementation
- Adapt current Random Forest to use exponential mechanism for node splitting based on current Information Gain 
computation.
  - Then add noise to leaf 
  - Then implement Algorithm 2

### Next step

- Reach out to paper authors
- Reach out to Philip so he can join next meeting (how important is it for you to know the relevance / expressiveness 
of each attribute). Weed out questionnaires.

## 2020-08-11

### Meeting

- Need to do some paperwork
- Need to sign project worker agreement (NDA like stuff)
- Decide on some process to keep Zurich's data confidential
- Boosted trees:
  - Use gradient descent to find the next tree in the ensemble of trees
- Privacy budget:
  - Ideally not larger than epsilon = 1
- Prediction accuracy:
  - Aiming at 80%
- Next steps:
  - Paperwork
  - https://www.aaai.org/ojs/index.php/AAAI/article/view/5422
  - https://arxiv.org/abs/2001.09384
  - Find something for random structure tree to find what's the best
  - Keep track of papers that we read

### Pre-meeting

- Added reading notes to `notes` folder. 
- Main takeways:
  - Smooth sensitivity performs great but trees are built upon disjoint subsets of the original dataset, making it useless if we only have little data available.
    - Small dataset = too small subsets = majority class labels per leaf node more sensitive to Laplacian noise.
    - If we can enrich our dataset with publicly available data this could solve the problem.
    - Disjoint datasets are preferred as the privacy budget per tree needs not to be added (parallel composition theorem) 
  - Survey shows that forests built using random trees perform similar to - if not better - forests built using greedy approaches
    - Random trees also allow us to save privacy budget since we do not need to query the dataset for node splitting
    - Computationally wise, better since building trees is faster (no need to find best split)
  - In the boosted tree paper, they suggest a new approach to fine-tune how much of the privacy budget to spend at each step rather than uniform spending.
    - Use as little data as `n x m = 3000`
    - No algorithm in the paper, only math :(
    - They cite Fletcher's smooth sensitivity as state of the art
  - This [paper](https://cspri.seas.gwu.edu/sites/g/files/zaxdzs1446/f/downloads/semi-supervised_0.pdf) seems to offer high accuracy with random trees and small datasets
    - But abstract says it needs public data to enrich original dataset
  - Recommended to avoid continous attributes if possible 
- Some questions we should answer:
  - How much data do we expect to have 
  - What's our privacy budget 
  - What's an acceptable prediction accuracy for Zurich insurance? 50%? 80%? This will impact how much privacy we can guarantee

## 2020-07-27

- Gradient boosted trees: perform better than random forest.
- Smooth sensitivity: explore a little more
  - Increase or decrease noise addition so that it leaks less information about outliers
- DP SGD: not very fast, we probably don't want that

Next steps:
  - Find link to paper for distributed DPRF
    - \[gtheo\]: http://fac.ksu.edu.sa/sites/default/files/a_privacy-preserving_algorithm_for_clinical_decision-support_systems_using_random_forest.pdf
  - Look for other prior work that doesn't necessarily involve random forest
    - https://xgboost.readthedocs.io/en/latest/tutorials/model.html
    - https://arxiv.org/abs/1611.01919
    - https://arxiv.org/abs/2001.09384
    - https://arxiv.org/pdf/1606.03572.pdf
  - TODO:
    - Find how much data each of these algorithm need
    - Find pros and cons
    - Summarize data for future comparison
 
## 2020-07-16

- Look for newer papers about DP random forest
- We don't want branching, hence why random trees are great for our use case
- If the method doesn't work, we can check another method.
- Check of attribute inference attacks + membership inference attacks.
  
 Next step: (2 weeks from now)
   - implement decision trees
   - (Optional) explore DP random forest papers
 
## 2020-06-25

- Insurance context:
	- Likelihood of the claim
	- Risk model on parameters
    - For cyber risks:
        - Mostly for companies, more heterogeneous
        - Harder to identify key risk drivers
        - Identify key processes in place
        - How much do these things influence the risk is still a question
        - Client not conveying the whole truth
        - Goal: understand what really drives cyber risk
    - If there was some kind of check post attacks, no one would buy the product
    
- Scientific context:
    - Tweak learning method to ensure privacy of customers
- Questionnaires
    - One from Zurich insurance
    - Insurance broker: middle man - independent party that seeks the best deal for the customer and then the broker goes to Zurich insurance
    - Two questionnaires are from brokers
    - Depending the questions, answers are different (binary, free text...)

- Thesis milestones:
    - Privacy preserving random forest running outside of SGX (as a start) that we can evaluate on categorical data.
    - Produce synthetique data which we can then use to prove the privacy of the system.
    - Perform a security assessment of the learning model.

- Differential privacy:
    - Datapoint should matter, but if you remove it it shouldn't change the result
    - The influence of every single data point can be denied