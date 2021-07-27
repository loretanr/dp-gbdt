## DP-GBDT Meeting with Moritz

### Questions

- What are and have you been working on in dp-gbdt?
- and what is your goal/vision for dp-gbdt?
  - what would you be looking for in the C code?

### Where I'm coming from

- No ML background
- Thesis was supposed to be mainly about system security (side channels)
  - but 2 weeks of python to C++ translation became 2 months
  - necessary because hardening an unfinished algorithm is a bit useless
- Learned a lot about DP-GBDT, but lacking a lot of ML basics/theory/intuition

### Where I could use help

- 2nd-split idea
  - https://gitlab.inf.ethz.ch/kkari/enclave-hardening-ML/-/blob/master/code/stuff/theos_hypothesis/description.docx

- We want "usable and verifiable" C code
  - I don't trust the python code
    - all formulas exactly right
    - DPBoost paper results seem a tick better
  - Esfandiar: We could annotate the code very nicely, like many big comments that describe each formula that is being used
    - Idea: I could prepare that to the best of my ability
    - And then you or we together can go through it to see if it makes sense (and is aligned with the proof)



----------------------------------------------------------------------------------------

### Notes and findings

- probably can leave away scaling  [-1, 1]

​	*nice, that would give better accuracy. At least in the case of regression where y is a bit spread out (like in abalone). For classification it likely has no effect.*

- need to read about python cpp

  *numpy uses C under the hood, need to find out when and how*

- For DP-GBDT they sometimes don't use all columns for training. This speeds up things and also makes the exponential mechanism work better. Because if you have a lot of probabilities, and you divide each one through the total sum to get them into [0,1] then they get very similar and youre not choosing the good one as often as you want.
- actually DPBoost also forgets to add noise to the init_score, which is a mistake

