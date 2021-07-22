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







probably can leave away scaling -1, 1



read about python cpp

