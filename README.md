# Enclave hardening for private ML

WIP: repo for sharing and backup
- Happy for all kinds of feedback

## Limitations
as of right now:
- so far the C++ algorithm seems to _work_ for regression and binary classification.
  - **regression** can be performed on abalone & yearMSD
    - for yearMSD I have only tested it on small subsets of yearMSD because the dataset is so large
  - **classification** can be performed on the adult dataset
  - _"work"_ meaning it gives the same results as Theo's python code and that the results are more or less in the range of the DPBoost paper results.
    - I am unsure why DPBoost seems to perform a little better (accuracy) than our python/cpp code.
    - Therefore I need to go over the entire algorithm at some point and check all the formulas etc. sigh.
      - **I'd say it's very likely that the python code has certain things wrong that I took over.**
  
- There is a global variable RANDOMIZATION that turns on/off randomization (introduced by e.g. the exponential mechanism and through adding laplacian noise to leaves)
  - I have **not yet** seriously played around and tested the code **with randomization turned on**, as I'm still developing the algorithm and thus require deterministic runs to compare C++ and python

- C++ code is not optimized at all yet
  - Have to do some profiling, think about data movement, cacheing, etc.
  - Not planning to use threading (because it's not useful for my thesis, as it should run in an enclave)
  - so far it's compilied with -O0
  - still much faster than the python code


## Requirements
tested on a fresh Ubuntu 20.04.2 VM
```bash
sudo apt-get install libspdlog-dev
sudo apt-get install icdiff
sudo apt install python3-pip
python3 -m pip install -r code/python_gbdt/requirements.txt
```
and add this to your .bashrc
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/.../code/python_gbdt
```
I don't know how this is usually done, please tell if you do


## Running
- **Running the verification  script to compare Python and C++**
```bash
cd code/
./verify.sh
```
- **Running C++ gbdt**
```bash
cd code/cpp_gbdt/
make
./run
(./run --verify)
```
- **Running python gbdt**
```bash
cd code/python_gbdt/
python3 results/abalone/cross_val.py
python3 verification/verification.py
...
```
whichever you want to run