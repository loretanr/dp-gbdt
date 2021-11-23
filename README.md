# Enclave hardening for private ML

This is the implementation of DP-GBDT as described in `resources/thesis.pdf` (chapters 4,5 and 6 mainly)

## Versions

To keep the code clean there exist 5 different versions:

`python_gbdt`: This is the (mostly) patched version of the DP-GBDT reference implementation that was created by T. Giovanna.

`cpp_gbdt`: the base C++ version. It does not run inside an enclave and is not hardened. It mainly exists for experimentation with the underlying algorithm. Several measures (e.g. multithreading) have been undertaken to boost its performance.

`enclave_gbdt`: This is cpp_gbdt but ported into an SGX enclave.

`hardened_gbdt`: the hardened version of cpp_gbdt. It reflects the minimum changes that are needed to achieve ε-differential privacy. The code does not run inside an SGX enclave.

`hardened_sgx_gbdt`: the final combination of hardened_gbdt and enclave_gbdt.




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

## CPP-DP-GBDT

Components:
- **main.cpp**
If you just want to play around with different parameters, different logging levels etc. you can do this in here. Use `make`, then `./run`.
Note, this version does not use threads, to make stuff obvious to debug. So it's naturally slower.
- **benchmark.cpp**
this component demonstrates the potential speed of the CPP implementation. It e.g. takes advantage of multithreading. To use it, adjust _benchmark.cpp_ according to your needs, compile the project with `make fast`, then do `./run --bench`.
- **evaluation.cpp**
this component allows running the model successively with multiple privacy budgets. It will create a .csv in the results/ directory. In there you can use _plot.py_ to create plots from these files. To compile and run, use `make fast`, then `./run --eval`. Be aware, running the code with _use_dp=false_ resp. _privacy_budget=0_ is much slower than using dp (because it uses **all** sample rows for each tree).
- **verification.cpp**
this is just to show that our algorithm results are consistent with the python implementation. (you can run this with _verify.sh_). It works by running both implementations without randomness, and then comparing intermediate values.


### Running
- **Running the verification script to compare Python and C++**
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
(./run --eval)
(./run --bench)
```

## Limitations
as of right now:

- C++ algorithm can do **regression** and **binary classification**.
  - **regression** can be performed on abalone & yearMSD
  - **classification** can be performed on the adult & BCW
  - though it is easy to add new datasets (have a look at _dataset_parser.cpp_)
- python_gbdt is missing the correct GDF functionality
- There are still small DP problems, such as
  - init\_score is leaking information about what values are present in a dataset
  - constant time floating point operations are only partially adopted

