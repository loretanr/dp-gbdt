# Enclave hardening for private ML

WIP: repo for sharing and backup
- Happy for all kinds of feedback

## Limitations
as of right now:
- works for regression (tested with abalone and subset of yearMSD, "works" meaning we get the same results as the python code)
- I am currently working on adding the adult dataset (and thus classification)


## Requirements
tested on a fresh Ubuntu 20.04.2 VM
```bash
sudo apt-get install libspdlog-dev
sudo apt-get install icdiff
sudo apt install python3-pip
python3 -m pip install -r code/python_gbdt/requirements.txt
```
and to your .bashrc
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/.../code/python_gbdt
```
I don't know how this is usually done, please tell if you do


## Running
- Verification Py <--> C++
```bash
cd code/
./verify.sh
```
- Running C++
```bash
cd code/cpp_gbdt/
make
./run
(./run --verify)
```
- Running python gbdt
```bash
cd code/python_gbdt/
python3 results/abalone/cross_val.py
python3 verification/verification.py
...
```
whichever you want to run