#!/bin/bash

CURR_DIR=$PWD

# compile and run all implementations
cd cpp_gbdt
./run --verify

cd python_gbdt


cd $CURR_DIR

cp cpp_gbdt/validation_logs/*.log validation/outputs
cp python_gbdt/validation_logs/*.log validation/outputs