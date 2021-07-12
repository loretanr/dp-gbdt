#!/bin/bash

CURR_DIR=$PWD
SHIFT_RIGHT='sed "s/^/    /"'
NRR=false

# use flag -nrr|--norerun to skip compiling and running -> just the comparison
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -nrr|--norerun) NRR=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$NRR" = false ] ; then

    # compile and run cpp
    echo "Compiling C++ ..."
    cd cpp_gbdt
    make | eval "$SHIFT_RIGHT"
    echo "Run C++ verification ..."
    ./run --verify | eval "$SHIFT_RIGHT"
    cd $CURR_DIR

    # run python implementation
    cd python_gbdt
    echo "Run python verification ..."
    python3 verification/verification.py | eval "$SHIFT_RIGHT"
    cd $CURR_DIR

    # fetch the outputs
    cp cpp_gbdt/verification_logs/*.log verification/outputs
    cp python_gbdt/verification/verification_logs/*.log verification/outputs
fi


# compare the outputs
echo "------------ diff ---------------"
DIFF_OUTPUT=$(icdiff --color-map='description:cyan,change:red_bold' -U 2 --cols=65 verification/outputs/*)
if [ $(wc -l <<< "$DIFF_OUTPUT") -eq 1 ] ; then
    echo "files are equal"
else
    echo "$DIFF_OUTPUT" | head -n 30  # only show first few lines
fi

