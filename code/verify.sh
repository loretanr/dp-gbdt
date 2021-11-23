#!/bin/bash

#
# Script to verify that both python and C++ gbdt produce the exact same output.
# This requires both implementations to have randomness turned off. That means 
# no shuffling of input data, no exp-mechanism, no addition of laplace noise to 
# the leaf values. 
# How it works: Both implementations write intermediate values (gradients & leaf 
# values) to logfiles after building each tree they build. This script (compiles
# and) runs both python and C++ gbdt, collects these logfiles and compares them.
# It helps tracking down if and where errors/differences start occuring. This is 
# convenient, because in gbdt tiny differences in intermediate values can lead to 
# completely different end results. (e.g. different gradients -> diff gains ->
# diff split chosen -> diff trees -> diff score)
#

CURR_DIR=$PWD
SHIFT_RIGHT='sed "s/^/    /"'
NRR=false
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

# mkdirs if not already exist
mkdir -p verification/outputs
mkdir -p cpp_gbdt/verification_logs
mkdir -p python_gbdt/verification/verification_logs

# download year dataset if not present (cause it's not on git)
# cd datasets/real/
# python3 -u get_year.py | eval "$SHIFT_RIGHT"
# cd $CURR_DIR


# use flag -nrr or --norerun to skip compiling and running
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -nrr|--norerun) NRR=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ "$NRR" = false ] ; then

    # compile and run cpp
    echo -e "${CYAN}Compiling C++ ...${NC}"
    cd cpp_gbdt
    # make clean | eval "$SHIFT_RIGHT"
    make | eval "$SHIFT_RIGHT"
    echo -e "${CYAN}Running C++ verification ...${NC}"
    rm verification_logs/*.log 2> /dev/null
    ./run --verify | eval "$SHIFT_RIGHT"
    cd $CURR_DIR

    # run python implementation
    cd python_gbdt
    echo -e "${CYAN}Running python verification ...${NC}"
    rm verification/verification_logs/*.log 2> /dev/null
    python3 -u verification/verification.py | eval "$SHIFT_RIGHT"
    cd $CURR_DIR

    # collect the outputs
    rm verification/outputs/*.log 2> /dev/null
    cp cpp_gbdt/verification_logs/*.log verification/outputs
    cp python_gbdt/verification/verification_logs/*.log verification/outputs
fi



# compare the outputs
echo "------------ diff ---------------"
cd verification/outputs
# loop over python logs
for py_filename in *.python.log; do
    # get matching cpp log file
    IFS='.'
    read -ra ADDR <<< "$py_filename"
    cpp_filename="${ADDR[0]}.cpp.log"
    IFS=' '
    if ! test -f "$cpp_filename"; then
        echo "$cpp_filename does not exist, skipping"
        echo "------------"
        continue
    fi
    # file found, compare contents
    DIFF_OUTPUT=$(icdiff --color-map='description:cyan,change:red_bold' -U 2 --cols=65 $py_filename $cpp_filename)
    if [ $(wc -l <<< "$DIFF_OUTPUT") -eq 1 ] ; then
        echo -e "${CYAN}$py_filename\n$cpp_filename${NC}"
        echo -e "${GREEN}files (and thus trees, splits etc.) are equal${NC}"
    else
    numlines=$(diff $py_filename $cpp_filename | grep "^>" | wc -l)
    echo "$numlines lines not matching, starting with:" | eval "$SHIFT_RIGHT"
        echo "$DIFF_OUTPUT" | head -n 12  # change number here if you need more output lines
    fi
    echo "------------"
done

