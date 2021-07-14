#!/bin/bash

CURR_DIR=$PWD
SHIFT_RIGHT='sed "s/^/    /"'
NRR=false
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

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
    make | eval "$SHIFT_RIGHT"
    echo -e "${CYAN}Running C++ verification ...${NC}"
    rm verification_logs/*.log 2> /dev/null
    ./run --verify | eval "$SHIFT_RIGHT"
    cd $CURR_DIR

    # run python implementation
    cd python_gbdt
    echo -e "${CYAN}Running python verification ...${NC}"
    rm verification/verification_logs/*.log 2> /dev/null
    python3 verification/verification.py | eval "$SHIFT_RIGHT"
    cd $CURR_DIR

    # collect the outputs
    rm verification/outputs/*.log 2> /dev/null
    cp cpp_gbdt/verification_logs/*.log verification/outputs
    cp python_gbdt/verification/verification_logs/*.log verification/outputs
fi



# compare the outputs
echo "------------ diff ---------------"
cd verification/outputs
for py_filename in *.python.log; do
    IFS='.'
    read -ra ADDR <<< "$py_filename"
    cpp_filename="${ADDR[0]}.cpp.log"
    IFS=' '
    DIFF_OUTPUT=$(icdiff --color-map='description:cyan,change:red_bold' -U 2 --cols=65 $py_filename $cpp_filename)
    if [ $(wc -l <<< "$DIFF_OUTPUT") -eq 1 ] ; then
        echo -e "${CYAN}$py_filename\n$cpp_filename${NC}"
        echo -e "${GREEN}files are equal${NC}"
    else
        echo "$DIFF_OUTPUT" | head -n 12  # only show first few lines
    fi
    echo "------------"
done

