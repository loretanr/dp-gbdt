#!/bin/bash

#
# Script to verify that hardened gives same output as non-hardened cpp
#

CURR_DIR=$PWD
SHIFT_RIGHT='sed "s/^/    /"'
NRR=false
CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

NON_HARDENED_PATH="../../cpp_gbdt"
HARDENED_PATH="."
COMBINED_OUTPUT_PATH="${HARDENED_PATH}/verification/combined_outputs"

# mkdirs if not already exist
mkdir -p $COMBINED_OUTPUT_PATH
mkdir -p ${NON_HARDENED_PATH}/verification_logs
mkdir -p ${HARDENED_PATH}/verification/verification_logs

# download year dataset if not present (cause it's not on git)
cd datasets/real/
python3 -u get_year.py | eval "$SHIFT_RIGHT"
cd $CURR_DIR

# compile and run non-hardened
echo -e "${CYAN}Compiling non-hardened ...${NC}"
cd $NON_HARDENED_PATH
# make clean | eval "$SHIFT_RIGHT"
make | eval "$SHIFT_RIGHT"
echo -e "${CYAN}Running non-hardened verification ...${NC}"
rm verification_logs/*.log 2> /dev/null
./run --verify | eval "$SHIFT_RIGHT"
cd $CURR_DIR

# compile and run hardened
echo -e "${CYAN}Compiling hardened ...${NC}"
cd $HARDENED_PATH
# make clean | eval "$SHIFT_RIGHT"
make | eval "$SHIFT_RIGHT"
echo -e "${CYAN}Running hardened verification ...${NC}"
rm verification/verification_logs/*.log 2> /dev/null
./run --verify | eval "$SHIFT_RIGHT"
cd $CURR_DIR


# collect the outputs
rm ${COMBINED_OUTPUT_PATH}/*.log 2> /dev/null
cp ${NON_HARDENED_PATH}/verification_logs/*.log ${COMBINED_OUTPUT_PATH}
cp ${HARDENED_PATH}/verification/verification_logs/*.log ${COMBINED_OUTPUT_PATH}


# compare the outputs
echo "------------ diff ---------------"
cd $COMBINED_OUTPUT_PATH
# loop over python logs
for h_filename in *.hardened.log; do
    # get matching cpp log file
    IFS='.'
    read -ra ADDR <<< "$h_filename"
    cpp_filename="${ADDR[0]}.cpp.log"
    IFS=' '
    if ! test -f "$cpp_filename"; then
        echo "$cpp_filename does not exist, skipping"
        echo "------------"
        continue
    fi
    # file found, compare contents
    DIFF_OUTPUT=$(icdiff --color-map='description:cyan,change:red_bold' -U 2 --cols=65 $h_filename $cpp_filename)
    if [ $(wc -l <<< "$DIFF_OUTPUT") -eq 1 ] ; then
        echo -e "${CYAN}$h_filename\n$cpp_filename${NC}"
        echo -e "${GREEN}files (and thus trees, splits etc.) are equal${NC}"
    else
    numlines=$(diff $h_filename $cpp_filename | grep "^>" | wc -l)
    echo "$numlines lines not matching, starting with:" | eval "$SHIFT_RIGHT"
        echo "$DIFF_OUTPUT" | head -n 12  # change number here if you need more output lines
    fi
    echo "------------"
done

