#!/bin/bash

for number in {1..4}
do
    echo "Running test_phase_count_${number}.py ..."
    python /script/test/test_all_combination/test_phase_count_${number}.py
done
