#!/bin/bash

# Base directory path
base_path="/localhdd/hza214/Vitis_Libraries_2020/Vitis_Libraries/database/L2/demos/host"

# Loop through numbers with leading zeros
for num in $(seq 1 22); do
    # Create the target directory path
    dir_num=$(printf "%02d" $num)
    target_dir="${base_path}/q${dir_num}/2022_sf1_sql2fpga_fpga"
    
    # Make the directory if it doesn't exist
    mkdir -p "$target_dir"
    echo $num 
    # Copy the files to the target directory
    cp "cfgFunc_q${num}.hpp" "$target_dir"
    cp "q${num}.hpp" "$target_dir"
    cp "test_q${num}.cpp" "$target_dir"
done

echo "Directories created and files copied."

