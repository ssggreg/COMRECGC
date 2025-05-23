#!/bin/bash

# Run COMRecGC experiments
python3 comrecgc.py --dataset mutagenicity
python3 comrecgc.py --dataset aids
python3 comrecgc.py --dataset nci1
python3 comrecgc.py --dataset proteins --theta 0.15

# Run Common Recourse experiments
python3 common_recourse.py --dataset mutagenicity
python3 common_recourse.py --dataset aids
python3 common_recourse.py --dataset nci1
python3 common_recourse.py --dataset proteins --theta 0.15