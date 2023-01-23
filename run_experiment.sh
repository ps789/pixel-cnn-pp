#!/bin/bash
/usr/bin/ps789
rm -r runs/apcnn_kernel_block4
python3 main.py --block_dim 4 #--load_params models/block4_alt_109.pth #models/block4_alt_59.pth
#python generate.py --load_params models/block1_alt_149.pth
