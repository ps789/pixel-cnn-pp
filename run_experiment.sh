#!/bin/bash
/usr/bin/ps789
#rm -r runs/conditional1
#python3 main.py --block_dim 4 --load_params models/block4_alt_109.pth #models/block4_alt_59.pth
python3 main.py --block_dim 1 --nr_resnet 3 --nr_filters 160 --name small_block_conditional --conditional #--load_params models/block2_alt_139.pth
#python generate.py --load_params models/pcnn_block2_79.pth --block_dim 2
#python generate.py --load_params models/small_block1_29.pth --block_dim 1 --nr_resnet 3 #--conditional --nr_resnet 3
