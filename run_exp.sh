python train.py --dataroot ./datasets/new_cycle_bin --name 0628_cycle_pretrain --model cycle_gan --which_model_netG unet_256 --dataset_mode unaligned --no_lsgan --norm instance --pool_size 50 --loadSize 384 --fineSize 256 --save_epoch_freq 10 --no_dropout --print_freq 1 --display_freq 1 --continue_train --save_epoch_freq 400 --do_not_load_D

python train.py --dataroot ./datasets/new_pix_bin --name 0628_pix_fm_10 --model pix2pixms --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned --no_lsgan --norm instance --pool_size 50 --loadSize 384 --fineSize 256 --save_epoch_freq 10 --no_dropout --print_freq 1 --display_freq 1 --use_feature_matching --lambda_FM 10 --save_epoch_freq 400

python train.py --dataroot ./datasets/new_pix_bin --name 0628_pix_fm_1 --model pix2pixms --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned --no_lsgan --norm instance --pool_size 50 --loadSize 384 --fineSize 256 --save_epoch_freq 10 --no_dropout --print_freq 1 --display_freq 1 --use_feature_matching --lambda_FM 1 --save_epoch_freq 400

