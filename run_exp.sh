set -e

# python train.py --dataroot ./datasets/dlg --name 0716_dlg_ff --model dlg --which_model_netG unet_128 --which_model_netF resnet_6blocks --which_model_netP unet_128 --which_direction AtoB --lambda_L1 100 --lambda_FM 1 --lambda_L1_I 1 --dataset_mode dlg --no_lsgan --norm instance --pool_size 0 --loadSize 512 --fineSize 256 --save_epoch_freq 50 --no_dropout --print_freq 1 --display_freq 1

# python train.py --dataroot ./datasets/dlg --name 0716_dlg_rf --model dlg --which_model_netG unet_128 --which_model_netF resnet_6blocks --which_model_netP unet_128 --which_direction AtoB --lambda_L1 100 --lambda_FM 1 --lambda_L1_I 1 --dataset_mode dlg --no_lsgan --norm instance --pool_size 0 --loadSize 512 --fineSize 256 --save_epoch_freq 50 --no_dropout --print_freq 1 --display_freq 1 --use_deep_supervision

# python train.py --dataroot ./datasets/dlg --name 0716_dlg_rr --model dlg --which_model_netG unet_128 --which_model_netF resnet_6blocks --which_model_netP unet_128 --which_direction AtoB --lambda_L1 100 --lambda_FM 1 --lambda_L1_I 1 --dataset_mode dlg --no_lsgan --norm instance --pool_size 0 --loadSize 512 --fineSize 256 --save_epoch_freq 50 --no_dropout --print_freq 1 --display_freq 1 --use_deep_supervision --no_color_embedding

# python train.py --dataroot ./datasets/dlg --name 0716_dlg_fr --model dlg --which_model_netG unet_128 --which_model_netF resnet_6blocks --which_model_netP unet_128 --which_direction AtoB --lambda_L1 100 --lambda_FM 1 --lambda_L1_I 1 --dataset_mode dlg --no_lsgan --norm instance --pool_size 0 --loadSize 512 --fineSize 256 --save_epoch_freq 50 --no_dropout --print_freq 1 --display_freq 1 --no_color_embedding


python train.py --dataroot ./datasets/new_pix_bin --name 0717_wce_384 --model regression --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned2 --no_lsgan --norm instance --pool_size 50 --loadSize 384 --fineSize 256 --use_gan_loss --niter 50 --niter_decay 50 --lambda_L1 100 --output_nc 1 --use_ce_loss --display_freq 1 --weights 0.12 0.88

python train.py --dataroot ./datasets/new_pix_bin --name 0717_wce_512 --model regression --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned2 --no_lsgan --norm instance --pool_size 50 --loadSize 512 --fineSize 256 --use_gan_loss --niter 50 --niter_decay 50 --lambda_L1 100 --output_nc 1 --use_ce_loss --display_freq 1 --weights 0.12 0.88

python train.py --dataroot ./datasets/new_pix_bin --name 0717_wce_1024 --model regression --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned2 --no_lsgan --norm instance --pool_size 50 --loadSize 1024 --fineSize 256 --use_gan_loss --niter 50 --niter_decay 50 --lambda_L1 100 --output_nc 1 --use_ce_loss --display_freq 1 --weights 0.12 0.88

python train.py --dataroot ./datasets/new_pix_bin --name 0717_wce_2048 --model regression --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned2 --no_lsgan --norm instance --pool_size 50 --loadSize 2048 --fineSize 256 --use_gan_loss --niter 50 --niter_decay 50 --lambda_L1 100 --output_nc 1 --use_ce_loss --display_freq 1 --weights 0.12 0.88
