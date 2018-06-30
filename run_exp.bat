rem python train.py --dataroot ./datasets/sketch_eq --name sketch_pix2pix_eq --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 384 --fineSize 256

rem python train.py --dataroot ./datasets/sketch_eq --name sketch_pix2pix_eq_L10 --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_L1 10 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 384 --fineSize 256

rem python train.py --dataroot ./datasets/sketch_cycle --name sketch_cyclegan_eq --model cycle_gan --no_dropout --loadSize 384 --fineSize 256

rem python train.py --dataroot ./datasets/sketch1k_cycle --name sketch1k_cyclegan_eq --model cycle_gan --no_dropout --loadSize 384 --fineSize 256 --save_epoch_freq 20

rem python train.py --dataroot ./datasets/sketch1k --name sketch1k_pix2pix_eq --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_L1 10 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 384 --fineSize 256 --save_epoch_freq 20

rem python train.py --dataroot ./datasets/new_cycle --name newcycle --model cycle_gan --no_dropout --loadSize 768 --fineSize 256 --save_epoch_freq 20

rem python train.py --dataroot ./datasets/new_pix --name newpix --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_L1 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 768 --fineSize 256 --save_epoch_freq 20

rem python train.py --dataroot ./datasets/new_pix_bin --name newreg_l1 --model regression --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 256 --niter 200 --niter_decay 200

rem python train.py --dataroot ./datasets/new_pix_bin --name newreg_ce --model regression --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 256 --use_ce_loss  --niter 200 --niter_decay 200

rem python train.py --dataroot ./datasets/new_pix_bin --name newreg_pix --model pix2pix --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 256  --niter 200 --niter_decay 200 --lambda_L1 100

python train.py --dataroot ./datasets/new_pix_bin --name newreg_ce --model regression --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 256 --use_ce_loss  --niter 200 --niter_decay 200

python train.py --dataroot ./datasets/new_pix_bin --name newreg_ce_really --model regression --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 512 --use_ce_loss  --niter 100 --niter_decay 100

python train.py --dataroot ./datasets/new_pix_bin --name newreg_l1_really --model regression --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 512 --niter 100 --niter_decay 100

python train.py --dataroot ./datasets/new_pix_bin --name newreg_ce_gan --model regression --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --loadSize 512 --fineSize 256 --use_ce_loss --use_gan_loss --niter 200 --niter_decay 200 --lambda_L1 100
