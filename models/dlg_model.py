import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import os


class DLGModel(BaseModel):
    def name(self):
        return 'DLGModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_TV', type=float, default=0.0)
            parser.add_argument('--lambda_L2', type=float, default=0.0)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'P_GAN', 'P_FM', 'P_L1', 'F_TV']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['real_A', 'fake_B', 'fake_C', 'fake_I', 'real_B', 'real_I']
        else:
            self.visual_names = ['real_A', 'fake_B', 'fake_C', 'fake_I']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'F', 'P', 'D', 'D2']
        else:  # during test time, only load Gs
            self.model_names = ['G', 'F', 'P']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netF = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netF, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netP = networks.define_G(opt.output_nc+opt.output_nc, opt.input_nc, opt.ngf,
                                      opt.which_model_netP, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netFM = networks.define_FM(opt.init_type, self.gpu_ids)
        self.netFM.load_state_dict(torch.load(os.path.join(self.opt.pretrained_model_dir, 'alexnet-owt-4df8aa71.pth')),
                                   strict=False)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD2 = networks.define_D(opt.input_nc, opt.ndf, opt.which_model_netD,
                                           opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_I_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(mse_loss=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_P = torch.optim.Adam(self.netP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            self.optimizers.append(self.optimizer_P)
            self.optimizers.append(self.optimizer_D2)

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        if self.isTrain:
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.real_I = input['A_' if AtoB else 'B_'].to(self.device)
            self.real_B_ = input['B_' if AtoB else 'A_'].to(self.device)
            self.image_paths = input['AB_paths']
        else:
            self.real_A = input['A'].to(self.device)
            self.image_paths = input['A_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        self.fake_C = self.netF(self.real_A)
        if self.isTrain:
            if not self.opt.use_deep_supervision and not self.opt.no_color_embedding:
                self.fake_I = self.netP(torch.cat((self.fake_B, self.fake_C), 1))
            elif self.opt.use_deep_supervision and not self.opt.no_color_embedding:
                self.fake_I = self.netP(torch.cat((self.real_B, self.fake_C), 1))
            elif not self.opt.use_deep_supervision and not self.opt.no_color_embedding:
                self.fake_I = self.netP(torch.cat((self.fake_B, self.real_A), 1))
            else:
                self.fake_I = self.netP(torch.cat((self.real_B, self.real_A), 1))
        else:
            self.fake_I = self.netP(torch.cat((self.fake_B, self.fake_C), 1))

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_D2(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_I = self.fake_I_pool.query(self.fake_I)
        pred_fake = self.netD2(fake_I.detach())
        self.loss_D2_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_I = self.real_I
        pred_real = self.netD2(real_I)
        self.loss_D2_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5

        self.loss_D2.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # P(B, C) should trick discriminator2
        fake_I = self.fake_I
        pred_fake = self.netD2(fake_I)
        self.loss_P_GAN = self.criterionGAN(pred_fake, True)

        # P(B, C) ~= A
        feature_A = self.netFM(self.real_A).detach()
        feature_A.requires_grad = False
        self.loss_P_FM = torch.nn.MSELoss()(self.netFM(self.fake_I), feature_A) * self.opt.lambda_FM
        self.loss_P_L1 = self.criterionL1(self.fake_I, self.real_A) * self.opt.lambda_L1_I

        # TV loss
        if self.opt.lambda_TV > 0:
            self.loss_F_TV = self.criterionTV(self.fake_C) * self.opt.lambda_TV
        else:
            self.loss_F_TV = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_P_GAN + self.loss_P_FM + self.loss_P_L1 + self.loss_F_TV

        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update D2
        self.set_requires_grad(self.netD2, True)
        self.optimizer_D2.zero_grad()
        self.backward_D2()
        self.optimizer_D2.step()

        # update G, F, P
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netD2, False)
        self.optimizer_G.zero_grad()
        self.optimizer_F.zero_grad()
        self.optimizer_P.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_F.step()
        self.optimizer_P.step()
