import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import itertools


class RegressionModel(BaseModel):
    def name(self):
        return 'RegressionModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(which_model_netG='unet_256')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_CE']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.isTrain:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        else:
            self.visual_names = ['real_A', 'fake_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.class_weights = None if opt.weights is None else self.Tensor(opt.weights)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      use_ce_loss=True)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = []
            for scale in self.opt.scale_factor:
                print('scale = {}'.format(scale))
                self.netD.append(networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                                   opt.which_model_netD,
                                                   opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                                   scale=scale))

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionL1 = networks.WeightedL1Loss()

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            params = itertools.chain()
            for netD in self.netD:
                params = itertools.chain(params, netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        if self.isTrain:
            AtoB = self.opt.which_direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_paths' if AtoB else 'B_paths']
        else:
            self.real_A = input['A'].to(self.device)
            self.image_paths = input['A_paths']

    def forward(self):
        self.logit_B = self.netG(self.real_A)
        self.fake_B = torch.nn.Tanh()(self.logit_B)
        # self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.loss_D_fake = 0
        for netD in self.netD:
            pred_fake = netD(fake_AB.detach())
            self.loss_D_fake += self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.loss_D_real = 0
        for netD in self.netD:
            pred_real = netD(real_AB)
            self.loss_D_real += self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        self.loss_G = 0

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.loss_G_GAN = 0
        for netD, lambda_D in zip(self.netD, self.opt.lambda_D):
            pred_fake = netD(fake_AB)
            self.loss_G_GAN += self.criterionGAN(pred_fake, True) * lambda_D

        # Cross-entropy loss
        # weight = (self.real_B + 1) / 2.0 * self.class_weights[0] + (1 - (self.real_B + 1) / 2.0) * self.class_weights[1]
        # self.loss_G_CE = torch.nn.BCELoss(weight=weight)((self.fake_B + 1) / 2.0, (self.real_B + 1) / 2.0)
        if self.class_weights is None:
            self.loss_G_CE = torch.nn.BCEWithLogitsLoss()(self.logit_B, (self.real_B + 1) / 2.0)
        else:
            weight = (self.real_B + 1) / 2.0 * self.class_weights[0] + (1 - (self.real_B + 1) / 2.0) * \
                     self.class_weights[1]
            self.loss_G_CE = torch.nn.BCEWithLogitsLoss(weight=weight)(self.logit_B, (self.real_B + 1) / 2.0)

        if self.opt.use_gan_loss:
            self.loss_G += self.loss_G_GAN

        # Second, G(A) = B
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)

        if self.opt.use_ce_loss:
            self.loss_G += self.loss_G_CE * self.opt.lambda_L1
            # print('cross entropy loss: {}'.format(self.loss_G_CE))
        else:
            self.loss_G += self.loss_G_L1 * self.opt.lambda_L1
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
