import os
import torch
from collections import OrderedDict
from torch.autograd import Variable
from .. import networks
from torchvision import models
from collections import namedtuple


def create_model(opt):
    print(opt.model)
    if opt.model == 'SDP_SPP':
        # 创建模型实例
        model = SDP()
    else:
        raise ValueError("Model [%s] creation failed." % opt.model)
    model.initialize(opt)
    print("model [%s] was successfully created/loaded" % (model.name()))
    return model


from .. import losses


class Feature(torch.nn.Module):
    
    def __init__(self, requires_grad=False):
        super(Feature, self).__init__()
        # VGG16 特征提取
        pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(5):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(5, 10):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(10, 17):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(17, 23):
            self.slice4.add_module(str(x), pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class BaseModel():
    # 模型基类
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        # 初始化模型状态
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        if os.path.exists(self.save_dir) is False:
            os.makedirs(self.save_dir)
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)

        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


class SDP(BaseModel):

    def name(self):
        return 'SDP__Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device('cuda')
        self.opt = opt
        self.isTrain = opt.isTrain

        self.Feature_ex = Feature(requires_grad=False) 
        self.Feature_ex = self.Feature_ex.cuda() 

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        self.mask_global = torch.ByteTensor(1, 1, opt.fineSize, opt.fineSize)
        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.fineSize / 4) + self.opt.overlap: int(self.opt.fineSize / 2) + int(
            self.opt.fineSize / 4) - self.opt.overlap,\
        int(self.opt.fineSize / 4) + self.opt.overlap: int(self.opt.fineSize / 2) + int(
            self.opt.fineSize / 4) - self.opt.overlap] = 1
        self.mask_type = opt.mask_type
        self.gMask_opts = {}

        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()

        self.netP = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                               opt.which_model_netP, opt, self.mask_global, opt.norm, opt.use_dropout,
                                               opt.init_type, self.gpu_ids, opt.init_gain)
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True

            self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          opt.init_gain)
            self.netF = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netF,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          opt.init_gain)

            if hasattr(opt, 'use_polarized_loss') and opt.use_polarized_loss:
                self.polarized_loss = losses.HybridPolarizedLoss(
                    Lc_lambda=opt.Lc_lambda,
                    Lp_lambda=opt.Lp_lambda,
                    content_l1_loss_lambda=opt.content_l1_loss_lambda,
                    content_l2_loss_lambda=opt.content_l2_loss_lambda,
                    content_perceptual_loss_lambda=opt.content_perceptual_loss_lambda,
                    content_gradient_loss_lambda=opt.content_gradient_loss_lambda,
                    stokes_s0_loss_lambda=opt.stokes_s0_loss_lambda,
                    stokes_s12_loss_lambda=opt.stokes_s12_loss_lambda,
                    stokes_s12_relative_loss_lambda=opt.stokes_s12_relative_loss_lambda
                )

        if not self.isTrain or opt.continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netP, 'P', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netF, 'F', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            self.schedulers = []
            self.optimizers = []

            self.optimizer_P = torch.optim.Adam(self.netP.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_P)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            print('Networks initialized')
            networks.print_network(self.netP)
            if self.isTrain:
                networks.print_network(self.netD)
                networks.print_network(self.netF)
            print('-----------------------------')

    def set_input(self, input, mask):
        input_A = input
        if self.opt.input_nc == 6 and input.size(1) == 6:
                                                  
            if getattr(self.opt, 'output_nc', 3) == 6:
                input_B = input.clone()
            else:
                input_B = input[:, :3, :, :].clone()
        else:
            input_B = input.clone()
        input_mask = mask
        self.input_A.resize_(input_A.size()).copy_(input_A)  
        self.input_B.resize_(input_B.size()).copy_(input_B) 
        self.image_paths = 0
        if self.opt.mask_type == 'center':
            self.mask_global = self.mask_global
        elif self.opt.mask_type == 'random':
            self.mask_global.zero_()
                                              
            if input_mask.dim() == 4 and input_mask.size(1) != 1:
                input_mask = input_mask.any(dim=1, keepdim=True).byte()
            self.mask_global = input_mask
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)
        self.ex_mask = self.mask_global 
        self.inv_ex_mask = torch.add(torch.neg(self.ex_mask.float()), 1).byte() 

        mask_global_new = self.mask_global[:, 0:1, :, :]
        self.input_A.narrow(1, 0, 1).masked_fill_(mask_global_new.bool(), 2 * 123.0 / 255.0 - 1.0)
        self.input_A.narrow(1, 1, 1).masked_fill_(mask_global_new.bool(), 2 * 104.0 / 255.0 - 1.0)
        self.input_A.narrow(1, 2, 1).masked_fill_(mask_global_new.bool(), 2 * 117.0 / 255.0 - 1.0)
        self.input_A.narrow(1, 3, 1).masked_fill_(mask_global_new.bool(), 2 * 123.0 / 255.0 - 1.0)
        self.input_A.narrow(1, 4, 1).masked_fill_(mask_global_new.bool(), 2 * 104.0 / 255.0 - 1.0)
        self.input_A.narrow(1, 5, 1).masked_fill_(mask_global_new.bool(), 2 * 117.0 / 255.0 - 1.0)

    def forward(self):
        self.real_A = self.input_A.to(self.device)
        self.fake_P = self.netP(self.real_A)    
        self.un = self.fake_P.clone()
        if self.opt.input_nc == 6 and self.fake_P.size(1) == 3:
                                              
            batch_size, _, height, width = self.real_A.size()
            fake_P_expanded = self.fake_P.repeat(1, 2, 1, 1)  
            self.Unknowregion = fake_P_expanded.data.masked_fill_(self.inv_ex_mask.bool(), 0)
            self.knownregion = self.real_A.data.masked_fill_(self.ex_mask.bool(), 0)
            self.Syn = self.Unknowregion + self.knownregion
        else:
            self.Unknowregion = self.un.data.masked_fill_(self.inv_ex_mask.bool(), 0)
            self.knownregion = self.real_A.data.masked_fill_(self.ex_mask.bool(), 0)
            self.Syn = self.Unknowregion + self.knownregion
        self.real_B = self.input_B.to(self.device)

    def test(self):
        self.real_A = self.input_A.to(self.device)
        self.fake_P = self.netP(self.real_A)
        self.un = self.fake_P.clone()
        if self.opt.input_nc == 6 and self.fake_P.size(1) == 3:
            batch_size, _, height, width = self.real_A.size()
            fake_P_expanded = self.fake_P.repeat(1, 2, 1, 1)  
            self.Unknowregion = fake_P_expanded.data.masked_fill_(self.inv_ex_mask.bool(), 0)
            self.knownregion = self.real_A.data.masked_fill_(self.ex_mask.bool(), 0)
            self.Syn = self.Unknowregion + self.knownregion
        else:
            self.Unknowregion = self.un.data.masked_fill_(self.inv_ex_mask.bool(), 0)
            self.knownregion = self.real_A.data.masked_fill_(self.ex_mask.bool(), 0)
            self.Syn = self.Unknowregion + self.knownregion
        self.real_B = self.input_B.to(self.device)

    def backward_D(self): 
        fake_AB = self.fake_P
        if self.opt.input_nc == 6 and self.fake_P.size(1) == 6:
            fake_for_vgg = self.fake_P[:, :3, :, :]
        else:
            fake_for_vgg = self.fake_P
        self.gt_latent_fake = self.Feature_ex(Variable(fake_for_vgg.data, requires_grad=False))
        if self.opt.input_nc == 6 and self.real_B.size(1) == 6:
            self.gt_latent_real = self.Feature_ex(Variable(self.real_B[:, :3, :, :], requires_grad=False))
        else:
            self.gt_latent_real = self.Feature_ex(Variable(self.input_B, requires_grad=False))
        real_AB = self.real_B
        self.pred_fake = self.netD(fake_AB.detach())
        if self.opt.input_nc == 6 and self.real_B.size(1) == 6 and fake_AB.size(1) == 6:
            self.pred_real = self.netD(self.real_B)
        else:
            self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)

        self.pred_fake_F = self.netF(self.gt_latent_fake.relu3_3.detach())
        self.pred_real_F = self.netF(self.gt_latent_real.relu3_3)
        self.loss_F_fake = self.criterionGAN(self.pred_fake_F, self.pred_real_F, True)

        self.loss_D = self.loss_D_fake * 0.5 + self.loss_F_fake * 0.5
        self.loss_D.backward()

    def backward_G(self): 
        fake_AB = self.fake_P
        fake_f = self.gt_latent_fake
        pred_fake = self.netD(fake_AB)
        pred_fake_f = self.netF(fake_f.relu3_3)
        pred_real = self.netD(self.real_B)
        pred_real_F = self.netF(self.gt_latent_real.relu3_3)

        self.loss_G_GAN = self.criterionGAN(pred_fake, pred_real, False) + self.criterionGAN(pred_fake_f, pred_real_F,
                                                                                             False)
        if self.opt.input_nc == 6 and self.fake_P.size(1) == 3 and self.real_B.size(1) == 6:
            self.loss_G_L1 = (self.criterionL1(self.fake_P, self.real_B[:, :3, :, :])) * self.opt.lambda_A
        else:
            self.loss_G_L1 = (self.criterionL1(self.fake_P,self.real_B)) * self.opt.lambda_A
            
        if hasattr(self, 'polarized_loss') and self.opt.use_polarized_loss:
            if (self.opt.input_nc == 6 and self.real_A.size(1) == 6 and
                    self.real_B.size(1) == 6 and self.fake_P.size(1) == 6):
                real_aop = self.real_B[:, :3, :, :]
                real_dop = self.real_B[:, 3:, :, :]
                
                fake_aop = self.fake_P[:, :3, :, :]
                fake_dop = self.fake_P[:, 3:, :, :]
                
                polarized_losses = self.polarized_loss(fake_aop, fake_dop, real_aop, real_dop)
                self.loss_G_polarized = polarized_losses['total_loss']
                self.loss_G = self.loss_G_polarized
            else:
                self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.opt.gan_weight
        else:
            self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.opt.gan_weight

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward_D()  
        self.optimizer_D.step()
        self.optimizer_F.step()
        self.optimizer_P.zero_grad()
        self.backward_G()  
        self.optimizer_P.step()

    def get_current_errors(self):
        errors = OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                            ('G_L1', self.loss_G_L1.data.item()),
                            ('D', self.loss_D_fake.data.item()),
                            ('F', self.loss_F_fake.data.item())
                            ])
        
        if hasattr(self, 'loss_G_polarized'):
            errors['G_Polarized'] = self.loss_G_polarized.data.item()
            
        return errors

    def get_current_visuals(self):
        real_A = self.real_A.data
        fake_P = self.fake_P.data
        real_B = self.real_B.data
        Unknowregion = self.Unknowregion.data
        knownregion = self.knownregion.data
        Syn = self.Syn.data

        if self.opt.input_nc == 6:
            real_A = real_A[:, :3, :, :]
                            
            if fake_P.size(1) == 6:
                fake_P = fake_P[:, :3, :, :]
                real_B = real_B[:, :3, :, :]
                Unknowregion = Unknowregion[:, :3, :, :]
                knownregion = knownregion[:, :3, :, :]
                Syn = Syn[:, :3, :, :]
            else:
                real_B = real_B[:, :3, :, :]
                Unknowregion = Unknowregion[:, :3, :, :]
                knownregion = knownregion[:, :3, :, :]
                Syn = Syn[:, :3, :, :]

        return real_A, real_B, fake_P, Syn, Unknowregion, knownregion

    def save(self, epoch):
        self.save_network(self.netP, 'P', epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', epoch, self.gpu_ids)
        self.save_network(self.netF, 'F', epoch, self.gpu_ids)

    def load(self, epoch):
        self.load_network(self.netP, 'P', epoch)
