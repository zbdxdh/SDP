import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler


from .hybrid_modules import Transformer_KSFA, Transformer_DCSA, SKFusion, DCSA, SGFT


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, opt, mask_global, norm='batch', use_dropout=False,
             init_type='normal', gpu_ids=[], init_gain=0.02):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG == 'unet_256':
        netG = DualStreamUnetGenerator(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            dcsa_heads=getattr(opt, 'dcsa_heads', 2),
            dcsa_dropout=0.0,
        )
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'feature':
        netD = PFDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)

    return init_net(netD, init_type, init_gain, gpu_ids)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



class GANLoss(nn.Module):
    def __init__(self, gan_type='wgan_gp', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if gan_type == 'wgan_gp':
            self.loss = nn.MSELoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'vanilla':
            self.loss = nn.BCELoss()
        else:
            raise ValueError("GAN type [%s] not recognized." % gan_type)

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, y_pred_fake, y_pred, target_is_real):
        target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
        if (target_is_real):
            errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - target_tensor) ** 2) + torch.mean(
                (y_pred_fake - torch.mean(y_pred) + target_tensor) ** 2)) / 2
            return errD

        else:
            errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + target_tensor) ** 2) + torch.mean(
                (y_pred_fake - torch.mean(y_pred) - target_tensor) ** 2)) / 2
            return errG


class UnetGenerator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError(
            "UnetGenerator has been replaced by DualStreamUnetGenerator; "
            "use define_G(..., which_model_netG='unet_256', ...) to construct."
        )


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, act='relu', kernel_size=3, stride=1, use_dropout=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        bias = isinstance(norm_layer, type) and norm_layer == nn.InstanceNorm2d
        if isinstance(norm_layer, functools.partial):
            bias = norm_layer.func == nn.InstanceNorm2d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = norm_layer(out_ch) if norm_layer is not None else nn.Identity()
        self.act = nn.ReLU(inplace=True) if act == 'relu' else nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm_layer, use_dropout=False):
        super().__init__()
        bias = isinstance(norm_layer, type) and norm_layer == nn.InstanceNorm2d
        if isinstance(norm_layer, functools.partial):
            bias = norm_layer.func == nn.InstanceNorm2d
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=bias)
        self.norm = norm_layer(out_ch) if norm_layer is not None else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, norm_layer, act='relu', kernel_size=3, stride=1, use_dropout=use_dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.norm(x)
        x = self.act(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class DualStreamUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False,
                 dcsa_heads=2, dcsa_dropout=0.0):
        super().__init__()
        if input_nc != 6:
            raise ValueError(f"DualStreamUnetGenerator expects input_nc=6 (AOP3+DOP3), got {input_nc}")
                
        self.aop_stem = ConvBlock(3, ngf, norm_layer, act='relu', kernel_size=3, stride=1, use_dropout=False)
        self.dop_stem = ConvBlock(3, ngf, norm_layer, act='relu', kernel_size=5, stride=1, use_dropout=False)
        self.stage1_fuse = nn.Conv2d(ngf * 2, ngf, kernel_size=1, bias=False)
                         
        ch1, ch2, ch3, ch4 = ngf, ngf * 2, ngf * 4, ngf * 8
        self.aop_down1 = ConvBlock(ch1, ch2, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.aop_down2 = ConvBlock(ch2, ch3, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.aop_down3 = ConvBlock(ch3, ch4, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.aop_down4 = ConvBlock(ch4, ch4, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.aop_down5 = ConvBlock(ch4, ch4, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.aop_down6 = ConvBlock(ch4, ch4, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.aop_down7 = ConvBlock(ch4, ch4, norm_layer, act='leaky', kernel_size=4, stride=2)

        self.dop_down1 = ConvBlock(ch1, ch2, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.dop_down2 = ConvBlock(ch2, ch3, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.dop_down3 = ConvBlock(ch3, ch4, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.dop_down4 = ConvBlock(ch4, ch4, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.dop_down5 = ConvBlock(ch4, ch4, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.dop_down6 = ConvBlock(ch4, ch4, norm_layer, act='leaky', kernel_size=4, stride=2)
        self.dop_down7 = ConvBlock(ch4, ch4, norm_layer, act='leaky', kernel_size=4, stride=2)
                  
        self.sk_64 = SKFusion(dim=ch3)
        self.sk_32 = SKFusion(dim=ch4)
                         
        self.sgft_32 = SGFT(ch4)
               
        self.dcsa_16 = DCSA(dim=ch4, num_heads=dcsa_heads, dropout=dcsa_dropout)
        self.dcsa_8 = DCSA(dim=ch4, num_heads=dcsa_heads, dropout=dcsa_dropout)
                   
        self.up1 = UpBlock(ch4, ch4, ch4, norm_layer, use_dropout=use_dropout)
        self.up2 = UpBlock(ch4, ch4, ch4, norm_layer, use_dropout=use_dropout)
        self.up3 = UpBlock(ch4, ch4, ch4, norm_layer, use_dropout=use_dropout)
        self.up4 = UpBlock(ch4, ch4, ch4, norm_layer, use_dropout=use_dropout)
        self.up5 = UpBlock(ch4, ch3, ch3, norm_layer, use_dropout=use_dropout)
        self.up6 = UpBlock(ch3, ch2, ch2, norm_layer, use_dropout=use_dropout)
        self.up7 = UpBlock(ch2, ch1, ch1, norm_layer, use_dropout=use_dropout)

        self.out_conv = nn.Sequential(
            nn.Conv2d(ch1, output_nc, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        aop = x[:, :3]
        dop = x[:, 3:]
        dop_gray = dop.mean(dim=1, keepdim=True)

        fa0 = self.aop_stem(aop)       
        fd0 = self.dop_stem(dop)       
        fused0 = self.stage1_fuse(torch.cat([fa0, fd0], dim=1))
        fa0 = fused0                                                

        fa1 = self.aop_down1(fa0)       
        fd1 = self.dop_down1(fd0)       

        fa2 = self.aop_down2(fa1)      
        fd2 = self.dop_down2(fd1)      
        fa2 = fa2 + self.sk_64(fa2, fd2)

        fa3 = self.aop_down3(fa2)      
        fd3 = self.dop_down3(fd2)      
        fa3 = self.sgft_32(fa3, dop_gray)
        fa3 = fa3 + self.sk_32(fa3, fd3)

        fa4 = self.aop_down4(fa3)      
        fd4 = self.dop_down4(fd3)      
        fa4 = self.dcsa_16(fa4, fd4)

        fa5 = self.aop_down5(fa4)     
        fd5 = self.dop_down5(fd4)     
        fa5 = self.dcsa_8(fa5, fd5)

        fa6 = self.aop_down6(fa5)     
        _fd6 = self.dop_down6(fd5)     

        fa7 = self.aop_down7(fa6)     
        _fd7 = self.dop_down7(_fd6)     

        x = self.up1(fa7, fa6)     
        x = self.up2(x, fa5)     
        x = self.up3(x, fa4)      
        x = self.up4(x, fa3)      
        x = self.up5(x, fa2)      
        x = self.up6(x, fa1)       
        x = self.up7(x, fa0)       
        return self.out_conv(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PFDiscriminator(nn.Module):
    def __init__(self):
        super(PFDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1))

    def forward(self, input):
        return self.model(input)
