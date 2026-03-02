import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

VGG19_FEATURES = models.vgg19(pretrained=True).features
CONV3_3_IN_VGG_19 = nn.Sequential(*list(VGG19_FEATURES.children())[:18]).cuda()

for param in CONV3_3_IN_VGG_19.parameters():
    param.requires_grad = False

class LaplaceOperator:
    
    def __init__(self):
        self.laplace_kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def compute(self, img):
        laplace_kernel = self.laplace_kernel.repeat(img.size(1), 1, 1, 1).to(img.device)
        
        grad = F.conv2d(img, laplace_kernel, padding=1, groups=img.size(1))
        return grad

class ContentLoss(nn.Module):
    
    def __init__(self, 
                 l1_loss_lambda=10,
                 l2_loss_lambda=100,
                 perceptual_loss_lambda=0.1,
                 gradient_loss_lambda=10):
        super(ContentLoss, self).__init__()
        self.l1_loss_lambda = l1_loss_lambda
        self.l2_loss_lambda = l2_loss_lambda
        self.perceptual_loss_lambda = perceptual_loss_lambda
        self.gradient_loss_lambda = gradient_loss_lambda
        
        self.laplace_operator = LaplaceOperator()
        
    def forward(self, fake_img, real_img):
        l1_loss = F.l1_loss(fake_img, real_img) * self.l1_loss_lambda
        
        l2_loss = F.mse_loss(fake_img, real_img) * self.l2_loss_lambda
        
        fake_features = CONV3_3_IN_VGG_19(fake_img)
        real_features = CONV3_3_IN_VGG_19(real_img).detach()
        fake_features = fake_features / (torch.norm(fake_features, dim=1, keepdim=True) + 1e-8)
        real_features = real_features / (torch.norm(real_features, dim=1, keepdim=True) + 1e-8)
        perceptual_loss = F.mse_loss(fake_features, real_features) * self.perceptual_loss_lambda
        
        fake_grad = self.laplace_operator.compute(fake_img)
        real_grad = self.laplace_operator.compute(real_img)
        gradient_loss = F.mse_loss(fake_grad, real_grad) * self.gradient_loss_lambda
        
        content_loss = l1_loss + l2_loss + perceptual_loss + gradient_loss
        
        return {
            'content_loss': content_loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'perceptual_loss': perceptual_loss,
            'gradient_loss': gradient_loss
        }

class StokesLoss(nn.Module):
    
    def __init__(self,
                 s0_loss_lambda=20,
                 s12_loss_lambda=500,
                 s12_relative_loss_lambda=500):
        super(StokesLoss, self).__init__()
        self.s0_loss_lambda = s0_loss_lambda
        self.s12_loss_lambda = s12_loss_lambda
        self.s12_relative_loss_lambda = s12_relative_loss_lambda
        
    def compute_stokes_from_aop_dop(self, aop, dop):
        epsilon = 1e-8
        
        S0 = aop * dop / (torch.sin(aop) + epsilon)
        S1 = aop * dop * torch.cos(aop)
        S2 = aop * dop * torch.sin(aop)
        
        return S0, S1, S2
    
    def compute_intensity_from_stokes(self, S0, S1, S2):
        I1 = (S0 + S1) / 2  
        I2 = (S0 + S2) / 2  
        I3 = (S0 - S1) / 2  
        I4 = (S0 - S2) / 2  
        
        return I1, I2, I3, I4
    
    def forward(self, fake_aop, fake_dop, real_aop, real_dop):
        fake_S0, fake_S1, fake_S2 = self.compute_stokes_from_aop_dop(fake_aop, fake_dop)
        real_S0, real_S1, real_S2 = self.compute_stokes_from_aop_dop(real_aop, real_dop)
        
        fake_I1, fake_I2, fake_I3, fake_I4 = self.compute_intensity_from_stokes(fake_S0, fake_S1, fake_S2)
        real_I1, real_I2, real_I3, real_I4 = self.compute_intensity_from_stokes(real_S0, real_S1, real_S2)
        
        S0_loss = F.mse_loss(fake_S0, real_S0) * self.s0_loss_lambda
        
        S1_loss = F.mse_loss(fake_S1, real_S1) * self.s12_loss_lambda
        S2_loss = F.mse_loss(fake_S2, real_S2) * self.s12_loss_lambda
        
        epsilon = 1e-8
        fake_ratio = (fake_S2 + epsilon) / (fake_S1 + epsilon)
        real_ratio = (real_S2 + epsilon) / (real_S1 + epsilon)
        S12_relative_loss = F.mse_loss(fake_ratio, real_ratio) * self.s12_relative_loss_lambda
        
        stokes_loss = S0_loss + S1_loss + S2_loss + S12_relative_loss
        
        return {
            'stokes_loss': stokes_loss,
            'S0_loss': S0_loss,
            'S1_loss': S1_loss,
            'S2_loss': S2_loss,
            'S12_relative_loss': S12_relative_loss,
            'intensities': {
                'fake_I1': fake_I1,
                'fake_I2': fake_I2,
                'fake_I3': fake_I3,
                'fake_I4': fake_I4,
                'real_I1': real_I1,
                'real_I2': real_I2,
                'real_I3': real_I3,
                'real_I4': real_I4
            }
        }

class HybridPolarizedLoss(nn.Module):
    
    def __init__(self,
                 Lc_lambda=1.5,
                 Lp_lambda=1.0,
                 content_l1_loss_lambda=10,
                 content_l2_loss_lambda=100,
                 content_perceptual_loss_lambda=0.1,
                 content_gradient_loss_lambda=10,
                 stokes_s0_loss_lambda=20,
                 stokes_s12_loss_lambda=500,
                 stokes_s12_relative_loss_lambda=500):
        super(HybridPolarizedLoss, self).__init__()
        self.Lc_lambda = Lc_lambda
        self.Lp_lambda = Lp_lambda
        
        self.content_loss = ContentLoss(
            l1_loss_lambda=content_l1_loss_lambda,
            l2_loss_lambda=content_l2_loss_lambda,
            perceptual_loss_lambda=content_perceptual_loss_lambda,
            gradient_loss_lambda=content_gradient_loss_lambda
        )
        
        self.stokes_loss = StokesLoss(
            s0_loss_lambda=stokes_s0_loss_lambda,
            s12_loss_lambda=stokes_s12_loss_lambda,
            s12_relative_loss_lambda=stokes_s12_relative_loss_lambda
        )
        
    def forward(self, guiding_aop_pred, guiding_dop_pred, aop_gt, dop_gt):
        aop_losses = self.content_loss(guiding_aop_pred, aop_gt)
        dop_losses = self.content_loss(guiding_dop_pred, dop_gt)
        
        Lc_content = (aop_losses['content_loss'] + dop_losses['content_loss']) / 2
        Lc = Lc_content * self.Lc_lambda
        
        stokes_losses = self.stokes_loss(guiding_aop_pred, guiding_dop_pred, aop_gt, dop_gt)
        Lp_stokes = stokes_losses['stokes_loss']
        
        Lp = Lp_stokes * self.Lp_lambda
        
        total_loss = Lc + Lp
        
        losses = {
            'total_loss': total_loss,
            'Lc': Lc,
            'Lp': Lp,
            'Lc_content': Lc_content,
            'Lp_stokes': Lp_stokes,
            'aop_losses': aop_losses,
            'dop_losses': dop_losses,
            'stokes_details': stokes_losses
        }
        
        return losses
