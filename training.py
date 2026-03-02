           
import time

from models.utils import Data_load

from models.core.models import create_model
import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import argparse
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('--train_dataset', type=str, default='./Dataset/AP')
parser.add_argument('--dop_dataset', type=str, default='./Dataset/DP')
parser.add_argument('--mask_dataset', type=str, default='./Dataset/Mask_dataset')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')

parser.add_argument('--batchSize', type=int, default=8)
parser.add_argument('--fineSize', type=int, default=256)


parser.add_argument('--input_nc', type=int, default=6)

parser.add_argument('--output_nc', type=int, default=3)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)


parser.add_argument('--which_model_netD', type=str, default='basic')
parser.add_argument('--which_model_netF', type=str, default='feature')

parser.add_argument('--which_model_netP', type=str, default='unet_256')

parser.add_argument('--name', type=str, default='SDP_SPP')
parser.add_argument('--model', type=str, default='SDP_SPP')
parser.add_argument('--n_layers_D', type=str, default='3')


parser.add_argument('--gpu_ids', type=list, default=[0])
parser.add_argument('--norm', type=str, default='instance')
parser.add_argument('--use_dropout', type=bool, default=False)
parser.add_argument('--init_type', type=str, default='normal')
parser.add_argument('--mask_type', type=str, default='random')


parser.add_argument('--lambda_A', type=int, default=100)
parser.add_argument('--init_gain', type=float, default=0.02)
parser.add_argument('--gan_type', type=str, default='lsgan')
parser.add_argument('--gan_weight', type=int, default=0.2)
parser.add_argument('--overlap', type=int, default=4)


parser.add_argument('--use_polarized_loss', type=bool, default=False)
parser.add_argument('--Lc_lambda', type=float, default=1.5)
parser.add_argument('--Lp_lambda', type=float, default=1.0)
parser.add_argument('--content_l1_loss_lambda', type=float, default=10)
parser.add_argument('--content_l2_loss_lambda', type=float, default=100)
parser.add_argument('--content_perceptual_loss_lambda', type=float, default=0.1)
parser.add_argument('--content_gradient_loss_lambda', type=float, default=10)
parser.add_argument('--stokes_s0_loss_lambda', type=float, default=20)
parser.add_argument('--stokes_s12_loss_lambda', type=float, default=500)
parser.add_argument('--stokes_s12_relative_loss_lambda', type=float, default=500)

parser.add_argument('--dcsa_heads', type=int, default=2)

parser.add_argument('--save_epoch_freq', type=int, default=2)
parser.add_argument('--continue_train', type=bool, default=False)   
parser.add_argument('--epoch_count', type=int, default=1)
parser.add_argument('--which_epoch', type=str, default='14')   


parser.add_argument('--niter', type=int, default=2)
parser.add_argument('--niter_decay', type=int, default=8)

parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lr_policy', type=str, default='lambda')

parser.add_argument('--isTrain', type=bool, default=True)


opt = parser.parse_args()
if getattr(opt, 'use_polarized_loss', False):
    opt.input_nc = 6
    opt.output_nc = 6

transform_mask = transforms.Compose(
    [transforms.Resize((opt.fineSize, opt.fineSize)),
     transforms.ToTensor(), ])
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.Resize((opt.fineSize, opt.fineSize)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

dataset_train = Data_load(opt.train_dataset, opt.mask_dataset, transform, transform_mask,
                         use_multimodal_data=True, dop_root=opt.dop_dataset)

iterator_train = (data.DataLoader(dataset_train, batch_size=opt.batchSize, shuffle=True))
print(len(dataset_train))

writer = SummaryWriter("LOGS")

model = create_model(opt)

total_steps = 0
iter_start_time = time.time()


for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()

    for image, mask in iterator_train:

        image = image.cuda()
        mask = mask.cuda()

        mask = mask.byte()
        total_steps += opt.batchSize

        model.set_input(image, mask) 
        model.optimize_parameters()  

        if total_steps % 1 == 0:

            real_A, real_B, fake_P, Syn, Unknowregion, knownregion = model.get_current_visuals()

            real_A = torch.clamp(real_A, -1, 1)  
            real_B = torch.clamp(real_B, -1, 1)  
            fake_P = torch.clamp(fake_P, -1, 1)  

            device = fake_P.device
            mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5], device=device).view(3, 1, 1)
            real_A_un = real_A * std + mean  
            fake_P_un = fake_P * std + mean  
            real_B_un = real_B * std + mean  

            input = torchvision.utils.make_grid(real_B_un, nrow=4)                       
            out = torchvision.utils.make_grid(fake_P_un, nrow=4)
            add_mask = torchvision.utils.make_grid(real_A_un, nrow=4)

            errors = model.get_current_errors()
            
            for loss_name, loss_value in errors.items():
                writer.add_scalar(loss_name, loss_value, total_steps)
            
            writer.add_image("result", out, total_steps)
            writer.add_image("add_mask", add_mask, total_steps)
            writer.add_image("input", input, total_steps)
            
    
    if epoch % opt.save_epoch_freq == 0:
        print('Save at the end of the epoch %d, iters %d' %(epoch, total_steps))
        model.save(epoch)
    
    print('Epoch end %d / %d \t Total Time: %d sec'
          %(epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    
    model.update_learning_rate()

writer.close()
