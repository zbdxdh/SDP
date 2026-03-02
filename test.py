import time
import os
import argparse
import random
import numpy as np
import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from models.utils import Data_load
from models.core.models import create_model

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_srcdata', type=str, default=os.path.join(PROJECT_ROOT, 'test', 'AOP'))
    parser.add_argument('--test_maskdata', type=str, default=os.path.join(PROJECT_ROOT, 'test', 'Mask_dataset'))
    parser.add_argument('--checkpoints_dir', type=str, default=os.path.join(PROJECT_ROOT, 'checkpoints'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(PROJECT_ROOT, 'result'))

    parser.add_argument('--load_epoch', type=int, default=6)
    parser.add_argument('--batchSize', type=int, default=8)
    parser.add_argument('--fineSize', type=int, default=256)

    parser.add_argument('--input_nc', type=int, default=6)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--which_model_netP', type=str, default='unet_256')

    parser.add_argument('--name', type=str, default='SDP_SPP')
    parser.add_argument('--model', type=str, default='SDP_SPP')

    parser.add_argument('--gpu_ids', type=list, default=[0])
    parser.add_argument('--norm', type=str, default='instance')
    parser.add_argument('--use_dropout', type=bool, default=False)
    parser.add_argument('--init_type', type=str, default='normal')
    parser.add_argument('--mask_type', type=str, default='random')

    parser.add_argument('--init_gain', type=float, default=0.02)

    parser.add_argument('--overlap', type=int, default=4)

    parser.add_argument('--dcsa_heads', type=int, default=2)

    parser.add_argument('--use_polarized_loss', type=bool, default=False)

    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--which_epoch', type=str, default='8')

    parser.add_argument('--isTrain', type=bool, default=False)
    return parser


def resolve_channels(opt):
    if getattr(opt, 'use_polarized_loss', False):
        opt.input_nc = 6
        opt.output_nc = 6
    return opt

def build_transforms(opt):
    transform_mask = transforms.Compose([
        transforms.Resize((opt.fineSize, opt.fineSize)),
        transforms.ToTensor()
    ])

    transform = transforms.Compose([
        transforms.Resize((opt.fineSize, opt.fineSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform, transform_mask

def build_loader(opt, transform, transform_mask):
    dataset_test = Data_load(
        opt.test_srcdata,
        opt.test_maskdata,
        transform,
        transform_mask,
        use_multimodal_data=True,
        dop_root=os.path.join(PROJECT_ROOT, 'test', 'DOP')
    )

    print(f"测试集样本数: {len(dataset_test)}")

    if os.path.exists(opt.test_srcdata):
        src_files = os.listdir(opt.test_srcdata)
        print(f"源文件总数: {len(src_files)}")
        print(f"前10个源文件: {src_files[:10]}")
    else:
        print(f"警告: 源数据目录不存在: {opt.test_srcdata}")

    if os.path.exists(opt.test_maskdata):
        mask_files = os.listdir(opt.test_maskdata)
        print(f"掩码文件总数: {len(mask_files)}")
    else:
        print(f"警告: 掩码数据目录不存在: {opt.test_maskdata}")

    test_sample_count = min(10, len(dataset_test))
    print(f"将测试前 {test_sample_count} 个样本")

    subset_indices = list(range(test_sample_count))
    dataset_subset = data.Subset(dataset_test, subset_indices)
    iterator_test = data.DataLoader(dataset_subset, batch_size=opt.batchSize, shuffle=False)
    return dataset_test, subset_indices, iterator_test, test_sample_count


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_batch_images(save_dir, names, real_aop, real_b, fake_aop, full_fake, full_real_b, full_real_a,
                      syn, unknowregion, knownregion):
    for i, name in enumerate(names):
        torchvision.utils.save_image(real_aop[i], f'{save_dir}/{name}_real_A.png', normalize=True)
        torchvision.utils.save_image(real_b[i], f'{save_dir}/{name}_real_B.png', normalize=True)
        torchvision.utils.save_image(fake_aop[i], f'{save_dir}/{name}_fake_P.png', normalize=True)

        if full_real_a.size(1) == 6:
            masked_dop = full_real_a[i, 3:, :, :]
            torchvision.utils.save_image(masked_dop, f'{save_dir}/{name}_real_A_DOP.png', normalize=True)

        if full_fake.size(1) == 6 and full_real_b.size(1) == 6:
            fake_dop = full_fake[i, 3:, :, :]
            real_dop = full_real_b[i, 3:, :, :]
            torchvision.utils.save_image(real_dop, f'{save_dir}/{name}_real_DOP.png', normalize=True)
            torchvision.utils.save_image(fake_dop, f'{save_dir}/{name}_fake_DOP.png', normalize=True)

        torchvision.utils.save_image(syn[i], f'{save_dir}/{name}_Syn.png', normalize=True)
        torchvision.utils.save_image(unknowregion[i], f'{save_dir}/{name}_Unknowregion.png', normalize=True)
        torchvision.utils.save_image(knownregion[i], f'{save_dir}/{name}_knownregion.png', normalize=True)


def run_test(opt):
    device = get_device()
    transform, transform_mask = build_transforms(opt)
    dataset_test, subset_indices, iterator_test, test_sample_count = build_loader(opt, transform, transform_mask)

    model = create_model(opt)
    model.load(opt.load_epoch)
    model.netP.eval()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    psnr = PSNR(data_range=2).to(device)
    ssim = SSIM(data_range=2).to(device)

    sample_count = 0
    global_pos = 0
    with torch.no_grad():
        for image, mask in iterator_test:
            iter_start_time = time.time()
            batch_size = image.size(0)
            batch_indices = subset_indices[global_pos:global_pos + batch_size]
            global_pos += batch_size
            sample_count += batch_size

            print(f"=== 正在处理第 {min(sample_count, test_sample_count)}/{test_sample_count} 个样本 ===", end=' ')

            image = image.to(device)
            mask = mask.to(device).byte()

            model.set_input(image, mask)
            model.test()

            real_aop, real_b, fake_aop, syn, unknowregion, knownregion = model.get_current_visuals()
            full_fake = model.fake_P
            full_real_b = model.real_B
            full_real_a = model.real_A

            result_psnr = psnr(fake_aop, real_b).item()
            origin_psnr = psnr(real_aop, real_b).item()

            result_ssim = ssim(fake_aop, real_b).item()
            origin_ssim = ssim(real_aop, real_b).item()

            print(f"原始PSNR: {origin_psnr:.4f}, 修复后PSNR: {result_psnr:.4f} ", end='')
            print(f"原始SSIM: {origin_ssim:.4f}, 修复后SSIM: {result_ssim:.4f} ", end='')

            names = []
            for idx in batch_indices:
                if idx < len(dataset_test.paths):
                    name_ = os.path.basename(dataset_test.paths[idx])
                    names.append(os.path.splitext(name_)[0])
                else:
                    names.append(f"sample_{idx}")

            save_batch_images(opt.save_dir, names, real_aop, real_b, fake_aop, full_fake, full_real_b, full_real_a,
                              syn, unknowregion, knownregion)

            iter_time = time.time() - iter_start_time
            print(f"处理时间: {iter_time:.2f}秒")

    print(f"=== 测试完成！共处理了 {min(sample_count, test_sample_count)} 个样本 ===")


if __name__ == '__main__':
    parser = build_parser()
    opt = resolve_channels(parser.parse_args())
    run_test(opt)
