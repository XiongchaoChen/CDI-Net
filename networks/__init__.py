import torch.nn as nn
import numpy as np
from utils import arange
from networks.networks import UNet, RDN, SERDUNet, scSERDUNet, scSERDUNet3, DUNet, CapUNet, DenseUNet, Dis
import pdb


def set_gpu(network, gpu_ids):
    network.to(gpu_ids[0])  # Default to the 1st GPU
    network = nn.DataParallel(network, device_ids=gpu_ids)  # Parallel computing on multiple GPU

    return network


def get_generator(name, opts, ic):
    # (2) DuRDN / default_depth = 4
    if name == 'DuRDN4':
        network = scSERDUNet(n_channels=ic, n_filters=opts.net_filter, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm=opts.norm, dropout=opts.dropout)

    # (2) DuRDN
    elif name == 'DuRDN3':
        network = scSERDUNet3(n_channels=ic, n_filters=opts.net_filter, n_denselayer=opts.n_denselayer, growth_rate=opts.growth_rate, norm=opts.norm, dropout=opts.dropout)

   # (1) UNet
    elif name == 'UNet':
        network = UNet(in_channels=ic, residual=False, depth=opts.net_depth, wf=opts.UNet_filters, norm=opts.norm, dropout=opts.dropout)


    # # (3) Dense UNet (Dense + Residual)
    # elif name == 'DUNet':
    #     ic = 0
    #     if opts.use_em:
    #         ic = ic + 1
    #     if opts.use_sc:
    #         ic = ic + 1
    #     if opts.use_sc2:
    #         ic = ic + 1
    #     if opts.use_sc3:
    #         ic = ic + 1
    #     if opts.use_gender:
    #         ic = ic + 1
    #     if opts.use_bmi:
    #         ic = ic + 1
    #
    #     network = DUNet(n_channels=ic, n_filters=32, n_denselayer=6, growth_rate=32, norm=opts.norm)
    #
    #
    # elif name == 'DenseUNet':
    #     ic = 1
    #
    #     if opts.use_em:
    #         ic = ic + 1
    #     if opts.use_sc:
    #         ic = ic + 1
    #     if opts.use_sc2:
    #         ic = ic + 1
    #     if opts.use_sc3:
    #         ic = ic + 1
    #     if opts.use_gender:
    #         ic = ic + 1
    #     if opts.use_bmi:
    #         ic = ic + 1
    #
    #     network = DenseUNet(n_channels=ic, n_filters=32, n_denselayer=6, growth_rate=32)
    #
    # elif name == 'RDN':
    #     ic = 1
    #
    #     if opts.use_em:
    #         ic = ic + 1
    #     if opts.use_sc:
    #         ic = ic + 1
    #     if opts.use_sc2:
    #         ic = ic + 1
    #     if opts.use_sc3:
    #         ic = ic + 1
    #     if opts.use_gender:
    #         ic = ic + 1
    #     if opts.use_bmi:
    #         ic = ic + 1
    #
    #     network = RDN(n_channels=ic, n_filters=32, n_denselayer=6, growth_rate=32, n_blocks=5, dropout=0)
    #
    # elif name == 'SERDUNet':
    #     ic = 1
    #     if opts.use_scatter:
    #         ic = ic + 1
    #     if opts.use_scatter2:
    #         ic = ic + 1
    #     if opts.use_scatter3:
    #         ic = ic + 1
    #     if opts.use_bmi:
    #         ic = ic + 1
    #     if opts.use_gender:
    #         ic = ic + 1
    #     network = SERDUNet(n_channels=ic, n_filters=32, n_denselayer=6, growth_rate=32)
    #
    # elif name == 'CapUNet':
    #     ic = 1
    #     if opts.use_scatter:
    #         ic = ic + 1
    #     if opts.use_scatter2:
    #         ic = ic + 1
    #     if opts.use_scatter3:
    #         ic = ic + 1
    #     if opts.use_bmi:
    #         ic = ic + 1
    #     if opts.use_gender:
    #         ic = ic + 1
    #     network = CapUNet(in_channels=ic, nonlinearity='sqaush', dynamic_routing='local')

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters of Generator: {}'.format(num_param))

    return set_gpu(network, opts.gpu_ids)

# GAN Discriminator; Only one type
def get_discriminator(opts):
    network = Dis(input_dim = opts.patch_size_train[0], norm=opts.norm_D)  # 0 for chanel; normal CNN for discriminator

    num_param = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('Number of parameters of Discriminator: {}'.format(num_param))

    return set_gpu(network, opts.gpu_ids)