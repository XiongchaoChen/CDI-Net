import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import AverageMeter, get_scheduler, get_gan_loss, psnr, get_nonlinearity
from networks.SE import *
import pdb

'''
U-Net
'''
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=4, wf=6, padding=True,
                 norm='None', up_mode='upconv', residual=False, dropout=False):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding  # True, 1
        self.depth = depth  # 3 or 4
        self.residual = residual  # False
        self.dropout = dropout

        prev_channels = in_channels  # ic (in_channels)

        self.down_path = nn.ModuleList()   # list for modules
        for i in range(depth): # (1, 2^6, 2^7, 2^8, 2^9)  i =0, 1, 2, 3;  | 16, 8, 4, 2
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf+i), padding, norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):  # 2^9, 2^8, 2^7, 2^6 | i = 2, 1, 0
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf+i), up_mode, padding, norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv3d(prev_channels, out_channels, kernel_size=1)  # 2^6 to 1

    def forward(self, x, opts_drop):
        input_ = x
        blocks = []
        p_set = [0, 0, 0, 0]  # probability of zero for dropout
        for i, down in enumerate(self.down_path):  # i = 0, 1, 2, 3
            x = down(x)

            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool3d(x, 2)  # Average pooling here, kernel_size = 2
                x = F.dropout(x, p=p_set[i])

        # Dropout, function.py at testing phase; avoid overfitting
        if self.dropout & opts_drop:
            x = F.dropout(x, p=0.3)

        p_set = [0, 0, 0, 0]
        for i, up in enumerate(self.up_path):  # i = 0, 1, 2
            x = up(x, blocks[-i-1], dropout=p_set[i])

        if self.residual:
            out = input_[:, [0], :, :, :] + self.last(x)  # choose the fisrt channel to residue, while keep the shape
        else:
            out = self.last(x)

        return out     # size = [batch_size, channel, 32, 32, 32]


class UNetConvBlock(nn.Module):  # "Conv3D (+ BN) + ReLU" + "Conv3d (+ BN) + ReLU"
    def __init__(self, in_size, out_size, padding, norm):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv3d(in_size, out_size, kernel_size=3, padding=int(padding)))  # pad = 1
        if norm == 'BN':
            block.append(nn.BatchNorm3d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm3d(out_size))
        block.append(nn.ReLU())

        block.append(nn.Conv3d(out_size, out_size, kernel_size=3, padding=int(padding)))
        if norm == 'BN':
            block.append(nn.BatchNorm3d(out_size))
        elif norm == 'IN':
            block.append(nn.InstanceNorm3d(out_size))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block) # list to module sequential

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv3d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width, layer_depth = layer.size()  # 32,32,32
        diff_y = (layer_height - target_size[0]) // 2  # floor division
        diff_x = (layer_width - target_size[1]) // 2
        diff_z = (layer_depth - target_size[2]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1]), diff_z:(diff_z + target_size[2])]

    def forward(self, x, bridge, dropout):
        up = self.up(x)

        crop1 = self.center_crop(bridge, up.shape[2:])

        out = torch.cat([up, crop1], 1)
        out = F.dropout(out, p=dropout)

        out = self.conv_block(out)


        return out


# CNN Discriminator for GAN
class Dis(nn.Module):
    def __init__(self, input_dim, n_layer=3, norm='None', sn=False):
        super(Dis, self).__init__()
        ch = 64

        model = []  # List
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=4, stride=2, padding=1, norm=norm, sn=sn)]
        for i in range(n_layer-1):  # 0, 1
            model += [LeakyReLUConv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1, norm=norm, sn=sn)]
            ch *= 2

        if sn:
            pass
        else:
            model += [nn.Conv2d(ch, 1, kernel_size=1, stride=1, padding=0)] # The output channels = 1, but size != 1x1

        # model += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)  # '*': change a list to parameters

    def forward(self, input_):
        return self.model(input_)  # [2,1,5,5] output size


class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]  # Boundary reflection padding
        if sn:
            pass
        else:
            model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]

        if norm == 'IN':
            model += [nn.InstanceNorm2d(n_out, affine=False)]
        elif norm == 'BN':
            model += [nn.BatchNorm2d(n_out)]

        model += [nn.LeakyReLU(0.2, inplace=True)]

        self.model = nn.Sequential(*model)

        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


'''
Residual Dense Network
'''
class RDN(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, n_blocks=10, dropout=None):
        super(RDN, self).__init__()
        # F-1
        self.conv1 = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1, bias=True)
        self.drop1 = nn.Dropout(p=dropout)
        # RDBs 3
        self.RDBs = nn.ModuleList([RDB(n_filters, n_denselayer, growth_rate) for _ in range(n_blocks)])
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv3d(n_filters*n_blocks, n_filters, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1, bias=True)
        self.drop2 = nn.Dropout(p=dropout)
        # conv
        self.conv3 = nn.Conv3d(n_filters, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        F_  = self.conv1(x)
        F_0 = self.conv2(F_)
        # F_0 = self.drop1(F_0)

        features = []
        x = F_0
        for RDB_ in self.RDBs:
            y = RDB_(x)
            features.append(y)
            x = y
        FF = torch.cat(features, 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        # FDF = self.drop2(FDF)

        output = self.conv3(FDF)

        return output


# Dense Block
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, norm='None'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.norm = norm
        self.bn = nn.BatchNorm3d(growthRate)


    def forward(self, x):
        out = self.conv(x)
        if self.norm == 'BN':
            out = self.bn(out)
        out = F.relu(out)

        out = torch.cat((x, out), 1)
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm='None'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, norm=norm))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)

        out = self.conv_1x1(out)

        out = out + x # Residual
        return out


'''
Squeeze and Excite Residual Dense UNet
'''
class SERDUNet(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32):
        super(SERDUNet, self).__init__()

        self.conv1 = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate)
        self.SE1 = ChannelSELayer3D(n_filters*1)

        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate)
        self.SE2 = ChannelSELayer3D(n_filters*1)

        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate)
        self.SE3 = ChannelSELayer3D(n_filters*1)

        # decode
        self.up3 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.RDB4 = RDB(n_filters*1, n_denselayer, growth_rate)

        self.SE4 = ChannelSELayer3D(n_filters*1)

        self.up4 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.RDB5 = RDB(n_filters*1, n_denselayer, growth_rate)

        self.conv2 = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)

        # encode
        RDB1 = self.RDB1(x)
        SE1 = self.SE1(RDB1)
        x = F.avg_pool3d(SE1, 2)

        RDB2 = self.RDB2(x)
        SE2 = self.SE2(RDB2)
        x = F.avg_pool3d(SE2, 2)

        RDB3 = self.RDB3(x)
        SE3 = self.SE3(RDB3)

        # decode
        up3 = self.up3(SE3)
        RDB4 = self.RDB4(up3 + SE2)
        SE4 = self.SE4(RDB4)

        up4 = self.up4(SE4)
        RDB5 = self.RDB5(up4 + SE1)

        output = self.conv2(RDB5)
        return output


'''
spatial-channel Squeeze and Excite Residual Dense UNet (depth = 4)
'''
class scSERDUNet(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, norm='None', dropout=False):
        super(scSERDUNet, self).__init__()
        # Channel-wise weight self-recalibration
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(n_channels, n_channels*2, bias=True)
        self.fc2 = nn.Linear(n_channels*2, n_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Input layers
        self.dropout = dropout
        self.conv_in = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE3 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB4 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE4 = ChannelSpatialSELayer3D(n_filters * 1, norm='None')

        # decode
        self.up3 = nn.ConvTranspose3d(n_filters * 1, n_filters * 1, kernel_size=2, stride=2)
        self.fuse_up3 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up3 = RDB(n_filters * 1, n_denselayer, growth_rate, norm)
        self.SE_up3 = ChannelSpatialSELayer3D(n_filters * 1, norm='None')

        self.up2 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up2 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.up1 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up1 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.conv_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x, opts_drop):
        # Channel-wise Self-recalibration
        batch_size, num_channels, D, H, W = x.size()
        squeeze_tensor = self.avg_pool(x)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        output_tensor = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        # encode
        down1 = self.conv_in(output_tensor)
        RDB1 = self.RDB1(down1)
        SE1 = self.SE1(RDB1)

        down2 = F.avg_pool3d(SE1, 2)
        RDB2 = self.RDB2(down2)
        SE2 = self.SE2(RDB2)

        down3 = F.avg_pool3d(SE2, 2)
        RDB3 = self.RDB3(down3)
        SE3 = self.SE3(RDB3)

        down4 = F.avg_pool3d(SE3, 2)
        RDB4 = self.RDB4(down4)
        SE4 = self.SE4(RDB4)

        # Dropout, function.py at testing phase; avoid overfitting
        if self.dropout & opts_drop:
            SE4 = F.dropout(SE4, p=0.3)

        # decode
        up3 = self.up3(SE4) # ([2, 64, 18, 18, 10])
        RDB_up3 = self.RDB_up3(self.fuse_up3(torch.cat((up3, SE3), 1)))
        SE_up3 = self.SE_up3(RDB_up3)

        up2 = self.up2(SE_up3)
        RDB_up2 = self.RDB_up2(self.fuse_up2(torch.cat((up2, SE2), 1)))
        SE_up2 = self.SE_up2(RDB_up2)

        up1 = self.up1(SE_up2)
        RDB_up1 = self.RDB_up1(self.fuse_up1(torch.cat((up1, SE1), 1)))
        SE_up1 = self.SE_up1(RDB_up1)

        output = self.conv_out(SE_up1)
        return output




'''
spatial-channel Squeeze and Excite Residual Dense UNet (depth = 3)
'''
class scSERDUNet3(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, norm='None', dropout=False):
        super(scSERDUNet3, self).__init__()
        # Channel-wise weight self-recalibration
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(n_channels, n_channels * 2, bias=True)
        self.fc2 = nn.Linear(n_channels * 2, n_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Input layer
        self.dropout = dropout
        self.conv_in = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE3 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        # decode
        self.up2 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up2 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.up1 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.fuse_up1 = nn.Conv3d(n_filters * 2, n_filters, kernel_size=3, padding=1, bias=True)
        self.RDB_up1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE_up1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.conv_out = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x, opts_drop):
        # Channel-wise Self-recalibration
        batch_size, num_channels, D, H, W = x.size()
        squeeze_tensor = self.avg_pool(x)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        output_tensor = torch.mul(x, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        # encode
        down1 = self.conv_in(output_tensor)
        RDB1 = self.RDB1(down1)
        SE1 = self.SE1(RDB1)

        down2 = F.avg_pool3d(SE1, 2)
        RDB2 = self.RDB2(down2)
        SE2 = self.SE2(RDB2)

        down3 = F.avg_pool3d(SE2, 2)
        RDB3 = self.RDB3(down3)
        SE3 = self.SE3(RDB3)

        # Dropout, function.py at testing phase; avoid overfitting
        if self.dropout & opts_drop:
            SE3 = F.dropout(SE3, p=0.3)

        # decode
        up2 = self.up2(SE3)
        RDB_up2 = self.RDB_up2(self.fuse_up2(torch.cat((up2, SE2), 1)))
        SE_up2 = self.SE_up2(RDB_up2)

        up1 = self.up1(SE_up2)
        RDB_up1 = self.RDB_up1(self.fuse_up1(torch.cat((up1, SE1), 1)))
        SE_up1 = self.SE_up1(RDB_up1)

        output = self.conv_out(SE_up1)

        return output


'''
Dense UNet (remove residual and SE in scSERDUNet)
'''
class DUNet(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, norm='None', dropout=False):
        super(DUNet, self).__init__()

        self.conv1 = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)
        self.norm = norm
        self.dropout = dropout

        self.bn =  nn.BatchNorm3d(n_filters)

        # encode
        self.DB1 = DB_o(n_filters*1, n_denselayer, growth_rate, norm)
        self.DB2 = DB_o(n_filters*1, n_denselayer, growth_rate, norm)
        self.DB3 = DB_o(n_filters*1, n_denselayer, growth_rate, norm)

        # decode
        self.up3 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.DB4 = DB_o(n_filters*1, n_denselayer, growth_rate, norm)

        self.up4 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.DB5 = DB_o(n_filters*1, n_denselayer, growth_rate, norm)

        self.conv2 = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x, opts_drop):
        x = self.conv1(x)

        # encode
        DB1 = self.DB1(x)
        x = F.avg_pool3d(DB1, 2)

        DB2 = self.DB2(x)
        x = F.avg_pool3d(DB2, 2)

        DB3 = self.DB3(x)

        # Dropout, function.py at testing phase; avoid overfitting
        if self.dropout & opts_drop:
            DB3 = F.dropout(DB3, p=0.3)

        # decode
        up3 = self.up3(DB3)

        DB4 = self.DB4(up3 + DB2)  # Here is addition; For Residual Block

        up4 = self.up4(DB4)

        DB5 = self.DB5(up4 + DB1)

        output = self.conv2(DB5)

        return output


# dense block (DB)
class DB_o(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm='None'):
        super(DB_o, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, norm=norm))
            nChannels_ += growthRate

        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)

        return out


'''
Dense UNet (The One Hundred Layers Tiramisu)
'''
class DenseUNet(nn.Module):
    def __init__(self, n_channels=1, n_filters=32, n_denselayer=6, growth_rate=32, dropout=False):  # suppose n_filters = growrate
        super(DenseUNet, self).__init__()

        self.conv1 = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)
        self.dropout = dropout

        # encode
        self.DB1 = DB(n_filters*1, n_denselayer, growth_rate)
        self.DB2 = DB(n_filters*n_denselayer + n_filters*1, n_denselayer, growth_rate)
        self.DB3 = DB(n_filters*n_denselayer + (n_filters*n_denselayer + n_filters*1), n_denselayer, growth_rate)

        # decode
        self.up3 = nn.ConvTranspose3d(n_filters*n_denselayer + (n_filters*n_denselayer + (n_filters*n_denselayer + n_filters*1)), 
                                      n_filters*n_denselayer + (n_filters*n_denselayer + n_filters*1), 
                                      kernel_size=2, stride=2)
        self.DB4 = DB((n_filters*n_denselayer + (n_filters*n_denselayer + n_filters*1))*2, 
                       n_denselayer, growth_rate)

        self.up4 = nn.ConvTranspose3d(n_filters*n_denselayer + (n_filters*n_denselayer + (n_filters*n_denselayer + n_filters*1))*2, 
                                      n_filters*n_denselayer + n_filters*1, 
                                      kernel_size=2, stride=2)
        self.DB5 = DB((n_filters*n_denselayer + n_filters*1)*2, 
                      n_denselayer, growth_rate)

        self.conv2 = nn.Conv3d(n_filters*n_denselayer + (n_filters*n_denselayer + n_filters*1)*2, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x, opts_drop):
        x = self.conv1(x)

        # encode
        print(x.size())
        DB1 = self.DB1(x)
        print(DB1.size())
        x = F.avg_pool3d(DB1, 2)

        DB2 = self.DB2(x)
        x = F.avg_pool3d(DB2, 2)

        DB3 = self.DB3(x)

        # Dropout, function.py at testing phase; avoid overfitting
        if self.dropout & opts_drop:
            DB3 = F.dropout(DB3, p=0.3)

        # decode
        up3 = self.up3(DB3)
        DB4 = self.DB4(torch.cat([up3,DB2],1))

        up4 = self.up4(DB4)
        DB5 = self.DB5(torch.cat([up4,DB1],1))

        output = self.conv2(DB5)

        return output


# dense block (DB)
class DB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(DB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)

    def forward(self, x):
        out = self.dense_layers(x)
        return out


'''
Capsule UNet
'''
class CapUNet(nn.Module):
    def __init__(self, in_channels=1, nonlinearity='sqaush', dynamic_routing='local'):
        super(CapUNet, self).__init__()
        self.ch = 64
        self.leakyrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=self.ch, kernel_size=5, padding=2, stride=1)

        self.convcaps1 = convolutionalCapsule(in_capsules=1, out_capsules=2, in_channels=self.ch, out_channels=self.ch,
                                              stride=2, padding=1, kernel=3,
                                              nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.convcaps2 = convolutionalCapsule(in_capsules=2, out_capsules=4, in_channels=self.ch, out_channels=self.ch,
                                              stride=1, padding=1, kernel=3,
                                              nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.convcaps3 = convolutionalCapsule(in_capsules=4, out_capsules=4, in_channels=self.ch, out_channels=self.ch * 2,
                                              stride=2, padding=1, kernel=3,
                                              nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.convcaps4 = convolutionalCapsule(in_capsules=4, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                              stride=1, padding=1, kernel=3,
                                              nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.convcaps5 = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 4,
                                              stride=2, padding=1, kernel=3,
                                              nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.convcaps6 = convolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 4, out_channels=self.ch * 2,
                                              stride=1, padding=1, kernel=3,
                                              nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.convcaps7 = convolutionalCapsule(in_capsules=16, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                              stride=1, padding=1, kernel=3,
                                              nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)

        self.deconvcaps1 = deconvolutionalCapsule(in_capsules=8, out_capsules=8, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                                  stride=2, padding=0, kernel=2,
                                                  nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.convcaps8 = convolutionalCapsule(in_capsules=16, out_capsules=4, in_channels=self.ch * 2, out_channels=self.ch * 2,
                                              stride=1, padding=1, kernel=3,
                                              nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.deconvcaps2 = deconvolutionalCapsule(in_capsules=4, out_capsules=4, in_channels=self.ch * 2, out_channels=self.ch,
                                                  stride=2, padding=0, kernel=2,
                                                  nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.convcaps9 = convolutionalCapsule(in_capsules=8, out_capsules=4, in_channels=self.ch, out_channels=self.ch,
                                              stride=1, padding=1, kernel=3,
                                              nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.deconvcaps3 = deconvolutionalCapsule(in_capsules=4, out_capsules=2, in_channels=self.ch, out_channels=self.ch,
                                                  stride=2, padding=0, kernel=2,
                                                  nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)
        self.convcaps10 = convolutionalCapsule(in_capsules=3, out_capsules=1, in_channels=self.ch, out_channels=self.ch,
                                               stride=1, padding=0, kernel=1,
                                               nonlinearity=nonlinearity, dynamic_routing=dynamic_routing)

        self.conv2 = nn.Conv3d(in_channels=self.ch, out_channels=1, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        batch_size = x.size(0)

        # encode
        x_1 = F.relu(self.conv1(x))
        x_1 = x_1.view(batch_size, 1, self.ch, x_1.size(2), x_1.size(3), x_1.size(4))

        x = self.convcaps1(x_1)
        x_2 = self.convcaps2(x)

        x = self.convcaps3(x_2)
        x_3 = self.convcaps4(x)

        x = self.convcaps5(x_3)
        x = self.convcaps6(x)

        # decode
        x = self.deconvcaps1(x)

        x = torch.cat([x, x_3], dim=1)
        x = self.convcaps8(x)

        x = self.deconvcaps2(x)

        x = torch.cat([x, x_2], dim=1)
        x = self.convcaps9(x)

        x = self.deconvcaps3(x)

        x = torch.cat([x, x_1], dim=1)
        x = self.convcaps10(x)

        x_out = x.view(batch_size, self.ch, x.size(3), x.size(4), x.size(5))
        out = self.conv2(x_out)

        return out


class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


if __name__ == '__main__':
    pass

