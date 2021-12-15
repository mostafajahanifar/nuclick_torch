""" Full assembly of the parts to form the complete network """

from .layers import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.net_name = 'UNet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


#NuClick NN

class NuClick_NN(nn.Module):


    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.net_name = 'NuClick'

        self.n_channels = n_channels
        self.n_classes = n_classes

        #-------------Conv_Bn_Relu blocks------------
        self.conv_block_1 = nn.Sequential(
            Conv_Bn_Relu(in_channels=self.n_channels, out_channels=64, kernelSize=7),
            Conv_Bn_Relu(in_channels=64, out_channels=32, kernelSize=5),
            Conv_Bn_Relu(in_channels=32, out_channels=32, kernelSize=3)
        )

        self.conv_block_2 = nn.Sequential(
            Conv_Bn_Relu(in_channels=64, out_channels=64),
            Conv_Bn_Relu(in_channels=64, out_channels=32),
            Conv_Bn_Relu(in_channels=32, out_channels=32)
        )

        self.conv_block_3 = Conv_Bn_Relu(in_channels=32, out_channels=self.n_classes,
            kernelSize=(1,1), actv=None, useBias=True, doBatchNorm=False)

        #-------------Residual_Conv blocks------------
        self.residual_block_1 = nn.Sequential(
            Residual_Conv(in_channels=32, out_channels=64),
            Residual_Conv(in_channels=64, out_channels=64)
        )

        self.residual_block_2 = Residual_Conv(in_channels=64, out_channels=128)

        self.residual_block_3 = Residual_Conv(in_channels=128, out_channels=128)

        self.residual_block_4 = nn.Sequential(
            Residual_Conv(in_channels=128, out_channels=256),
            Residual_Conv(in_channels=256, out_channels=256),
            Residual_Conv(in_channels=256, out_channels=256)
        )

        self.residual_block_5 = nn.Sequential(
            Residual_Conv(in_channels=256, out_channels=512),
            Residual_Conv(in_channels=512, out_channels=512),
            Residual_Conv(in_channels=512, out_channels=512)
        )

        self.residual_block_6 = nn.Sequential(
            Residual_Conv(in_channels=512, out_channels=1024),
            Residual_Conv(in_channels=1024, out_channels=1024)
        )

        self.residual_block_7 = nn.Sequential(
            Residual_Conv(in_channels=1024, out_channels=512),
            Residual_Conv(in_channels=512, out_channels=256)
        )

        self.residual_block_8 = Residual_Conv(in_channels=512, out_channels=256)

        self.residual_block_9 = Residual_Conv(in_channels=256, out_channels=256)

        self.residual_block_10 = nn.Sequential(
            Residual_Conv(in_channels=256, out_channels=128),
            Residual_Conv(in_channels=128, out_channels=128)
        )

        self.residual_block_11 = Residual_Conv(in_channels=128, out_channels=64)

        self.residual_block_12 = Residual_Conv(in_channels=64, out_channels=64)


        #-------------Multiscale_Conv_Block blocks------------
        self.multiscale_block_1 = Multiscale_Conv_Block(in_channels=128, out_channels=32,
            kernelSizes=[3,3,5,5], dilatationRates=[1,3,3,6], isDense=False
        )

        self.multiscale_block_2 = Multiscale_Conv_Block(in_channels=256, out_channels=64,
            kernelSizes=[3,3,5,5], dilatationRates=[1,3,2,3], isDense=False
        )

        self.multiscale_block_3 = Multiscale_Conv_Block(in_channels=64, out_channels=16,
            kernelSizes=[3,3,5,7], dilatationRates=[1,3,2,6], isDense=False
        )
            
        #-------------MaxPool2d blocks------------
        self.pool_block_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_2 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_3 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_4 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool_block_5 = nn.MaxPool2d(kernel_size=(2,2))

        #-------------ConvTranspose2d blocks------------
        self.conv_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
            kernel_size=2, stride=(2,2),
        )

        self.conv_transpose_5 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
            kernel_size=2, stride=(2,2),
        )

    def forward(self, input):

        conv1 = self.conv_block_1(input)    #conv1: 32 channels
        pool1 = self.pool_block_1(conv1)     #poo1: 32 channels

        conv2 = self.residual_block_1(pool1)    #conv2: 64 channels
        pool2 = self.pool_block_2(conv2)    #pool2: 64 channels        

        conv3 = self.residual_block_2(pool2)   #conv3: 128 channels
        conv3 = self.multiscale_block_1(conv3)  #conv3: 128 channels
        conv3 = self.residual_block_3(conv3)    #conv3: 128 channels    
        pool3 = self.pool_block_3(conv3)    #pool3: 128 channels

        conv4 = self.residual_block_4(pool3)    #conv4: 256 channels
        pool4 = self.pool_block_4(conv4)    #pool4: 512 channels

        conv5 = self.residual_block_5(pool4)    #conv5: 512 channels
        pool5 = self.pool_block_5(conv5)    #pool5: 512  channels

        conv51 = self.residual_block_6(pool5) #conv51: 1024 channels

        up61 = torch.cat([self.conv_transpose_1(conv51),conv5], dim=1)  #up61: 1024 channels
        conv61 = self.residual_block_7(up61)    #conv61: 256 channels
        
        up6 = torch.cat([self.conv_transpose_2(conv61), conv4], dim=1)  #up6: 512 channels
        conv6 = self.residual_block_8(up6) #conv6: 256 channels
        conv6 = self.multiscale_block_2(conv6)  #conv6: 256 channels
        conv6 = self.residual_block_9(conv6)    #conv6: 256 channels

        up7 = torch.cat([self.conv_transpose_3(conv6), conv3], dim=1)   #up7: 256 channels
        conv7 = self.residual_block_10(up7)     #conv7: 128 channels

        up8 = torch.cat([self.conv_transpose_4(conv7), conv2], dim=1)   #up8: 128 channels
        conv8 = self.residual_block_11(up8)     #conv8: 64 channels
        conv8 = self.multiscale_block_3(conv8)  #conv8: 64 channels
        conv8 = self.residual_block_12(conv8)   #conv8: 64 channels

        up9 = torch.cat([self.conv_transpose_5(conv8), conv1], dim=1)   #up9: 64 channels
        conv9 = self.conv_block_2(up9)  #conv9: 32 channels

        conv10 = self.conv_block_3(conv9)   #conv10: out_channels
        
        return conv10

