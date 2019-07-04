import torch
import torch.nn as nn
import torch.nn.functional as F

class unet_block(nn.Module):

    def __init__(self, outer_nc, inner_nc, K=3,dropout=0.5, submodule=None, outermost=False):
        super(unet_block,self).__init__()
        self.outermost = outermost
        self.sub = submodule

        P = int( (K-1)/2 )

        model = [nn.Conv2d(outer_nc, outer_nc, kernel_size=K, stride=1, padding=P),
                 nn.LeakyReLU(0.2,True),
                 nn.Conv2d(outer_nc, outer_nc, kernel_size=K, stride=1, padding=P),
                 nn.LeakyReLU(0.2,True)]
        self.input = nn.Sequential(*model)

        model = [nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2,True)]
        self.down = nn.Sequential(*model)

        if submodule is None:
            model = [nn.Conv2d(inner_nc, inner_nc, kernel_size=K, stride=1, padding=P),
                     nn.LeakyReLU(0.2,True)]
            self.sub = nn.Sequential(*model)

        model = [nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1),
                 nn.ReLU(True),
                 nn.Dropout(dropout)]
        self.up = nn.Sequential(*model)

        model = [nn.Conv2d(2*outer_nc, outer_nc, kernel_size=K, stride=1, padding=P),
                 nn.LeakyReLU(0.2,True),
                 nn.Conv2d(outer_nc, outer_nc, kernel_size=K, stride=1, padding=P),
                 nn.ReLU(True)]
        self.output = nn.Sequential(*model)

    def forward(self, x):

            x1  = self.input(x)
            x2  = self.down(x1)
            x3  = self.sub(x2)
            x4  = self.up(x3)
            x5  = torch.cat([x1,x4],dim=1)
            x6  = self.output(x5)
            return x6

class unet(nn.Module):

    def __init__(self, Cin, Cout, nc, nblocks, K, dropout):
        super(unet, self).__init__()

        inconv  = nn.Conv2d(Cin, nc, kernel_size=1, stride=1, padding=0)
        self.input = nn.Sequential(inconv)

        submodule = None

        for i in range(nblocks-1):
            submodule =  unet_block(nc, nc, K, dropout, submodule=submodule, outermost=False )
        self.ublock = unet_block(nc, nc, K, dropout, submodule=submodule, outermost=True )

        outconv  = nn.Conv2d(nc, Cout, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sequential(outconv)

    def forward(self,x):

        x = self.input(x)
        x = self.ublock(x)
        x = self.output(x)
        x = F.sigmoid(x)
        return x


class encoder(nn.Module):

    def __init__(self, Cin, nc=16, nlevels=4):
        super(encoder, self).__init__()

        K    = 4
        P    = int((K-2)/2)
        cin  = Cin
        cout = nc

        model= []
        for i in range(nlevels):
            model += [nn.Conv2d(cin, cout, K, stride=2, padding=P),
                      nn.LeakyReLU(0.2,True)]
            cin = cout
            cout*=2

        model += [nn.Conv2d(cin, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self,x):

        x = self.model(x)
        x = F.sigmoid(x)

        return x

class GAN(nn.Module):

    def __init__(self, cin, cout, nc, nblocks, K, dropout):
        super(GAN, self).__init__()

        self.g = unet(cin, cout, nc, nblocks, K, dropout)
        self.d = encoder(cout, nc=16, nlevels=4)

        self.g.apply(self.init_weights)
        self.d.apply(self.init_weights)

    def init_weights(self,m):
        if type(m)==nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.uniform_(m.bias,-0.1,0.1)

    def forward(self,x):
        x = self.g(x)
        x = self.d(x)
        return x
