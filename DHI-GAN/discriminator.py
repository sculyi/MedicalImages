'''
author: lyi
date 20210129
desc to predict the real or fake data


'''


from torch import nn
#from utils import FReLU


class TI_Dis(nn.Module):

    def __init__(self, input_dim=3, hidden_dim = 64):

        super(TI_Dis,self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Conv2d(self.input_dim, self.hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # the following two blocks is designed for TMI_TI
            nn.Conv2d(self.hidden_dim * 4, self.hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.hidden_dim * 4, self.hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),


            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.hidden_dim * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )



    def forward(self, input):
        output = self.network(input)

        return output.view(-1, 1).squeeze(1)


