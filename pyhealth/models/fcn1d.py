import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN1D(nn.Module):

    def __init__(self, n_class=4, in_channels = 1, kernel_size=3, inplanes = 64):
        super(FCN1D, self).__init__()

        """
        Implement a Fully Convolutional Network for semantic segmentation
        as defined in the paper https://arxiv.org/pdf/1411.4038

        Args:
          n_classes:    The number of possible classes for the output prediction

          in_channels:  The number of channels in the input. For instance,
                        a 12-lead ecg would have 12

          kernel_size:  The size of the kernel to be used in all convolutional
                        layers except for the final one
          
          inplanes:     The number of output channels for the first 
                        convolutional block. Subsequent blocks will use
                        inplanes *2, inplanes*4, inplanes*8 respectively
        """

        conv_padding = int(kernel_size / 2)
        l2 = 2* inplanes
        l3 = 4* inplanes
        self.features_123 = nn.Sequential(
            # conv1
            
            nn.Conv1d(in_channels, inplanes, kernel_size, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv1d(inplanes, inplanes, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True),  # 1/2

            # conv2

            nn.Conv1d(inplanes, l2, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(l2, l2, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True),  # 1/4

            # conv3

            nn.Conv1d(l2, l3, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(l3, l3, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(l3, l3, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True),  # 1/8
        )
        l4 = 8*inplanes
        self.features_4 = nn.Sequential(
            # conv4

            nn.Conv1d(l3, l4, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(l4, l4, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(l4, l4, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True),  # 1/16
        )
        self.features_5 = nn.Sequential(
            # conv5 features
            nn.Conv1d(l4, l4, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(l4, l4, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(l4, l4, kernel_size, padding=conv_padding),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=2, ceil_mode=True),  # 1/32
        )
        self.classifier = nn.Sequential(
            # fc6
            nn.Conv1d(l4, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout1d(),

            # fc7
            nn.Conv1d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout1d(),

            # score_fr
            nn.Conv1d(4096, n_class, 1),
        )
        self.score_feat3 = nn.Conv1d(l3, n_class, 1)
        self.score_feat4 = nn.Conv1d(l4, n_class, 1)
        self.upscore = nn.ConvTranspose1d(n_class, n_class, 16, stride=8,
                                              bias=False)
        self.upscore_4 = nn.ConvTranspose1d(n_class, n_class, 4, stride=2,
                                              bias=False)
        self.upscore_5 = nn.ConvTranspose1d(n_class, n_class, 4, stride=2,
                                              bias=False)

    def forward(self, x):
        """
        Input shape (Batch, channels, Sequence Length)
        """
        feat3 = self.features_123(x)  #1/8
        feat4 = self.features_4(feat3)  #1/16
        feat5 = self.features_5(feat4)  #1/32

        score5 = self.classifier(feat5)
        upscore5 = self.upscore_5(score5)
        score4 = self.score_feat4(feat4)
        score4 = score4[:, :, 5:5+upscore5.size()[2]].contiguous() #, 5:5+upscore5.size()[3]
        score4 += upscore5

        score3 = self.score_feat3(feat3)
        upscore4 = self.upscore_4(score4)
        score3 = score3[:, :, 9:9+upscore4.size()[2]].contiguous() #, 9:9+upscore4.size()[3]
        score3 += upscore4
        h = self.upscore(score3)
        h = h[:, :, 28:28+x.size()[2]].contiguous() #, 28:28+x.size()[3]

        return h
    

if __name__ == "__main__":
    
  """ simple test """
  n_class = 4
  n_samples = 16
  n_channels=3
  seq_len = 5000

  model = FCN1D(n_class=n_class, in_channels = n_channels, kernel_size=3, inplanes = 64)

  test_x = torch.randn((n_samples, n_channels, seq_len))

  test_y = model(test_x)

  print(test_y.shape)

  assert test_y.shape[0] == n_samples

  assert test_y.shape[1] == n_class

  assert test_y.shape[-1] == seq_len
