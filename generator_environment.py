import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
# from torchvision.utils import save_image

class DQN(nn.Module):

    def __init__(self, h, w, outputs, device):
        super(DQN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 16, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 1, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print("blah")
        # print(x.size())
        # print(x.size(0))
        # print(x.view(x.size(0), -1))
        # print(x.view(x.size(0), -1).shape)
        # print(self.head)
        return self.head(x.view(x.size(0), -1))
        # return self.head(x.view(x.size(0), -1))
#
# class MulticlassClassification(nn.Module):
#     def __init__(self, num_feature, num_class):
#         super(MulticlassClassification, self).__init__()
#
#         self.layer_1 = nn.Linear(num_feature, 512)
#         self.layer_2 = nn.Linear(512, 128)
#         self.layer_3 = nn.Linear(128, 64)
#         self.layer_out = nn.Linear(64, num_class)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.2)
#         self.batchnorm1 = nn.BatchNorm1d(512)
#         self.batchnorm2 = nn.BatchNorm1d(128)
#         self.batchnorm3 = nn.BatchNorm1d(64)
#
#     def forward(self, x):
#         x = self.layer_1(x)
#         x = self.batchnorm1(x)
#         x = self.relu(x)
#
#         x = self.layer_2(x)
#         x = self.batchnorm2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         x = self.layer_3(x)
#         x = self.batchnorm3(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         x = self.layer_out(x)
#         return x
#

#
# class GeneratorEnvironment:
#     def __init__(self, corpus, device):
#         self.uc = torch.tensor([[[ 2.8699e-01, -1.3061e-01,  8.6316e-03,  1.3696e-01,  5.2970e-01,
#         -5.0282e-01,  7.8231e-02,  7.8714e-01, -5.7608e-01, -2.8497e-01,
#         -7.4669e-02, -1.0538e-01,  3.2667e-01, -2.2295e-01, -5.0512e-01,
#         -7.7677e-02, -1.6824e-01,  4.1058e-01, -3.7778e-01,  4.5622e-01,
#         -3.5574e-01,  2.6545e-01,  3.2107e-02, -1.5508e-01,  1.8140e-01,
#          2.9542e-01, -9.5338e-02,  3.7212e-02, -6.2266e-01, -1.7874e-01,
#          4.6274e-01, -3.7304e-01,  1.4313e-01,  4.7745e-01, -2.0041e-01,
#         -4.9707e-01,  4.3558e-02,  1.6147e-01, -4.2380e-01,  5.0578e-02,
#         -1.6410e-01, -3.2034e-01,  2.5008e-01, -3.1832e-01,  1.5186e-01,
#         -2.0789e-01, -4.4791e-01,  1.8324e-01, -4.8349e-02, -4.9884e-01,
#         -3.2135e-01, -5.4520e-02, -5.9430e-01, -2.2781e-03,  5.3567e-01,
#          2.7617e-01, -2.7955e-01, -2.4671e-01,  2.7637e-01,  4.2538e-02,
#         -4.9209e-01, -3.3102e-01, -5.8562e-02, -3.0317e-01, -1.1896e-01,
#         -3.0265e-01,  1.6411e-01,  2.8470e-01, -2.5686e-01, -5.5965e-01,
#         -2.0371e-01, -2.8656e-01, -2.4610e-01, -3.7548e-01, -6.1157e-01,
#         -3.2826e-01, -5.0560e-01,  3.2929e-01, -4.3152e-02, -4.4335e-01,
#         -3.3025e-01,  7.2476e-01,  7.3303e-02, -2.7141e-03,  4.4829e-02,
#         -2.0232e-01,  1.6289e-01, -3.8574e-01,  2.2319e-01,  6.0061e-01,
#         -3.0956e-01,  3.6498e-01, -1.9308e-01,  1.0114e-01,  2.2660e-01,
#         -2.1611e-01, -2.7655e-01, -1.1151e-01,  1.4243e-01, -9.2815e-03,
#          2.8011e-01, -9.8526e-01,  5.6201e-02,  2.9653e-01,  3.1355e-01,
#          7.2933e-02, -2.1141e-01,  1.6271e-01,  1.0262e-01, -9.1207e-03,
#          2.2504e-01, -2.6987e-02,  2.2139e-01, -1.4566e-01, -2.9635e-01,
#          3.9109e-02, -1.5157e-01, -2.6471e-02,  1.4093e-01, -1.0780e-01,
#         -3.8882e-01, -2.2879e-01, -3.0630e-01,  4.2430e-01,  3.5177e-01,
#          1.7161e-01, -1.9235e-01,  1.0809e-01,  5.8905e-02, -3.1387e-01,
#          4.0762e-01,  1.1046e-01, -1.6646e-02,  4.9697e-02,  1.0476e-01,
#          3.4812e-01,  1.7014e-01,  3.1764e-01,  1.2853e-01, -7.5686e-01,
#         -1.2146e-01,  2.4131e-01, -1.6947e-01,  4.6189e-01,  2.7758e-01,
#         -1.7192e-01, -3.4973e-02,  1.3485e-01, -4.8362e-01,  2.6712e-01,
#         -9.5615e-02,  7.4699e-01, -1.4413e-01, -3.5958e-01, -6.8509e-02,
#         -2.2691e-01, -1.8903e-01,  4.1435e-01,  2.1882e-01,  8.3887e-02,
#          4.6017e-01,  2.2194e-01, -3.2900e-01,  7.6490e-02, -5.5079e-01,
#          3.5410e-01, -2.2028e-01,  2.9926e-01,  2.9454e-01,  9.2131e-02,
#          9.1896e-03, -1.4974e-01,  5.7777e-01,  4.4610e-01,  4.3993e-01,
#         -1.5486e-01, -4.1923e-01,  4.7560e-01, -5.9601e-02,  8.1371e-01,
#         -8.7876e-01,  4.8956e-01,  6.7629e-01, -4.5140e-02, -5.1308e-01,
#         -2.1268e-01, -8.4959e-02, -5.8131e-01,  1.6116e-01, -9.5833e-02,
#         -1.9200e-01,  1.9683e-01, -1.4921e-01, -6.2115e-01, -9.8531e-02,
#         -6.5684e-01, -4.1577e-02,  7.1345e-02,  2.8209e-01, -1.3653e-02,
#          8.3481e-02,  4.4487e-01,  7.0161e-02, -2.3971e-01, -1.3780e-01,
#          1.0490e+00, -1.4356e-01, -2.8440e-01, -5.2677e-01,  3.0325e-01,
#         -6.6890e-01,  4.4759e-01, -2.0509e-02, -4.6442e-01, -4.6426e-01,
#          2.0460e-01, -3.5428e-01, -2.0329e-01,  6.4442e-02, -1.1025e-01,
#          2.6919e-01, -3.3775e-01, -3.7425e-01,  1.6976e-02, -5.2783e-01,
#          3.8417e-01,  6.9101e-02,  3.8555e-01,  2.2729e-01, -2.8298e-01,
#          4.6225e-01, -3.3916e-01, -2.5816e-01,  3.0740e-01,  1.0118e-01,
#          2.1852e-01,  1.8029e-01, -1.7472e-01,  4.8300e-02,  2.2544e-01,
#         -3.4734e-01, -1.8502e-01,  3.2778e-01, -3.7516e-01,  2.4916e-01,
#          5.5979e-01,  8.3131e-02, -2.7640e-01,  3.6365e-01,  2.1308e-01,
#          2.5315e-01, -1.1834e+00,  5.5541e-02, -1.3502e-01, -3.2656e-01,
#         -6.8246e-01, -3.6448e-02, -3.0588e-02, -3.1030e-01, -3.3844e-01,
#          4.5033e-01,  3.8030e-01,  3.3283e-01, -1.8203e-01, -1.4706e-01,
#         -6.4128e-01, -5.5777e-02,  3.4403e-01,  8.2396e-02, -4.2013e-02,
#         -2.1082e-01, -4.7175e-02,  9.5943e-02,  3.6320e-02, -6.8078e-01,
#          6.4128e-02,  1.7556e-01,  3.1849e-01,  1.9858e-01, -4.4841e-01,
#         -4.6788e-01, -1.9665e-01, -7.1055e-01,  4.5634e-01,  5.8528e-01,
#         -9.6617e-02, -1.6418e-01, -1.8096e-01,  1.7546e-02, -7.8262e-01,
#          1.2674e-01,  4.0208e-01,  2.4947e-02, -3.0891e-01,  1.2305e+00,
#         -2.2365e-01,  3.3606e-02,  3.7693e-01, -1.4110e-01, -7.0846e-01,
#          4.0692e-01, -6.1874e-02, -9.5316e-03, -1.2641e-01,  2.2775e-01,
#         -2.1415e-01, -1.9425e-01, -4.2640e-01, -6.9603e+00,  2.2194e-01,
#         -3.9800e-01,  3.9092e-03,  4.0280e-01, -9.0126e-02,  2.7320e-01,
#          3.8863e-01, -4.2080e-01, -6.8265e-02, -2.8653e-02, -2.7225e-02,
#          3.0601e-01,  3.2074e-01, -5.2955e-01,  3.2624e-01,  4.2942e-01,
#         -2.4989e-01, -5.2061e-01,  3.4608e-01,  8.6900e-02, -9.9542e-01,
#          2.8187e-01,  2.7719e-01, -1.2284e-01, -2.8708e-02,  1.6582e-01,
#         -2.9040e-02, -1.0352e-01,  1.2787e-01, -1.4311e-01,  4.3619e-01,
#          4.3046e-01, -5.5023e-02,  2.2400e-02,  1.7485e-01,  1.6629e-01,
#         -5.7904e-02,  4.3649e-01,  1.7797e-01, -4.1757e-02,  3.8085e-03,
#          3.0046e-01, -2.3633e-01,  5.5302e-01, -2.0077e-01,  5.2925e-01,
#          5.0039e-01, -1.9726e-01,  2.0752e-01,  1.6202e-02,  8.1718e-02,
#          7.8762e-01,  3.5425e-02, -4.1314e-01, -8.5976e-02, -9.2967e-03,
#          1.0298e+00, -4.1758e-02, -2.1042e-01,  3.4290e-01, -4.8570e-01,
#          1.7516e-01,  3.4653e-01,  1.0801e-03,  3.4310e-01, -6.8028e-01,
#         -6.2681e-01, -1.5538e-01,  5.0329e-02, -1.6004e-01,  1.5806e-01,
#         -3.5866e-02, -1.6747e+00, -6.0990e-01, -1.3829e-01, -2.2987e-01,
#          4.1932e-01, -7.9315e-02,  8.7949e-02, -2.5882e-01, -1.1615e-01,
#          2.4467e-01,  6.0941e-02,  3.2133e-01,  2.1772e-01,  1.3220e-02,
#         -5.2403e-01, -1.5435e-01, -9.6020e-02,  4.2068e-01,  2.9070e-01,
#          2.1196e-01, -1.0958e-01,  8.8291e-02, -4.4237e-01,  1.9082e-02,
#         -4.1930e-01, -1.8195e-01, -1.3286e-01, -1.4558e-01, -7.8467e-02,
#          6.9000e-03, -2.1418e-01,  5.5206e-02, -3.8651e-01, -4.7944e-01,
#          3.0542e-01, -2.1569e-01,  5.0189e-01, -1.7245e-01, -2.9071e-01,
#          6.2823e-02,  2.7569e-01, -5.4053e-01,  6.5577e-02,  3.8364e-01,
#          6.2603e-01,  7.5622e-02, -2.1055e-01,  4.8702e-01,  4.1627e-01,
#         -9.8866e-03, -3.6597e-01,  4.4420e-01,  9.1638e-02, -1.6765e-01,
#         -1.0078e+00, -1.9952e-01, -3.4179e-01, -1.6425e-01,  2.2729e-01,
#         -1.8496e-01, -1.4211e-01, -4.1657e-01, -2.4966e-01, -1.5576e-01,
#          5.5399e-01, -2.2910e-01,  2.6343e-01,  1.2730e-01,  8.7628e-03,
#         -1.0327e-01, -3.8991e-01,  2.4551e-01, -2.2782e-01,  2.9856e-01,
#         -8.3166e-02, -2.9501e-02,  2.6185e-01, -2.0793e-01, -6.2349e-01,
#         -1.7995e-01,  2.0444e-01, -8.4174e-02,  7.8507e-02,  1.7900e-02,
#         -3.5265e-01,  5.3873e-01, -6.7307e-01, -3.4361e-02, -1.7153e-01,
#         -2.2953e-01,  2.9241e-01,  6.0656e-02,  3.5654e-01,  8.3309e-03,
#         -1.5221e-01, -5.9336e-01,  3.9605e-01,  2.2714e-01, -2.4133e-01,
#          2.9925e-01, -6.4026e-02,  9.4565e-02,  2.5752e-02,  3.3976e-01,
#         -4.0011e-02, -6.6448e-02,  1.3536e-01,  3.9022e-01, -3.3643e-01,
#         -5.5690e-01, -9.5238e-02,  2.4499e-01,  2.1463e-01, -1.9213e-01,
#          3.4095e-01,  2.2370e-01, -8.5643e-02, -2.1294e-01,  2.5633e-01,
#         -2.1285e-01, -2.2150e-01, -2.1565e-01, -1.7814e-01, -2.0867e-01,
#          6.0421e-02,  3.0997e-01,  4.7149e-01, -2.1125e-01, -3.6545e-01,
#         -2.1591e-01,  1.0872e-01, -7.2829e-02,  5.4355e-01,  6.8389e-01,
#          4.5484e-01, -2.7260e-01, -2.7587e-01, -5.3108e-01, -3.0611e-01,
#         -4.9529e-01, -9.1465e-02,  1.3323e-01,  1.4631e-01, -1.9489e-01,
#         -1.4801e-01, -5.1434e-01, -3.0579e-02,  1.8731e-01,  5.1503e-01,
#          5.5846e-02,  1.1132e-01, -2.4691e-01,  5.4796e-02,  8.0343e-02,
#          2.2289e-01, -2.0173e-01,  1.8415e-01, -4.3800e-01, -5.1463e-01,
#         -8.7455e-03, -1.9626e-01,  4.4153e-01,  1.9125e-01, -1.9849e-01,
#          1.2565e-02, -4.4613e-02, -2.4169e-01, -9.6077e-02, -6.8212e-01,
#         -6.4495e-02, -3.5651e-01,  4.5172e-02, -3.5156e-01, -3.4969e-01,
#          6.6269e-01, -9.4356e-02,  3.6469e-01, -2.2566e-01,  1.8458e-02,
#         -1.9426e-01, -5.1162e-01,  2.4071e-01, -4.7241e-01, -4.8642e-01,
#         -1.9505e-01,  3.5609e-01,  5.4559e-01, -3.7198e-01,  6.5193e-01,
#          5.2634e-02,  2.6405e-02, -2.2360e-01,  3.3810e-01,  3.5132e-02,
#          1.4112e-01, -6.6679e-02, -4.3788e-02,  8.6035e-02, -4.3471e-01,
#          9.9257e-02,  2.5438e-01, -2.8164e-01, -5.4871e-02,  3.1955e-01,
#         -4.7375e-01, -1.0366e-01, -6.4371e-02, -6.7950e-01,  4.8438e-01,
#          7.0932e-02, -3.3722e-02,  1.3368e-01,  3.9896e-01, -4.7353e-01,
#         -1.8770e-01,  1.4294e-02,  3.7459e-01,  4.3065e-01,  2.8935e-01,
#         -2.0861e-01,  8.8362e-01, -8.7442e-02,  4.5671e-01,  2.0594e-01,
#          1.7363e-02, -2.3602e-01, -1.3865e-01,  2.9057e-01,  1.5402e-01,
#          3.0283e-01,  1.7748e-01, -1.3677e-01,  4.7039e-01,  5.8701e-02,
#         -8.5705e-02,  4.7866e-01, -1.0962e-01,  3.1133e-03, -5.5992e-01,
#          8.2927e-02,  5.1993e-01,  8.6856e-02,  9.7301e-02,  3.4685e-02,
#         -3.4256e-01, -5.3860e-01,  5.5095e-02,  5.9109e-02,  2.6272e-01,
#          3.9280e-01, -2.8144e-01,  1.6353e-01,  1.3860e-01, -4.0998e-01,
#         -8.5989e-02, -2.3144e-03,  3.0711e-01, -1.5768e-01,  1.3357e-02,
#         -8.9992e-03,  1.4138e-02,  3.2260e-01, -2.6084e-01, -5.1165e-01,
#          6.8886e-03, -3.3519e-01,  3.2208e-01, -6.6072e-02,  2.0930e-01,
#          4.6820e-01,  2.6998e-01,  1.3627e-01,  2.0200e-01, -6.3257e-01,
#          1.2327e-01,  5.9530e-01,  5.0058e-01, -1.4802e-01, -5.7471e-01,
#          2.1097e-02, -2.8760e-02,  3.5423e-01, -3.0772e-01,  3.3448e-01,
#          3.3356e-01, -1.9780e-01,  1.2328e-01,  1.9073e-01,  1.4597e-01,
#          2.0827e-02, -8.0914e-01,  9.1224e-02, -2.8124e-01,  9.2574e-02,
#          2.0324e-01, -2.0550e-01, -2.1067e-01, -1.0136e-01, -1.1435e-01,
#          2.1457e-01,  3.0529e-01,  1.7499e-01, -9.5718e-02, -3.5839e-01,
#          2.1185e-01, -3.4900e-01, -1.6695e-01,  5.8157e-02, -6.8906e-01,
#         -3.2069e-01, -1.7024e-01, -3.8585e-01,  9.6186e-02, -1.4070e-01,
#         -1.9575e-01,  7.2330e-01, -5.3810e-02, -5.1457e-01,  1.9791e-01,
#         -1.3807e-01, -6.1748e-03, -2.5194e-01,  3.7989e-01, -1.7560e-01,
#         -2.9074e-01,  3.7905e-02, -6.4008e-01,  3.2436e-02, -4.7844e-01,
#          1.3602e-01, -6.3730e-01,  5.7572e-02,  1.9942e-01, -2.5498e-01,
#         -4.3850e-02, -7.5470e-01, -5.3969e-01,  2.1883e-01,  5.6351e-01,
#         -7.1851e-02,  1.3687e-01,  1.5572e-02,  2.5807e-01,  2.1313e-01,
#         -2.3847e-01,  4.9660e-01, -3.0258e-01,  6.0770e-01, -2.3119e-01,
#         -1.0493e-01,  1.9885e-01,  2.8531e-01, -6.2832e-02, -3.0579e-01,
#          7.0830e-01, -1.4758e-01, -2.6738e-01, -3.6199e-01,  9.9154e-02,
#          3.5979e-01, -5.2635e-02,  7.7397e-02,  1.7882e-01, -1.9493e-01,
#          3.8088e-02, -3.5828e-01,  4.9735e-02, -2.1624e-01,  1.6715e-03,
#         -8.5087e-02,  1.3702e-01, -3.2520e-01,  2.7179e-02, -2.2317e-01,
#          3.7348e-02, -3.9783e-01, -1.7729e-01, -1.6385e-01, -1.6487e-01,
#          5.7190e-01,  2.5678e-01,  2.2145e-01,  2.6308e-01, -6.0074e-01,
#         -3.8712e-01, -5.1841e-01,  2.2427e-01]]]).to(device)
#         self.grass_green = torch.tensor([[[-1.1272e-01, -4.2158e-01, -5.4724e-02,  1.7511e-01,  2.3005e-01,
#          3.7133e-02, -1.8699e-01,  1.0386e+00, -7.1984e-01, -5.9138e-01,
#          6.0872e-02, -3.7143e-01, -3.1244e-01,  2.1062e-01, -5.8843e-01,
#          7.2982e-01,  3.2796e-01,  3.0826e-01,  3.1759e-02,  6.3549e-01,
#          4.8248e-02, -6.5918e-02, -4.3420e-01, -1.9576e-01,  3.4424e-01,
#          4.5779e-01,  1.2751e-01,  4.1101e-01,  3.7435e-01,  4.6835e-01,
#          1.6429e-01,  1.2245e-01,  1.2399e-01, -1.8338e-01, -2.3387e-01,
#         -3.6701e-01, -4.3599e-01, -7.1388e-02, -8.8454e-01,  4.7744e-01,
#         -8.1029e-01, -3.7892e-01,  5.3191e-01,  5.8453e-02, -4.2986e-02,
#         -3.2189e-01, -8.5888e-02,  1.0207e-01,  1.9045e-01, -1.2399e-01,
#         -1.9046e-01, -1.9680e-01, -6.6974e-01,  6.3950e-02, -2.5769e-01,
#          4.3543e-01, -1.7955e-01,  1.2638e-01,  3.3385e-01,  3.3940e-02,
#         -4.6499e-01, -2.5811e-01, -4.8065e-02,  1.6090e-01, -2.1227e-01,
#          3.9308e-01, -1.4959e-01,  1.3104e-01,  1.6878e-02,  3.6418e-01,
#         -1.5341e-01, -4.0740e-03,  5.6280e-02,  3.3139e-01, -7.0755e-01,
#         -9.8256e-02, -8.4336e-02,  4.5607e-01, -4.8640e-01, -3.7335e-01,
#         -1.8315e-01,  6.6806e-01,  1.4881e-01,  1.4449e-01, -2.0697e-02,
#          3.7833e-02, -2.1829e-01,  1.7548e-01,  9.8210e-02,  9.5063e-02,
#         -6.7286e-02,  1.2093e-01, -1.0567e-01,  1.2575e-01,  2.5427e-01,
#         -2.7207e-01,  1.3680e-01, -3.9013e-01,  2.7505e-01,  1.8207e-01,
#          5.2732e-03,  6.2133e-02, -1.0624e-01,  2.6587e-01, -4.6035e-01,
#         -5.4226e-02, -1.4459e-01, -1.6375e-01,  2.9461e-01,  9.3399e-02,
#         -1.3583e-01,  1.7082e-01,  4.0184e-01,  3.5787e-01, -2.5825e-01,
#          1.0998e+00,  1.4802e-01,  7.8193e-02, -1.4152e-01,  5.2229e-01,
#         -1.7104e-01, -3.1357e-01,  2.3503e-01,  2.6565e-01, -2.3881e-01,
#         -2.2962e-01, -4.8619e-01, -1.4935e-01, -1.0039e-01,  1.3635e-01,
#          3.8913e-01,  1.6611e-01,  2.1053e-02, -9.6143e-02,  1.8026e-01,
#          2.2481e-01,  1.5288e-01, -7.0379e-02, -5.1827e-01,  1.7356e-01,
#         -7.0186e-01,  4.3782e-01, -1.9230e-01,  1.7959e-01, -7.2058e-02,
#         -6.6780e-01, -1.0451e-01, -9.9280e-02,  1.4990e-01, -3.6037e-01,
#          2.2174e-01,  3.5784e-01, -1.4012e-01, -1.4161e-01, -6.2277e-01,
#         -1.6900e-01, -5.0654e-01,  2.9271e-01,  2.9122e-01, -9.1463e-02,
#          6.8196e-03, -3.6687e-01,  5.0026e-01,  1.8899e-01, -1.0203e-01,
#         -2.5002e-01,  2.6899e-02,  2.4129e-01, -1.3515e-01,  9.6403e-02,
#          2.1971e-02, -1.7198e-01,  1.1234e+00, -3.5151e-01,  4.0785e-01,
#         -1.9896e-02, -5.2718e-01, -3.2154e-01, -1.3553e-01,  6.0554e-01,
#         -6.7273e-01, -5.5533e-02, -3.6644e-01, -1.8987e-01,  4.1901e-01,
#          1.0489e-01,  3.7942e-01, -2.8263e-01,  3.3935e-01,  2.8845e-02,
#         -8.4864e-02,  4.4160e-02,  8.2183e-02, -5.8066e-02,  4.2427e-01,
#         -8.4007e-01,  1.1945e-01, -6.5825e-01, -2.8356e-02, -1.1734e-01,
#          2.9480e-01, -2.0501e-01, -1.6065e-02, -3.1638e-01, -4.6293e-01,
#          8.0685e-01, -2.8279e-01, -3.6496e-02, -1.3040e-01,  1.3609e-01,
#          4.3952e-02,  3.7603e-01,  5.5332e-01, -2.5841e-01,  1.0650e-01,
#         -3.4781e-01,  6.7129e-01,  1.6868e-01,  2.5809e-01,  3.2737e-01,
#          2.7550e-01, -3.6059e-01, -3.0383e-01,  6.8676e-01, -1.3022e-01,
#          2.5934e-01,  7.8226e-01,  1.2263e-02,  1.6156e-01, -1.0268e-01,
#         -1.6986e-01,  5.2082e-02,  4.0366e-01, -8.3519e-02, -1.3499e-01,
#         -1.4329e-02,  2.2201e-01,  2.4566e-01,  3.7495e-02,  4.5147e-01,
#         -2.9039e-01,  1.5555e-01,  6.3546e-01,  1.2303e-01,  1.6822e-01,
#         -2.3156e-01,  3.2978e-01, -3.9274e-01,  6.7953e-02, -4.5587e-01,
#         -2.3946e-01, -3.4797e-01, -7.1038e-02, -1.6343e-01, -7.5228e-01,
#          1.7495e-02,  2.1517e-01,  5.7923e-01,  4.5702e-01, -5.0370e-03,
#         -3.2399e-02, -6.9121e-01,  1.8122e-01, -4.0416e-01, -7.5326e-02,
#          3.6137e-01,  2.6551e-01, -1.6032e-01,  2.0040e-01, -2.4840e-01,
#          1.5482e-01, -1.4363e-01,  4.4453e-01,  3.1137e-01, -1.4788e-01,
#         -7.1284e-01,  4.6672e-01,  5.5175e-03, -1.1410e-01, -3.1578e-01,
#         -1.5032e-01,  3.6135e-01, -5.0308e-01, -1.2795e-01,  3.1364e-01,
#         -1.9191e-01,  2.1289e-01, -1.0063e-01,  1.6320e-01, -3.7179e-02,
#         -3.3809e-02,  3.1473e-01,  2.2643e-02, -4.9070e-01,  9.0337e-01,
#         -2.0343e-01, -2.9348e-01, -4.4367e-01, -2.0158e-01, -8.0454e-01,
#         -1.2212e-01, -3.4882e-02,  1.3706e-01, -1.8312e-01, -9.3515e-02,
#         -3.0064e-02, -3.9210e-03, -4.8983e-02, -8.8458e+00, -4.0814e-01,
#          1.7704e-01,  2.0032e-01,  3.4801e-01, -5.3328e-01, -5.2973e-01,
#          3.6427e-01, -1.9809e-01, -4.5938e-01, -2.7293e-01, -2.9121e-01,
#          8.1799e-02,  4.9918e-01, -4.5823e-02, -2.9397e-01,  2.8910e-01,
#         -4.6607e-01, -4.7384e-01,  2.5643e-01, -2.4190e-02, -1.9571e-01,
#          1.0508e-01, -1.3225e-01, -4.1918e-01,  3.9785e-01,  1.1239e-01,
#         -1.6499e-01, -4.3147e-01, -1.6567e-01, -3.0886e-01, -3.7188e-01,
#          5.3323e-01,  5.4217e-01, -5.6593e-01,  1.7482e-01, -3.7584e-01,
#         -1.2295e-01,  5.2019e-01,  7.0921e-01, -6.9715e-02, -2.4575e-01,
#          6.0152e-01,  5.3154e-01,  1.7633e-01, -2.9031e-01,  2.4660e-02,
#         -2.1500e-01,  8.6558e-02,  4.2180e-01,  1.7852e-01,  5.4063e-02,
#          7.0315e-01, -5.4084e-01, -5.6616e-02,  3.3268e-02,  1.0884e+00,
#          6.3027e-01,  4.4974e-01, -1.0002e+00,  9.4973e-02, -2.3050e-01,
#          2.1384e-02,  2.5524e-01, -2.5808e-01, -1.8028e-01, -7.3290e-01,
#         -4.0259e-01, -6.4848e-02,  1.1093e-01, -1.1681e-01, -1.4281e-01,
#         -1.6668e-01, -1.7420e+00, -7.6441e-01, -1.4027e-01, -1.8599e-01,
#         -1.8212e-01,  2.1163e-01, -6.0507e-02, -1.6709e-03, -4.6904e-01,
#          1.8654e-01,  5.6574e-02, -4.8903e-01, -4.9443e-01, -4.8215e-03,
#         -4.8009e-01, -4.0297e-01,  5.8060e-01,  2.4942e-01, -3.7202e-01,
#         -1.2483e-01,  1.7185e-01, -3.2639e-01,  3.8666e-02,  5.1991e-01,
#         -5.8472e-01,  4.0975e-01,  6.0941e-03,  5.3956e-01, -3.2190e-02,
#          2.1582e-01,  4.4079e-01,  5.1690e-02, -2.6976e-01, -4.2387e-01,
#          4.2040e-01,  2.2856e-01,  9.3787e-02,  5.8900e-01, -2.8679e-01,
#          3.2620e-01, -4.6502e-01, -3.1707e-02, -8.4862e-02,  7.3229e-01,
#          2.2303e-01,  2.7578e-01,  1.1820e-01,  2.1433e-01,  3.9588e-01,
#          1.8802e-01, -4.9105e-01, -2.1090e-01,  3.2014e-01, -2.4623e-01,
#         -7.5696e-01, -5.6058e-01,  4.4992e-01, -3.6036e-02, -4.0698e-01,
#         -3.1531e-01, -3.9987e-02, -3.0113e-01,  2.1576e-01, -5.2212e-01,
#         -3.4464e-01, -2.3047e-01,  2.2124e-01,  3.7953e-01,  4.8534e-02,
#          5.1055e-01, -8.6260e-02, -3.4697e-01, -2.4601e-01, -4.1027e-02,
#         -2.6159e-01,  1.0039e-01, -6.5045e-02,  2.3757e-02, -5.1150e-01,
#         -1.5968e-01, -7.4402e-02, -2.8282e-01, -8.2208e-01,  1.3997e-01,
#         -2.8945e-01, -5.1438e-01, -5.8233e-01,  1.6340e-01, -1.5759e-01,
#         -3.7231e-01,  2.2692e-01, -5.8590e-02,  2.1479e-01,  2.9464e-01,
#         -2.8443e-01, -1.4180e-01,  1.0166e-01, -1.5292e-01, -3.3876e-01,
#         -1.3808e-01,  5.1380e-01,  9.1901e-02,  3.0390e-01,  2.5259e-02,
#          4.5899e-02, -3.3531e-01,  4.5009e-01, -2.0979e-01,  3.4302e-01,
#         -5.5333e-01, -5.0188e-02,  3.5394e-01, -2.0881e-01, -1.2465e-01,
#         -3.1422e-01, -5.5219e-03,  4.2422e-01,  1.8648e-01,  2.8002e-01,
#         -2.2852e-02, -4.0197e-01, -2.3885e-01, -1.0734e-01,  4.3852e-01,
#         -2.4279e-01,  3.3549e-02,  4.9093e-01,  8.8161e-02, -1.1902e-01,
#          1.1161e-01,  9.5993e-02, -7.2172e-01,  3.0469e-01,  4.4417e-02,
#          3.2412e-01, -3.7982e-01, -2.3029e-01, -3.4987e-01, -4.2498e-01,
#          2.4770e-02, -5.5692e-01, -5.7750e-01,  1.8420e-01, -2.3091e-01,
#         -3.3785e-02, -5.1871e-01,  1.9248e-01, -9.2583e-02, -2.1766e-01,
#          2.7517e-01,  4.8763e-02, -1.9717e-01, -5.1125e-01, -6.1541e-02,
#         -1.0402e-01, -3.9787e-01, -2.1125e-01,  5.4275e-02, -4.4816e-01,
#          2.3492e-01,  7.3312e-02, -3.0561e-01,  3.8017e-01, -2.0985e-01,
#         -9.6843e-02,  3.0311e-01, -4.6657e-02, -4.7185e-02, -6.0980e-01,
#         -4.5456e-01, -2.7630e-01, -2.2813e-01,  2.3857e-01, -1.5739e-01,
#          8.6956e-02,  5.2625e-03,  1.9503e-01, -2.0647e-02,  4.0797e-01,
#          1.7377e-02, -3.1735e-01,  7.3151e-04, -3.2720e-01, -1.0167e-01,
#          1.1584e-01,  5.8311e-02, -1.4194e-01, -3.3995e-01,  4.7410e-01,
#          5.7677e-01,  8.1534e-02, -4.0992e-01, -4.9437e-01, -1.1042e-01,
#          4.3175e-01,  4.1470e-01,  4.4082e-02, -4.4849e-01, -9.3484e-02,
#          3.7944e-02, -6.0233e-02,  3.0573e-01, -2.1256e-01, -3.6759e-01,
#         -4.1138e-01,  2.6653e-01, -8.4644e-01, -2.6427e-01,  1.9039e-02,
#          3.0831e-01, -1.0121e-01, -2.4576e-01,  7.6075e-02,  7.3263e-02,
#          6.2095e-01, -1.1920e-01,  2.0358e-01, -2.5522e-02,  5.5796e-01,
#          3.3156e-01, -2.8891e-02,  2.3480e-01,  5.0475e-01,  2.7822e-01,
#          9.8097e-02, -6.1749e-02, -5.2471e-01, -7.1600e-01, -3.5413e-01,
#          3.9377e-01,  1.5386e-02, -2.9142e-01,  5.2910e-01, -1.4333e-01,
#          4.9358e-02, -1.1029e-01, -3.7662e-01,  5.1148e-02,  3.1390e-01,
#          6.8158e-02,  5.1599e-01,  3.2595e-01,  6.2470e-02, -1.9295e-01,
#         -1.9620e-01,  5.2164e-02,  7.2487e-01, -3.7806e-02,  9.4144e-03,
#          5.6233e-01, -7.3511e-02,  1.3338e-01,  6.0825e-01, -4.3840e-01,
#         -7.0543e-02,  1.9253e-01,  3.6454e-01,  6.2674e-02, -1.5863e-01,
#         -2.8499e-01,  2.7923e-01,  5.9363e-02,  1.4903e-01, -2.2921e-01,
#          2.7017e-01,  2.4537e-01,  1.6428e-01,  3.6569e-01,  2.5539e-01,
#          3.8360e-01, -2.8060e-01,  1.2361e-01, -1.1526e-02, -1.9266e-01,
#         -5.9680e-01,  4.3066e-01,  1.3559e-01,  3.0028e-01, -5.9617e-02,
#         -1.4276e-01, -3.7636e-01,  3.3238e-01,  1.6023e-01, -1.3176e-01,
#          4.7244e-02,  1.6542e-01,  2.2063e-01,  2.8546e-01, -6.1646e-02,
#         -4.7287e-01, -2.4634e-01,  9.5971e-02,  7.6168e-01, -3.6434e-01,
#         -5.2604e-02, -3.9802e-01, -2.9125e-01,  1.2927e-01, -3.0375e-01,
#          2.6685e-01, -1.6527e-01, -1.4572e-02,  2.9109e-01, -1.9375e-01,
#         -8.2894e-04,  9.6895e-02,  3.1722e-01,  2.5358e-02,  3.3655e-01,
#         -6.6408e-01, -4.7226e-02,  3.3285e-01, -4.1841e-01,  1.6408e-01,
#         -1.5067e-01, -3.7905e-01, -4.0689e-02, -6.9304e-02,  2.1773e-01,
#          4.4702e-02, -8.6090e-02,  7.2685e-02, -9.1370e-02, -3.8193e-01,
#         -4.6545e-01,  3.4848e-01, -3.7423e-01,  6.8712e-01, -5.1667e-01,
#          1.6739e-01, -1.8621e-01, -4.5266e-01, -2.4555e-01,  2.3510e-02,
#          2.2828e-01, -1.8290e-01, -3.8778e-01, -7.4615e-02,  5.0491e-01,
#         -1.5686e-01,  2.1906e-01,  3.2318e-01, -1.7289e-01, -1.0035e-01,
#         -2.2430e-01,  1.1674e-01,  9.5808e-02, -4.6600e-02, -4.1000e-01,
#         -6.1778e-01,  3.2140e-01,  4.7372e-01, -8.7243e-01,  1.8615e-01,
#          5.8243e-01, -2.6814e-01,  1.7639e-02, -4.6681e-01,  2.5475e-01,
#         -1.2259e-01, -5.7225e-01, -2.0738e-01,  8.7355e-02,  2.6069e-02,
#         -3.1170e-01, -6.7945e-02, -5.1853e-01,  1.5353e-01,  2.5949e-01,
#          3.3959e-01, -6.2935e-02, -1.3637e-01,  4.3082e-01,  1.7123e-02,
#         -2.0531e-01, -1.2607e-01, -6.9892e-01,  1.7694e-01, -4.2990e-01,
#          4.0172e-01,  4.1446e-01,  7.4672e-02,  5.8570e-02, -8.5846e-03,
#         -2.0707e-01,  7.1902e-02,  1.5358e-01]]]).to(device)
#         self.corpus = [self.uc, self.grass_green]
#         # self.corpus.append(data=self.uc)
#         # # , self.grass_green])
#         self.state_map = None
#         self.expected_action = None
#         self.reset()
#
#     def reset(self):
#         self.expected_action = random.randint(0,1)
#         self.state_map = self.corpus[self.expected_action]
#
#     def get_screen(self):
#         return self.state_map
#
#     def step(self, step_action):
#         # before_quality = self._map_quality()
#         # NB: punishment is always a positive integer
#         # step_punishment, step_reward = self.apply_action(step_action)
#         # step_reward = step_reward - step_punishment
#         # after_quality = self._map_quality()
#         # # reward = (after_quality - before_quality) - punishment_for_not_moving
#         # is_completed = after_quality >= 1
#         if step_action == self.expected_action:
#             return 1, True
#         else:
#             return 0, True




class Station:
    def __init__(self):
        self.x = 0
        self.y = 0

    def set_random_coords(self, map_width, map_height):
        self.x = random.randint(0, map_width - 1)
        self.y = random.randint(0, map_height - 1)

    def add_to_map(self, state_map):
        state_map[self.y, self.x] = 1
        return state_map

    def move(self, state_map, move_action, max_width, max_height, other_x, other_y):
        state_map[self.y, self.x] = 0
        moved = False
        move_reward = 0
        punishment = 0
        if move_action == 0 and self.y > 0:
            # print("up")
            # up
            before_difference = self._y_difference(other_y)
            self.y -= 1
            move_reward += self._reward(before_difference, self._y_difference(other_y))
            punishment += self._punishment(before_difference, self._y_difference(other_y))
            moved = True
        if move_action == 1 and self.x + 1 < max_width:
            # print("right")
            # right
            before_difference = self._x_difference(other_x)
            self.x += 1
            move_reward += self._reward(before_difference, self._x_difference(other_x))
            punishment += self._punishment(before_difference, self._x_difference(other_x))
            moved = True
        if move_action == 2 and self.y + 1 < max_height:
            # down
            # print("down")
            before_difference = self._y_difference(other_y)
            self.y += 1
            move_reward += self._reward(before_difference, self._y_difference(other_y))
            punishment += self._punishment(before_difference, self._y_difference(other_y))
            moved = True
        if move_action == 3 and self.x > 0:
            # left
            # print("left")
            before_difference = self._x_difference(other_x)
            self.x -= 1
            move_reward += self._reward(before_difference, self._x_difference(other_x))
            punishment += self._punishment(before_difference, self._x_difference(other_x))
            moved = True

        if not moved:
            print(f"no change: {str(move_action)}")
            punishment += 1
        state_map[self.y, self.x] = 1
        return [state_map, punishment, move_reward]

    def _x_difference(self, other_x):
        return abs(self.x - other_x)

    def _y_difference(self, other_y):
        return abs(self.y - other_y)

    def _reward(self, before_difference, after_difference):
        if after_difference < before_difference:
            return 0.1
        return 0

    def _punishment(self, before_difference, after_difference):
        if after_difference >= before_difference:
            return 1
        return 0


class GeneratorEnvironment:
    def __init__(self, width, height):
        self.width = 10
        self.height = 10
        self.state_map = None
        self.station_one = Station()
        self.station_two = Station()
        self.reset()

    def reset(self):
        while self._map_quality() == 1 or self._map_quality() == -1:
            self.state_map = torch.zeros(self.width, self.height)
            self.station_one.set_random_coords(self.width, self.height)
            self.station_two.set_random_coords(self.width, self.height)
            self.state_map = self.station_one.add_to_map(self.state_map)
            self.state_map = self.station_two.add_to_map(self.state_map)
        print(self.state_map)

    def get_screen(self):
        return self.state_map.unsqueeze(0).unsqueeze(0)

    def step(self, step_action):
        # before_quality = self._map_quality()
        # NB: punishment is always a positive integer
        step_punishment, step_reward = self.apply_action(step_action)
        step_reward = step_reward - step_punishment
        after_quality = self._map_quality()
        # reward = (after_quality - before_quality) - punishment_for_not_moving
        is_completed = after_quality >= 1
        return step_reward, is_completed

    def apply_action(self, action_to_apply):
        # print("before")
        # print(self.map)
        print(f"original action: {action_to_apply}")
        if action_to_apply <= 3:
            self.state_map, punishment, action_reward = self.station_one.move(self.state_map, action_to_apply, self.width, self.height, self.station_two.x, self.station_two.y)
        else:
            action_to_apply -= 4
            self.state_map, punishment, action_reward = self.station_two.move(self.state_map, action_to_apply, self.width, self.height, self.station_one.x, self.station_one.y)
        return punishment, action_reward
        # print("after")
        # print(self.map)

    def _map_quality(self):
        quality = 0
        if self.station_one.x == self.station_two.x:
            quality = 1
        elif self.station_one.y == self.station_two.y:
            quality = 1
        if self.station_one.x == self.station_two.x and self.station_one.y == self.station_two.y:
            # Being in the same location is bad
            quality = -1
        return quality