import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_d11(nn.Module):
    def __init__(self):
        super(Conv_d11, self).__init__()
        kernel = [[-0.5, 0, 0],
                  [0, 1, 0],
                  [0, 0, -0.5]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)

class Conv_d12(nn.Module):
    def __init__(self):
        super(Conv_d12, self).__init__()
        kernel = [[0, -0.5, 0],
                  [0, 1, 0],
                  [0, -0.5, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d13(nn.Module):
    def __init__(self):
        super(Conv_d13, self).__init__()
        kernel = [[0, 0, -0.5],
                  [0, 1, 0],
                  [-0.5, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)


class Conv_d14(nn.Module):
    def __init__(self):
        super(Conv_d14, self).__init__()
        kernel = [[0, 0, 0],
                  [-0.5, 1, -0.5],
                  [0, 0, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=1)

class Conv_d21(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[-0.5, 0.5],
                  [0, 0]]
        # 初始化卷积核 (2,2)
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, x):
        # 输入尺寸处理：256x256 -> 填充后257x257
        x_padded = F.pad(x, (0, 1, 0, 1), mode='reflect')  # 右侧和下侧各填充1像素
        # 执行2x2卷积（无自动填充）
        return F.conv2d(x_padded, self.weight, padding=0)

class Conv_d22(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[0.5, -0.5],
                  [0, 0]]
        # 初始化卷积核 (2,2)
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, x):
        # 输入尺寸处理：256x256 -> 填充后257x257
        x_padded = F.pad(x, (0, 1, 0, 1), mode='reflect')  # 右侧和下侧各填充1像素
        # 执行2x2卷积（无自动填充）
        return F.conv2d(x_padded, self.weight, padding=0)

class Conv_d23(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[0.5, 0],
                  [-0.5, 0]]
        # 初始化卷积核 (2,2)
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, x):
        # 输入尺寸处理：256x256 -> 填充后257x257
        x_padded = F.pad(x, (0, 1, 0, 1), mode='reflect')  # 右侧和下侧各填充1像素
        # 执行2x2卷积（无自动填充）
        return F.conv2d(x_padded, self.weight, padding=0)

class Conv_d24(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[-0.5, 0],
                  [0.5, 0]]
        # 初始化卷积核 (2,2)
        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(self.kernel, requires_grad=False)

    def forward(self, x):
        # 输入尺寸处理：256x256 -> 填充后257x257
        x_padded = F.pad(x, (0, 1, 0, 1), mode='reflect')  # 右侧和下侧各填充1像素
        # 执行2x2卷积（无自动填充）
        return F.conv2d(x_padded, self.weight, padding=0)

# class Conv_d11(nn.Module):
#     def __init__(self):
#         super(Conv_d11, self).__init__()
#         kernel = [[-0.5, 0, 0],
#                   [0, 1, 0],
#                   [0, 0, -0.5]]
#         # 扩展为3个输入通道，每个通道使用相同核
#         kernel = torch.FloatTensor(kernel).unsqueeze(0)  # (1,3,3)
#         kernel = kernel.repeat(3, 1, 1).unsqueeze(0)     # (1,3,3,3)
#         self.weight = nn.Parameter(data=kernel, requires_grad=False)
#
#     def forward(self, input):
#         return F.conv2d(input, self.weight, padding=1)
#
# class Conv_d12(nn.Module):
#     def __init__(self):
#         super(Conv_d12, self).__init__()
#         kernel = [[0, -0.5, 0],
#                   [0, 1, 0],
#                   [0, -0.5, 0]]
#         kernel = torch.FloatTensor(kernel).unsqueeze(0)
#         kernel = kernel.repeat(3, 1, 1).unsqueeze(0)
#         self.weight = nn.Parameter(data=kernel, requires_grad=False)
#
#     def forward(self, input):
#         return F.conv2d(input, self.weight, padding=1)
#
# class Conv_d13(nn.Module):
#     def __init__(self):
#         super(Conv_d13, self).__init__()
#         kernel = [[0, 0, -0.5],
#                   [0, 1, 0],
#                   [-0.5, 0, 0]]
#         kernel = torch.FloatTensor(kernel).unsqueeze(0)
#         kernel = kernel.repeat(3, 1, 1).unsqueeze(0)
#         self.weight = nn.Parameter(data=kernel, requires_grad=False)
#
#     def forward(self, input):
#         return F.conv2d(input, self.weight, padding=1)
#
# class Conv_d14(nn.Module):
#     def __init__(self):
#         super(Conv_d14, self).__init__()
#         kernel = [[0, 0, 0],
#                   [-0.5, 1, -0.5],
#                   [0, 0, 0]]
#         kernel = torch.FloatTensor(kernel).unsqueeze(0)
#         kernel = kernel.repeat(3, 1, 1).unsqueeze(0)
#         self.weight = nn.Parameter(data=kernel, requires_grad=False)
#
#     def forward(self, input):
#         return F.conv2d(input, self.weight, padding=1)
#
# class Conv_d21(nn.Module):
#     def __init__(self):
#         super().__init__()
#         kernel = [[-0.5, 0.5],
#                   [0, 0]]
#         kernel = torch.FloatTensor(kernel).unsqueeze(0)  # (1,2,2)
#         kernel = kernel.repeat(3, 1, 1).unsqueeze(0)     # (1,3,2,2)
#         self.weight = nn.Parameter(kernel, requires_grad=False)
#
#     def forward(self, x):
#         x_padded = F.pad(x, (0, 1, 0, 1), mode='reflect')
#         return F.conv2d(x_padded, self.weight, padding=0)
#
# class Conv_d22(nn.Module):
#     def __init__(self):
#         super().__init__()
#         kernel = [[0.5, -0.5],
#                   [0, 0]]
#         kernel = torch.FloatTensor(kernel).unsqueeze(0)
#         kernel = kernel.repeat(3, 1, 1).unsqueeze(0)
#         self.weight = nn.Parameter(kernel, requires_grad=False)
#
#     def forward(self, x):
#         x_padded = F.pad(x, (0, 1, 0, 1), mode='reflect')
#         return F.conv2d(x_padded, self.weight, padding=0)
#
# class Conv_d23(nn.Module):
#     def __init__(self):
#         super().__init__()
#         kernel = [[0.5, 0],
#                   [-0.5, 0]]
#         kernel = torch.FloatTensor(kernel).unsqueeze(0)
#         kernel = kernel.repeat(3, 1, 1).unsqueeze(0)
#         self.weight = nn.Parameter(kernel, requires_grad=False)
#
#     def forward(self, x):
#         x_padded = F.pad(x, (0, 1, 0, 1), mode='reflect')
#         return F.conv2d(x_padded, self.weight, padding=0)
#
# class Conv_d24(nn.Module):
#     def __init__(self):
#         super().__init__()
#         kernel = [[-0.5, 0],
#                   [0.5, 0]]
#         kernel = torch.FloatTensor(kernel).unsqueeze(0)
#         kernel = kernel.repeat(3, 1, 1).unsqueeze(0)
#         self.weight = nn.Parameter(kernel, requires_grad=False)
#
#     def forward(self, x):
#         x_padded = F.pad(x, (0, 1, 0, 1), mode='reflect')
#         return F.conv2d(x_padded, self.weight, padding=0)
