import numpy as np
import scipy.io
import torch
import torch.nn.functional as F

mat = scipy.io.loadmat('psf.mat')
psf = mat['normalized_VIS_PSF']

y_num, x_num = 8, 10
patch_h, patch_w = 64, 64

psf_4d = psf.reshape(x_num, y_num, patch_h, patch_w)
psf_4d = np.transpose(psf_4d, (1, 0, 2, 3))

psf_80= psf_4d.reshape(80, 64, 64)

np.save('psf_80.npy', psf_80)
psf_npy = np.load('psf_80.npy').reshape(8, 10, 64, 64)

psf_pt = torch.tensor(psf_npy).float()
torch.save(psf_pt, 'psf_64.pt')
psf_pt = F.adaptive_avg_pool2d(psf_pt, (7, 7))
print(psf_pt.shape)

# img = np.zeros((y_num * patch_h, x_num * patch_w))
# for i in range(y_num):
#     for j in range(x_num):
#         img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = psf_npy[i, j]
#
# plt.imshow(img * 100, cmap='gray', vmin=0, vmax=np.max(img)*100)
# plt.show()
#
# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
#
# img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
#
# print(f"原始张量尺寸: {img_tensor.shape}")
#
# kernel_size = 2  # 池化窗口大小
# stride = 2       # 步长
#
# pooled_tensor = F.avg_pool2d(img_tensor, kernel_size=kernel_size, stride=stride)
#
# print(f"池化后张量尺寸: {pooled_tensor.shape}")

import torch
import numpy as np
import matplotlib.pyplot as plt

# 要加载的文件名
filename = 'psf_7.pt'

try:
    # 1. 加载 .pt 张量文件
    psf_tensor = torch.load(filename)
    print(f"成功加载 '{filename}'")
    print(f"张量形状: {psf_tensor.shape}")

except FileNotFoundError:
    print(f"错误: 文件 '{filename}' 未找到。请确保文件存在于当前目录。")
    exit()
except Exception as e:
    print(f"加载文件时发生错误: {e}")
    exit()

# 检查张量维度是否符合预期
if psf_tensor.dim() != 4:
    print(f"错误: 期望一个4维张量 (N_y, N_x, H, W)，但得到 {psf_tensor.dim()} 维。")
    exit()

# 2. 将张量转换为 NumPy 数组以便拼接和显示
psf_numpy = psf_tensor.cpu().numpy()

# 从形状中获取网格和图像块的尺寸
y_num, x_num, patch_h, patch_w = psf_numpy.shape

# 3. 创建一个空画布来拼接所有图像块
stitched_image = np.zeros((y_num * patch_h, x_num * patch_w))

# 4. 循环并将每个图像块放入大图中
for i in range(y_num):
    for j in range(x_num):
        stitched_image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w] = psf_numpy[i, j]

# 5. 使用 matplotlib 显示拼接后的图像
plt.figure(figsize=(8, 6))
plt.imshow(stitched_image, cmap='gray')
plt.show()