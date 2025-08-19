import torch
import torch.nn as nn
import torch.nn.functional as F

from ConverseNet import  ConverseUSRNet

class Net(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: int = 64, patch_size: int = 128,
                 patch_overlap: int = 64):

        super().__init__()

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.stride = patch_size - patch_overlap

        self.deblur_model = ConverseUSRNet(num_iterations=5, in_channels=features, num_blocks=7)

        hanning_window = torch.hann_window(patch_size, periodic=False)
        self.patch_weight_window = hanning_window.unsqueeze(1) * hanning_window.unsqueeze(0)
        self.patch_weight_window = self.patch_weight_window.unsqueeze(0).unsqueeze(0)

    def forward(self, blurry_image: torch.Tensor, kernel_grid: torch.Tensor, sf: int = 1) -> torch.Tensor:
        """
        对具有空间变化模糊核的图像执行去模糊。

        Args:
            blurry_image (torch.Tensor): 输入的模糊图像，形状为 [B, C, H, W]。
            kernel_grid (torch.Tensor): 空间变化的模糊核网格。
                                        形状为 [grid_h, grid_w, kernel_h, kernel_w]，
                                        例如 [8, 10, 8, 8]。
            sf (int): 缩放因子，传递给去模糊模型。

        Returns:
            torch.Tensor: 去模糊后的图像。
        """
        B, C, H, W = blurry_image.shape
        device = blurry_image.device
        self.patch_weight_window = self.patch_weight_window.to(device)
        kernel_grid = kernel_grid.to(device)

        grid_h, grid_w, _, _ = kernel_grid.shape

        result_image = torch.zeros_like(blurry_image)
        weight_accumulator = torch.zeros(B, 1, H, W, device=device)

        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                # 确定块的边界，并处理边缘情况
                y_start = y
                y_end = min(y + self.patch_size, H)
                x_start = x
                x_end = min(x + self.patch_size, W)

                # 从完整图像中裁剪出当前块，如果需要则进行填充
                image_patch = torch.zeros(B, C, self.patch_size, self.patch_size, device=device)
                image_patch[:, :, :y_end - y_start, :x_end - x_start] = blurry_image[:, :, y_start:y_end, x_start:x_end]

                # 计算块中心，并映射到模糊核网格以提取局部模糊核
                center_y = y_start + (y_end - y_start) // 2
                center_x = x_start + (x_end - x_start) // 2

                # 根据中心点位置计算在核网格中的索引
                grid_y_idx = int(center_y / H * grid_h)
                grid_x_idx = int(center_x / W * grid_w)

                # 确保索引在有效范围内
                grid_y_idx = min(grid_y_idx, grid_h - 1)
                grid_x_idx = min(grid_x_idx, grid_w - 1)

                # 提取局部模糊核并调整其形状以适应去模糊模型
                local_kernel = kernel_grid[grid_y_idx, grid_x_idx, :, :]
                local_kernel = local_kernel.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

                # 使用去模糊模型处理块
                deblurred_patch = self.deblur_model(image_patch, local_kernel, sf)

                # 将加权后的去模糊块累加到结果图像中
                result_image[:, :, y_start:y_end, x_start:x_end] += \
                    deblurred_patch[:, :, :y_end - y_start, :x_end - x_start] * self.patch_weight_window[:, :,
                                                                                :y_end - y_start, :x_end - x_start]
                weight_accumulator[:, :, y_start:y_end, x_start:x_end] += self.patch_weight_window[:, :,
                                                                          :y_end - y_start, :x_end - x_start]

        # 通过除以权重总和来归一化结果
        final_image = result_image / (weight_accumulator + 1e-8)

        return final_image