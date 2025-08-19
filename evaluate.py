import glob
import numpy as np
import os
import torch
import cv2
from piqa import PSNR, SSIM
from typing import Dict, Any
from config import load_config


def evaluate_net(config: Dict[str, Any]):
    device = torch.device('cuda' if config['device']['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    data_config = config['data']

    psnr_metric = PSNR().to(device)
    ssim_metric = SSIM(n_channels=1).to(device)

    psnr_list = []
    ssim_list = []

    tests_path = data_config['test_path']
    tests_path = glob.glob(os.path.join(tests_path, 'blur/*.jpg'))

    print(f"开始评估 {len(tests_path)} 个测试样本...")

    for idx, test_path in enumerate(tests_path):
        eval_path = test_path.replace('blur', 'sharp')

        image = cv2.imread(test_path)
        label = cv2.imread(eval_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        image = image.astype(np.float32) / 255.0
        label = label.astype(np.float32) / 255.0

        image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
        label = np.expand_dims(np.expand_dims(label, axis=0), axis=0)

        image_tensor = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        label_tensor = torch.from_numpy(label).to(device=device, dtype=torch.float32)

        psnr_value = psnr_metric(image_tensor, label_tensor).item()
        ssim_value = ssim_metric(image_tensor, label_tensor).item()

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)

        if idx % 10 == 0:
            print(f"处理进度: {idx}/{len(tests_path)}, PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.3f}")

    if len(psnr_list) > 0:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f'\n评估完成！')
        print(f'平均 PSNR: {avg_psnr:.2f} dB')
        print(f'平均 SSIM: {avg_ssim:.3f}')
        print(f'有效样本数: {len(psnr_list)}/{len(tests_path)}')
    else:
        print("没有成功处理任何样本！")

if __name__ == "__main__":
    config = load_config('config.yaml')
    evaluate_net(config)
