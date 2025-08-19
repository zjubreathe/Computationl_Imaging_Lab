# Computational Imaging Lab
## 文献仓库
### 1.Image restoration for spatially varying degradation based on PSF properties
- 光学退化过程可以看作是清晰图像与模糊核的卷积。
- 当模糊核随空间位置变化时，卷积模型不再有效的适用。
- 有效的空间特异解卷积需要满足：
- firstly, the blur kernels used should change continuously with the FoV, and this change pattern is not a linear or monotonic variation
- secondly, the blur kernels for deconvolution should vary on a per-pixel basis, specifically in terms of shape and energy distribution
- 使用中心FoV的PSF进行维纳反卷积，该反卷积不随FoV而变化，无法表征空间变化问题
- 逐块反卷积策略，相邻块之间的 PSF 是阶跃变化的，这与实际的连续变化特性不匹配
- 为了表征PSF的连续变化特性，将PSF的连续变化建模为随着FoV的增加而均匀变化
- 实际的PSF变化模式可能不是单调的，某些位置的模糊会随着FoV的增加而降低
1. Fast approximations of shift-variant blur 2015
2. Annular computational imaging: Capture clear panoramic images through simple lens 2022
3. End-to-end learned single lens design using improved Wiener deconvolution 2023
4. Improved performance of a hybrid optical/digital imaging system with fast piecewise Wiener deconvolution 2022
5. Deep learning for fast spatially varying deconvolution 2022
6. Optical aberrations correction in postprocessing using imaging simulation csq
7. Extreme-quality computational imaging via degradation framework csq
8. Computational optics for mobile terminals in mass production csq
- 卷积核基于全局共享机制，空间不变，表征不了模糊核的权重和形状随FOV变化的特性
- 逐像素空间变化卷积核（形状和权重），计算成本高，表征不了模糊核随FOV变化的特性
- 基于PSF感知的修复工作，忽略了PSF的连续变化特性，仅依靠采样场的PSF直接求解反卷积核
1. Non-blind optical degradation correction via frequency self-adaptive and finetune tactics cyt
2. Optical aberrations correction in postprocessing using imaging simulation csq
3. Extreme-quality computational imaging via degradation framework csq 
4. Minimalist and high-quality panoramic imaging with PSF-aware transformers wkw
5. OSRT: Omnidirectional image super-resolution with distortion-aware transformer 2023
6. OPDN: Omnidirectional position-aware deformable network for omnidirectional image super-resolution 2023