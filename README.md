# GenSeT-CUDA
This is the repository of our published paper in ISSPIT2018: [Accelerating GenSeT Reconstruction for Sparsely Sampled DCE-MRI with GPU](https://ieeexplore.ieee.org/document/8642674)

For quickly checking the result in the main paper, please refer to **demo2.sln** and **kernel.cu**. The project is best opened in Visual Studio Code with CUDA installation.

## Architecture
![alt text](https://github.com/xzZero/GenSeT-CUDA/blob/main/architech.png)
## Result
- Speed-factor: 48x
- Error: reduced from 14% to 3%

![alt text](https://github.com/xzZero/GenSeT-CUDA/blob/main/result.png)
## Abstract
Dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI) uses radioactive contrast agents as a tracer to provide tumor morphology and contrast kinetics information of tumor regions, which are crucial in breast cancer diagnosis and treatment. The effectiveness of the imaging modality relies on its capacity to acquire dynamic data at a sufficient sampling rate to gain the desired temporal resolution. This can be achieved by sparsely sampling the k-space data and applying advanced image reconstruction method that exploits compressed sensing, such as the recently proposed GenSeT (Generalized Series with Temporal constraint) method. Due to the highly nonlinear nature of compressed-sensing-based approach, computational complexity of the reconstruction algorithm is a practical challenge, especially for large breast DCE-MRI datasets. In this study, the GenSeT algorithm was implemented in GPU using CUDA platform to significantly reduce reconstruction time, yielding a much more practical solution. Experimental results showed that for a breast DCE-MRI data, the proposed GPU-based GenSeT implementation achieved approximately 48 times faster in the reconstruction time as compared to the CPU approach, without sacrificing the image quality. Although this work focuses on accelerating image reconstruction for sparsely-sampled breast DCE-MRI, the proposed GPU-based algorithm can be easily applied for sparsely sampled DCE-MRI of other organs.
