# 🎭 VividFace: A Diffusion-Based Hybrid Framework for High-Fidelity Video Face Swapping

***“Revolutionize Video Face Swapping with Robust, Diffusion-Powered, Temporal Consistency-Driven Innovation!”*** 

![pipeline](assets/pipeline.png)


This repository contains code for the paper [VividFace: A Diffusion-Based Hybrid Framework for High-Fidelity Video Face Swapping](https://arxiv.org/abs/2412.11279).

We propose a diffusion-based framework for video face swapping, featuring hybrid training, an AIDT dataset, and 3D reconstruction for superior identity preservation and temporal consistency.

🌐 [**Project Page**](https://hao-shao.com/projects/vividface.html) | 🤗 [**Hugging Face Models**](https://huggingface.co/deepcs233/VividFace/tree/main)

## News

- **2025-10-15**: 🔓 Code and pre-trained weights released!  
- **2025-09-19**: 🎉 Our paper **VividFace** has been **accepted to NeurIPS 2025**!

## Installation

1. Create and activate the Conda environment:
   ```bash
   conda create --name vividface python=3.8
   conda activate vividface
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the dependency for Deep3DFaceRecon_pytorch:
    ```bash
   cd Deep3DFaceRecon/nvdiffrast
   pip install .
   ```   

## Model Weights

Download the required models and place them in the correct directories:

| Model | Source | Destination |
|-------|---------|-------------|
| **VividFace Weights** | [Hugging Face](https://huggingface.co/deepcs233/VividFace/tree/main) | `weights/` |
| **BFM Model** | Included in VividFace weights | `Deep3DFaceRecon/BFM/` |
| **Stable Diffusion v1.5** | [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | `weights/stable-diffusion-v1-5/` |


## Testing

We have prepared some sample files in the `examples/` folder for testing. Use the following command to run a test:

```bash
python infer.py examples
```

This will sequentially replace faces in the videos located in `examples/videos/` with faces from `examples/faces/` (e.g., the first video in `examples/videos/` will have its face replaced by the first face in `examples/faces/`).

After execution, you can find the output results in the `outputs/` directory.

---

## Testing with Custom Data

If you want to test your own data, follow the format of the `examples/` folder:

1. Each video must have a corresponding `.txt` file with the same name.
2. The `.txt` file should have the same number of lines as the number of frames in the video.
3. Each line must contain **14 values**:
   - The first 4 values represent the **face bounding box (bbox)**.
   - The next 10 values represent **5 facial keypoints**.
4. Ensure that faces are cropped properly. We recommend using [insightface](https://github.com/deepinsight/insightface) for face cropping.

---

## Training

We provide training scripts, but the dataset is not yet available for public use. Therefore, training cannot be executed at this time. However, you can explore the code by running:

```bash
bash run.sh
```

---

## VAE Training

To train our **Face3DVAE**, navigate to the `face3dvae` directory and run the training script:

```bash
cd face3dvae
bash train.sh
```

---


## Citation

If you find our work helpful, please cite:

```
@article{shao2024vividface,
  title={VividFace: A Diffusion-Based Hybrid Framework for High-Fidelity Video Face Swapping},
  author={Shao, Hao and Wang, Shulun and Zhou, Yang and Song, Guanglu and He, Dailan and Qin, Shuo and Zong, Zhuofan and Ma, Bingqi and Liu, Yu and Li, Hongsheng},
  journal={arXiv preprint arXiv:2412.11279},
  year={2024}
}
```

## Disclaimer
This project is released for academic use. We disclaim responsibility for user-generated content.

## Acknowledgements

Our work builds upon the following excellent projects:
[Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch?tab=readme-ov-file#prepare-prerequisite-models)
[InsightFace](https://github.com/deepinsight/insightface)
[AnimateDiff](https://github.com/guoyww/AnimateDiff/tree/main)

## License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
