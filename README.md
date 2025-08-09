# CALAMITI-based Medical Image Harmonization

This repository presents an advanced implementation of **medical image harmonization** using deep learning, focusing on **multi-modality MRI datasets**. The project leverages **CALAMITI** (Channel-wise Alignment Learning for Multi-modality Image Translation with Unpaired Data) and compares its performance against two other baseline models — **CycleGAN** and **FusionNet** — for effective cross-modality image synthesis and harmonization.

---

## 🧠 Project Overview

Multi-modal medical images (e.g., T1, T2, FLAIR) often differ in appearance due to variations in scanning protocols, vendor-specific artifacts, and patient-specific factors. This can hinder the performance of downstream AI tasks such as segmentation or classification. The goal of this project is to **standardize image appearance across modalities** while retaining anatomical fidelity using deep learning-based image translation techniques.

---

## 🔬 Models Implemented

1. **CycleGAN**
   - Unpaired image-to-image translation using adversarial training.
   - Trained to map between different MRI modalities using 2D slice-based inputs.

2. **FusionNet**
   - A deep convolutional encoder-decoder architecture.
   - Combines features across multiple scales for better harmonization output.

3. **CALAMITI**
   - Channel-wise Alignment Learning to adapt to variations in multi-modal images.
   - Utilizes feature-level alignment and perceptual loss to improve harmonization quality.

---

## 📁 Dataset

- **Source**: Publicly available unpaired multi-modal MRI datasets (e.g., BraTS, IXI)
- **Preprocessing**:
  - Rescaling and intensity normalization
  - 2D slice extraction (Axial view)
  - Unpaired modality assignment

---

## 🛠️ Tech Stack

- **Framework**: PyTorch
- **Environment**: Google Colab / CPU-based training (initially)
- **Deployment Ready**: Streamlit app for visualization and live inference

---

## 🚀 Features

- 🔄 **Unpaired Harmonization** using advanced GAN techniques
- 📊 **Comparison Dashboard** to evaluate harmonized output across models
- 🧪 **Metrics Tracked**: SSIM, PSNR, MAE (Mean Absolute Error)
- 🧰 **Visualization Tools**: Confusion matrix (where applicable), side-by-side viewer
- 📎 **Streamlit App**: Plug-and-play interface for uploading and harmonizing new scans

---

## 📈 Results Snapshot

| Model      | SSIM ↑ | PSNR ↑ | Qualitative Fidelity |
|------------|--------|--------|----------------------|
| CycleGAN   | 0.72   | 22.4 dB| Medium               |
| FusionNet  | 0.78   | 24.1 dB| Good                 |
| CALAMITI   | 0.84   | 26.5 dB| Excellent            |

> *Note: Metrics based on a held-out validation set with T1 → T2 harmonization task.*

---

## 🧪 Run Locally

```bash
git clone https://github.com/yourusername/calamiti-image-harmonization.git
cd calamiti-image-harmonization
pip install -r requirements.txt
python train.py --model cycle_gan  # or fusionnet / calamiti

