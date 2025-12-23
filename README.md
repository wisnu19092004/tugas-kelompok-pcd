# Aplikasi Segmentasi Citra Digital (Streamlit)

Aplikasi segmentasi citra digital berbasis Streamlit dengan pemisahan:
- `segmentation.py` = engine/algoritma segmentasi
- `app.py` = aplikasi UI (termasuk Dashboard + histori run)

## Fitur
- Upload gambar (JPG/PNG)
- Pilih algoritma:
  - Otsu Thresholding
  - K-Means Clustering
  - Watershed
  - DeepLabV3-ResNet50 (pretrained torchvision)
  - FCN-ResNet50 (pretrained torchvision)
  - UNet (SMP) + upload weights (.pth/.pt)
  - FPN (SMP) + upload weights (.pth/.pt)
- Dashboard:
  - Ringkasan session
  - Preview hasil terakhir
  - Histori run
  - Grafik jumlah run dan runtime per algoritma
- Download output:
  - Overlay (PNG)
  - Mask (PNG)

## Anggota Kelompok
- Wisnu Dwi Haryanto (A11.2023.15266)
- Disna Refarlan Widistomo (A11.2023.15265)
- Ali Murtadho (A11.2023.15269)
- Erlangga Rayhan Hernawan (A11.2022.14419)
- Adiestya Rully Hermawan (A11.2022.14386)
- Unggul Sakti Ardiansyah (A11.2022.14335)

## Cara Menjalankan (Windows + VS Code)
1. Buka folder project:
   ```powershell
   cd E:\seg_app
