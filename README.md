# GAN-based Anomaly Detection (GANomaly-style)

This project use a **GAN-based anomaly detector** inspired by **GANomaly**:
- **Generator**: encoder–decoder–encoder
- **Discriminator**: patch discriminator on images
- **Anomaly score**: weighted sum of image reconstruction error and latent consistency error.

It also includes a **Streamlit app** for quick demo and thresholding tools.

## Quick start

```bash
# 1) Install
pip install -r requirements.txt

# 2) Prepare data
Normal training images under data/train/<class_name>/
Test images under data/test/<class_name>/good and data/test/<class_name>/anomaly

# 3) Train
python train.py --data_root data --category <class_name> --epochs 50

# 4) Inference (scores + optional masks)
python inference.py --data_root data --category <class_name> --weights runs/<run_name>/weights/best.pt

# 5) App
streamlit run app.py
```

## Structure
```
anomaly_gan_project/
├── app.py
├── configs.json
├── models/
│   └── ganomaly.py
├── src/
│   ├── dataset.py
│   ├── utils.py
│   └── metrics.py
├── train.py
├── inference.py
├── explain.py
├── requirements.txt
└── README.md
```

## Notes
- Default image size: 256. Change via CLI or `configs.json`.
- Works on grayscale or RGB images (auto-detected).
- Threshold can be set via percentile on normal validation scores or Youden’s J on ROC if labels available.

## 📚 References

* **MVTec AD Dataset:** [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
* **GANomaly:** Akcay S., Atapour-Abarghouei A., Breckon T.P. (2019) GANomaly: Semi-supervised Anomaly Detection via Adversarial Training. In: Jawahar C., Li H., Mori G., Schindler K. (eds) Computer Vision – ACCV 2018. ACCV 2018. Lecture Notes in Computer Science, vol 11363. Springer, Cham
* **LIME:** Ribeiro et al., *“Why Should I Trust You?”* (KDD 2016)
* **Autoencoder for Anomaly Detection:** [https://arxiv.org/abs/2007.14115](https://arxiv.org/abs/2007.14115)
