# Pix2Pix Image-to-Image Translation Model

Pix2Pix is a deep learning model for image-to-image translation tasks. It uses conditional Generative Adversarial Networks (cGANs) to translate images from one domain to another. This project demonstrates the implementation of Pix2Pix using a dataset of map-to-aerial image translations.

---

## Folder Structure

```
.
├── .gitkeep                     # Placeholder for empty directories
├── config.py                    # Configuration file for the model and dataset
├── dataset.py                   # Dataset preparation and loading script
├── install_requirements.sh      # Shell script to install required dependencies
├── models.py                    # Model definition for Pix2Pix (generator and discriminator)
└── pix2pix.py                   # Main script to train and test the Pix2Pix model
```

---

## Dataset

The dataset used for this project is sourced from Kaggle and contains paired images of maps and aerial views. The Pix2Pix model is trained to translate between these paired domains.

- **Dataset Source**: [Pix2Pix Dataset - Kaggle](https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset/data?select=maps)

---

## Getting Started

### Prerequisites

Before running the project, ensure that the following dependencies are installed:

- Python >= 3.7
- PyTorch
- torchvision
- Other dependencies listed in `install_requirements.sh`

Run the following command to install dependencies:
```bash
bash install_requirements.sh
```




## Features

- Fully configurable Pix2Pix model implementation
- Dataset preprocessing and augmentation scripts
- Training and evaluation pipelines

---

## Acknowledgments

This project was developed at **TU Berlin**, in the **Machine Learning Group**, by the following contributors:

- **Iyadh Ben Cheikh El Arbi**
- **Maroua Goghr**
- **Safouane Nciri**

Special thanks to the creators of the [Pix2Pix Dataset](https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset/data?select=maps) on Kaggle.

---

