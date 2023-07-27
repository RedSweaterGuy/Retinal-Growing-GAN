# VCBM-Growing-GAN

## Requirements
* We used a Windows system (10 Pro 64 bit), with 64 GB of RAM and an Nvidia RTX 3090.
* We recommend to use at least 64 GB of RAM and a modern GPU with VRAM of 24 GB or more.
* For Python, we provide a ``requirements.txt`` file containing all the required packages for Python version 3.9. CUDA was used in version 12.0.

## Results from paper
The training results can be found in the ``runs`` folder, with those of the original paper in ``original``, and ours in ``growing``.

Each runs folder name contains:
* dataset (DRIVE or STARE),
* rotation of the images (1 to 360 degrees),
* number of training rounds, or list of training rounds,
* and date and time of training.

The sample segmentations can be found in ``segmentation_results_``.

The auc statistics can be found in ``auc_image_1``

The models produced and their weights can be found in ``model_image_1``


## Rerunning code
Man führt die ``python main.py`` aus, der random seed ist schon gesetzt.
Für andere Ergebnisse können die run Konfigurationen geändert oder der seed geändert werden.
