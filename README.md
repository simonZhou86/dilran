# DILRAN for medical image fusion

University of Toronto CSC2529 Computational Imaging project.

''An attention-based Multi-Scale Feature Learning Framework for Multimodal Medical Image Fusion''

data is available upon request

## Usage

You may want to change to your own dataset. If you have a 3-channel PET or SPECT image, you may want to change the dataset_loader.py file

To train the network, run
```bash
python3 ./train_with_val.py --batch_size 4 --epochs 100 --lambda1 0.2 --lambda2 0.2
```

To see the full list of parameters, run
```bash
python3 ./train_with_val.py -h
```

Comment out anything related to wandb in the code if you do not want to use it to visualize the result.


## Citation
```bibtex
```
