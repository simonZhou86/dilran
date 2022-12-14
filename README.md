# DILRAN for medical image fusion

University of Toronto CSC2529 Computational Imaging project.

''An attention-based Multi-Scale Feature Learning Framework for Multimodal Medical Image Fusion''

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

To evaluate the results, run
```bash
python3 ./inference.py
```
If you are using different model, you may have to modify a little bit of the code.


Comment out anything related to wandb in the code if you do not want to use it to visualize the result.


## Citation
```bibtex
@article{zhou2022attention,
  title={An Attention-based Multi-Scale Feature Learning Network for Multimodal Medical Image Fusion},
  author={Zhou, Meng and Xu, Xiaolan and Zhang, Yuxuan},
  journal={arXiv preprint arXiv:2212.04661},
  year={2022}
}
```
