# DILRAN for medical image fusion

''An attention-based Multi-Scale Feature Learning Framework for Multimodal Medical Image Fusion'' [Paper link](https://arxiv.org/pdf/2212.04661.pdf)

The extended version of this work is accepted by IEEE BIBM 2024. The code is based on this repo and will be released at [Here](https://github.com/simonZhou86/en_dran/tree/main)
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
If you are using a different model, you may have to modify a little bit of the code.


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
