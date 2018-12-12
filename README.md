# IDN-tensorflow
[[Original Caffe version]](https://github.com/Zheng222/IDN-Caffe)
## Testing
* Install Tensorflow 1.11, Matlab R2017a
* Download [Test datasets](https://drive.google.com/open?id=1_K6mchwDGOQMIXuBIGrlDA4EAYgbtdmU)
* Modify `config.py` (if you want to test x3 model on Set14, `config.TEST.model_path = 'checkpoint_x3/model.ckpt'` `config.TEST.dataset = 'Set14'`) and `test.py` (`scale = 3`).
* Run testing:
```bash
python test.py
```

## Training
* Download [Training dataset](https://drive.google.com/open?id=12hOYsMa8t1ErKj6PZA352icsx9mz1TwB)
* Modify `config.py` (if you want to train x4 model, `config.TRAIN.hr_img_path = '/path/to/DIV2K_train_HR/'` `config.TRAIN.checkpoint_dir = 'checkpoint_x4/'` `config.VALID.hr_img_path = '/path/to/DIV2K_valid_HR/'` `config.VALID.lr_img_path = '/path/to/DIV2K_valid_LR_x4/'`) and `train_SR.py` (`scale = 4`)
* Run training:
```bash
python train_SR.py
```
## Note
This TensorFlow version is trained with DIV2K training dataset on RGB channels. Additionally, We modify the upsample layer to subpixel convolution (the original version is transposed convolution).

## Results
[Test_results](https://drive.google.com/open?id=1saFhGV8t2ytzRLHE2CaFc4H_UkvJo9KS)

The following PSNR/SSIMs are evaluated on Matlab R2017a and the code can be referred to [Evaluate_PSNR_SSIM.m](https://github.com/yulunzhang/RCAN/blob/master/RCAN_TestCode/Evaluate_PSNR_SSIM.m).

| <sub>Training dataset</sub> | <sub>Scale</sub> | <sub>Set5</sub> | <sub>Set14</sub> | <sub>B100</sub> | <sub>Urban100</sub> |
|:---:|:---:|:---:|:---:|:---:|:---:|
| <sub> 291 </sub> | <sub>×2</sub> | <sub>37.83 / 0.9600<sub> | <sub>33.30 / 0.9148</sub>|<sub>32.08 / 0.8985</sub>|<sub>31.27 / 0.9196</sub>|
| <sub> DIV2K </sub> | <sub>×2</sub> | <sub>37.85 / 0.9598<sub> | <sub>33.58 / 0.9178</sub>|<sub>32.11 / 0.8989</sub>|<sub>31.95 / 0.9266</sub>|
| <sub> 291 </sub> | <sub>×3</sub> | <sub>34.11 / 0.9253<sub> | <sub>29.99 / 0.8354</sub>|<sub>28.95 / 0.8013</sub>|<sub>27.42 / 0.8359</sub>|
| <sub> DIV2K </sub> | <sub>×3</sub> | <sub>34.24 / 0.9260<sub> | <sub>30.27 / 0.8408</sub>|<sub>29.03 / 0.8038</sub>|<sub>27.99 / 0.8489</sub>|
| <sub> 291 </sub> | <sub>×4</sub> | <sub>31.82 / 0.8903<sub> | <sub>28.25 / 0.7730</sub>|<sub>27.41 / 0.7297</sub>|<sub>25.41 / 0.7632</sub>|
| <sub> DIV2K </sub> | <sub>×4</sub> | <sub>31.99 / 0.8928<sub> | <sub>28.52 / 0.7794</sub>|<sub>27.52 / 0.7339</sub>|<sub>25.92 / 0.7801</sub>|
## Citation

If you find IDN useful in your research, please consider citing:

```
@inproceedings{Hui-IDN-2018,
  title={Fast and Accurate Single Image Super-Resolution via Information Distillation Network},
  author={Hui, Zheng and Wang, Xiumei and Gao, Xinbo},
  booktitle={CVPR},
  pages = {723--731},
  year={2018}
}
```
