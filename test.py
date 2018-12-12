from config import config
import numpy as np
from scipy import misc
import os
import tensorflow as tf
import glob
from model import IDN
import utils
import skimage.color as sc

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
dataset = config.TEST.dataset
model_path = config.TEST.model_path
saved_path = config.TEST.save_path
scale = 2  # 2 | 3 | 4
rgb = False

def main():
    ## data
    print('Loading data...')
    test_hr_path = os.path.join('data/', dataset)
    if dataset == 'Set5':
        ext = '*.bmp'
    else:
        ext = '*.png'
    hr_paths = sorted(glob.glob(os.path.join(test_hr_path, ext)))

    ## model
    print('Loading model...')
    tensor_lr = tf.placeholder('float32', [1, None, None, 3], name='tensor_lr')
    tensor_b = tf.placeholder('float32', [1, None, None, 3], name='tensor_b')

    tensor_sr = IDN(tensor_lr, tensor_b, scale)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    ## result
    save_path = os.path.join(saved_path, dataset+'/x'+str(scale))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    psnr_score = 0
    for i, _ in enumerate(hr_paths):
        print('processing image %d' % (i+1))
        img_hr = utils.modcrop(misc.imread(hr_paths[i]), scale)
        img_lr = utils.downsample_fn(img_hr, scale=scale)
        img_b = utils.upsample_fn(img_lr, scale=scale)
        [lr, b] = utils.datatype([img_lr, img_b])
        lr = lr[np.newaxis, :, :, :]
        b = b[np.newaxis, :, :, :]
        [sr] = sess.run([tensor_sr], {tensor_lr: lr, tensor_b: b})
        sr = utils.quantize(np.squeeze(sr))
        img_sr = utils.shave(sr, scale)
        img_hr = utils.shave(img_hr, scale)
        if not rgb:
            img_pre = utils.quantize(sc.rgb2ycbcr(img_sr)[:, :, 0])
            img_label = utils.quantize(sc.rgb2ycbcr(img_hr)[:, :, 0])
        else:
            img_pre = img_sr
            img_label = img_hr
        psnr_score += utils.compute_psnr(img_pre, img_label)
        misc.imsave(os.path.join(save_path, os.path.basename(hr_paths[i])), sr)

    print('Average PSNR: %.4f' % (psnr_score / len(hr_paths)))
    print('Finish')

if __name__ == '__main__':
    main()