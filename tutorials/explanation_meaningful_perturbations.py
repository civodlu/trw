import torch
from torch.autograd import Variable
from torchvision import models
import numpy as np
import matplotlib as plt
import trw.train
from PIL import Image
import os
import functools


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()

    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    preprocessed_img_tensor = torch.from_numpy(preprocessed_img)
    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=False)


def load_model():
    model = models.vgg19(pretrained=True)

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False
    return model


def export_fn(iter, feature_name, perturbated_feature_value, mask):
    Image.fromarray(np.uint8(255 * plt.cm.jet(1 - trw.train.to_value(mask)[0, 0]))).save('{}mask_color_{}.png'.format(export_path, iter))


if __name__ == '__main__':
    img_path = r'D:\package_trw\scripts\trw\tutorials\input_images\cat_dog.png'
    export_path = 'c:/tmp/'

    torch.manual_seed(0)
    model = load_model()
    information_removal_fn = functools.partial(trw.train.default_information_removal_smoothing, blurring_sigma=5)
    perturbation = trw.train.MeaningfulPerturbation(model=model, iterations=400, l1_coeff=0.05, export_fn=export_fn, information_removal_fn=information_removal_fn)

    original_img = Image.open(img_path).convert('RGB')
    original_img = original_img.resize((224, 224))
    img = np.float32(original_img) / 255
    img = preprocess_image(img)

    # target = 281 -> tiger cat
    # target =     ->
    #output_name, r = perturbation(img, target_class_name=None, target_class=281)
    output_name, r = perturbation(img, target_class_name=None)

    # export the mask
    Image.fromarray(np.uint8(r['input']['mask'][0, 0] * 255)).save(os.path.join(export_path, 'mask.png'))
    original_img.save(os.path.join(export_path, 'image.png'))

    # export the masked image
    rgb_mask = (plt.cm.jet(r['input']['mask'][0, 0]) * 255)[:, :, 0:3]
    masked_image = np.float32(np.array(original_img)) + rgb_mask
    masked_image_ui = np.uint8(masked_image / np.max(masked_image) * 255)
    Image.fromarray(masked_image_ui).save('c:/tmp/masked_image.png')