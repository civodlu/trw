import os
import functools
import collections
from PIL import Image
import torchvision
import trw
import xml.etree.ElementTree as ET
from . import utils


def _load_image_and_mask(batch, transform, normalize_0_1=True):
    images = []
    masks = []
    for image_path, mask_path in zip(batch['images'], batch['masks']):
        image = utils.pic_to_tensor(Image.open(image_path).convert('RGB'))
        if normalize_0_1:
            image = image.float() / 255.0

        images.append(image)

        mask = utils.pic_to_tensor(Image.open(mask_path))
        masks.append(mask)

    batch = {
        'images': trw.transforms.stack(images),
        'masks': trw.transforms.stack(masks),
    }

    if transform is not None:
        batch = transform(batch)

    return batch


def _parse_voc_xml(node):
    """
    Extracted from torchvision
    """
    voc_dict = {}
    children = list(node)
    if children:
        def_dic = collections.defaultdict(list)
        for dc in map(_parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == 'annotation':
            def_dic['object'] = [def_dic['object']]
        voc_dict = {
            node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
        }
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


def _load_image_and_bb(batch, transform, normalize_0_1=True):
    images = []
    annotations = []

    for image_path, annotation_path in zip(batch['images'], batch['annotations']):
        image = utils.pic_to_tensor(Image.open(image_path).convert('RGB'))
        if normalize_0_1:
            image = image.float() / 255.0
        images.append(image)

        annotation = _parse_voc_xml(ET.parse(annotation_path).getroot())
        annotations.append(annotation)

    batch = {
        'image_path': image_path,
        'sample_uid': batch['sample_uid'],
        'images': trw.transforms.stack(images),
        'annotations': annotations
    }

    if transform is not None:
        batch = transform(batch)

    return batch


def default_voc_transforms():
    criteria_images = functools.partial(trw.transforms.criteria_feature_name, feature_names=['images'])
    return trw.transforms.TransformCompose([
        trw.transforms.TransformResize(size=[250, 250]),
        #trw.transforms.TransformRandomCropPad(feature_names=['images', 'masks'], padding=None, shape=[3, 224, 224]),
        trw.transforms.TransformNormalizeIntensity(criteria_fn=criteria_images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_voc_segmentation_dataset(
        batch_size=40,
        root=None,
        transform_train=default_voc_transforms(),
        transform_valid=None,
        nb_workers=2,
        year='2012'):
    """
    Create the VOC segmentation dataset

    Args:
        batch_size: the number of samples per batch
        root: the root of the dataset
        transform_train: the transform to apply on each batch of data of the training data
        transform_valid: the transform to apply on each batch of data of the validation data
        nb_workers: the number of worker process to pre-process the batches

    Returns:
        a datasets with dataset `voc2012` and splits `train`, `valid`.
    """
    if root is None:
        # first, check if we have some environment variables configured
        root = os.environ.get('TRW_DATA_ROOT')

    if root is None:
        # else default a standard folder
        root = './data'

    path = os.path.join(root, f'VOC{year}')

    download = False
    try:
        # test if we have access to the data. If this fails, it means we need to download it!
        torchvision.datasets.VOCSegmentation(root=path, image_set='val', transform=None, download=False, year=year)
    except:
        download = True

    # train split

    # here batch_size = 1 since the images do not all have the same size, so we need to process them
    # independently. The `transform` should normalize the size (resize, central_crop) so that
    # the images can be batched subsequently
    train_dataset = torchvision.datasets.VOCSegmentation(
        root=path, image_set='train', transform=None, download=download, year=year)
    train_sequence = trw.train.SequenceArray({
        'images': train_dataset.images,
        'masks': train_dataset.masks,
    }, trw.train.SamplerRandom(batch_size=1))
    train_sequence = train_sequence.map(functools.partial(_load_image_and_mask, transform=transform_train), nb_workers=nb_workers, max_jobs_at_once=2 * nb_workers)
    if batch_size != 1:
        train_sequence = train_sequence.batch(batch_size)

    # valid split
    valid_dataset = torchvision.datasets.VOCSegmentation(
        root=path, image_set='train', transform=None, download=download, year=year)
    valid_sequence = trw.train.SequenceArray({
        'images': valid_dataset.images,
        'masks': valid_dataset.masks,
    }, trw.train.SamplerSequential(batch_size=1))
    valid_sequence = valid_sequence.map(functools.partial(_load_image_and_mask, transform=transform_valid), nb_workers=nb_workers, max_jobs_at_once=2 * nb_workers)
    if batch_size != 1:
        valid_sequence = valid_sequence.batch(batch_size)

    return {
        f'voc{year}_seg': collections.OrderedDict([
            ('train', train_sequence.collate()),
            ('valid', valid_sequence.collate()),
        ])
    }


def create_voc_detection_dataset(
        root=None,
        transform_train=None,
        transform_valid=None,
        nb_workers=2,
        year='2012'):
    """
    PASCAL VOC detection challenge

    Notes:
        - Batch size is always `1` since we need to sample from the image various anchors,
          locations depending on the task (so each sample should be postprocessed by a custom
          transform)
    """
    if root is None:
        # first, check if we have some environment variables configured
        root = os.environ.get('TRW_DATA_ROOT')

    if root is None:
        # else default a standard folder
        root = './data'

    path = os.path.join(root, f'VOC{year}')

    download = False
    try:
        # test if we have access to the data. If this fails, it means we need to download it!
        torchvision.datasets.VOCDetection(root=path, image_set='val', transform=None, download=False, year=year)
    except:
        download = True

    train_dataset = torchvision.datasets.VOCDetection(
        root=path, image_set='train', transform=None, download=download, year=year)
    train_sequence = trw.train.SequenceArray({
        'images': train_dataset.images,
        'annotations': train_dataset.annotations
    }, trw.train.SamplerRandom(batch_size=1))
    train_sequence = train_sequence.map(functools.partial(_load_image_and_bb, transform=transform_train),
                                        nb_workers=nb_workers, max_jobs_at_once=2 * nb_workers)

    # valid split
    valid_dataset = torchvision.datasets.VOCDetection(
        root=path, image_set='train', transform=None, download=download, year=year)
    valid_sequence = trw.train.SequenceArray({
        'images': valid_dataset.images,
        'annotations': valid_dataset.annotations,
    }, trw.train.SamplerSequential(batch_size=1))
    valid_sequence = valid_sequence.map(functools.partial(_load_image_and_mask, transform=transform_valid),
                                        nb_workers=nb_workers, max_jobs_at_once=2 * nb_workers)

    return {
        f'voc{year}_detect': collections.OrderedDict([
            ('train', train_sequence.collate()),
            ('valid', valid_sequence.collate()),
        ])
    }
