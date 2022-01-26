**MDef-DETR minus Language**
========

The repository contains the training code of MDef-DETR model with language branch removed. 
The paper is available on [arxiv](https://arxiv.org/abs/2111.11430).

## Requirements

```
pip install -r requirements.txt
```

## Training

Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):
```
pip install submitit
```

The config file for pretraining is configs/pretrain.json and looks like:

```json
{
    "combine_datasets": ["flickr", "mixed"],
    "combine_datasets_val": ["flickr", "gqa", "refexp"],
    "coco_path": "",
    "vg_img_path": "",
    "flickr_img_path": "",
    "refexp_ann_path": "annotations/",
    "flickr_ann_path": "annotations/",
    "gqa_ann_path": "annotations/",
    "refexp_dataset_name": "all",
    "GT_type": "separate",
    "flickr_dataset_path": ""
}
```

* Download the original Flickr30k image dataset from : [Flickr30K webpage](http://shannon.cs.illinois.edu/DenotationGraph/) and update the `flickr_img_path` to the folder containing the images.
* Download the original Flickr30k entities annotations from: [Flickr30k annotations](https://github.com/BryanPlummer/flickr30k_entities) and update the `flickr_dataset_path` to the folder with annotations.
* Download the gqa images at [GQA images](https://nlp.stanford.edu/data/gqa/images.zip) and update `vg_img_path` to point to the folder containing the images.
* Download COCO images [Coco train2014](http://images.cocodataset.org/zips/train2014.zip). Update the `coco_path` to the folder containing the downloaded images.
* Download pre-processed annotations that are converted to coco format (all datasets present in the same zip folder for MDETR annotations): [Pre-processed annotations](https://zenodo.org/record/4729015/files/mdetr_annotations.tar.gz?download=1) and update the `flickr_ann_path`, `gqa_ann_path` and `refexp_ann_path` to this folder with pre-processed annotations.

##### Alternatively, you can download the preprocessed data from [link]() as a single zip file and extract it under 'data' directory. 

## Script to run training

This command will reproduce the training of the resnet 101.
```
python run_with_submitit.py --dataset_config configs/pretrain.json  --ngpus 8 --nodes 4 --ema --epochs 10
```

## Citation 
If you use our work, please consider citing MDef-DETR:
```bibtex
    @article{Maaz2021Multimodal,
        title={Multi-modal Transformers Excel at Class-agnostic Object Detection},
        author={Muhammad Maaz and Hanoona Rasheed and Salman Khan and Fahad Shahbaz Khan and Rao Muhammad Anwer and Ming-Hsuan Yang},
        journal={ArXiv 2111.11430},
        year={2021}
    }
```

## Credits
This codebase is modified from the [MDETR repository](https://github.com/ashkamath/mdetr). 
We thank them for their implementation.