#Vision_Based_UDA Implementation 

## Dataset

### Office-Home

Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

## Training

```python
# Office-Home
# The order: Ar-Cl	Ar-Pr	Ar-Rw	Cl-Ar	Cl-Pr	Cl-Rw	Pr-Ar	Pr-Cl	Pr-Rw	Rw-Ar	Rw-Cl	Rw-Pr
python train_image.py --gpu_id id --net vit_small_patch16_224 --dset office-home --test_interval 2000 --s_dset_path ../data/office-home/Real_World.txt --t_dset_path ../data/office-home/Product.txt
```

```python 
# various evalution
#12th layer blocked - for layerwise attention patch fool attack
!python eval.py \
    --gpu_id 0 \
    --net vit_small_patch16_224 \
    --dset office-home \
    --t_dset_path ../data/office-home/Art.txt \
    --model_path /content/VT-ADA/pytorch/snapshot/vit_pgd_test_run/iter_40000_model.pth.tar \
    --output_dir distance_adv_product
```

