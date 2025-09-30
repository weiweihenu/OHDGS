# OHDGS:Opacity-guided Hierarchical Densification in Gaussian Splatting for Sparse-view 3D Reconstruction


## Training
Train FSGS on LLFF dataset with 3 views
``` 
python train.py  --source_path dataset/nerf_llff_data/trex --model_path output/trex --eval  --n_views 3 
```


Train FSGS on MipNeRF-360 dataset with 24 views
``` 
python train.py  --source_path dataset/mipnerf360/garden --model_path output/garden  --eval  --n_views 24 
```
