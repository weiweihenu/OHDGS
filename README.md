# Opacity-Guided Hierarchical Densification: Enhancing Sparse-View 3D Reconstruction with Gaussian Splatting



## Environmental Setups
We provide install method based on Conda package and environment management:
```bash
conda env create --file environment.yml
conda activate OHDGS
```
**CUDA 11.7** is strongly recommended.




## Training
Train FSGS on LLFF dataset with 3 views
``` 
python train.py  --source_path dataset/nerf_llff_data/horns --model_path output/horns --eval  --n_views 3 
```


Train FSGS on MipNeRF-360 dataset with 24 views
``` 
python train.py  --source_path dataset/mipnerf360/garden --model_path output/garden  --eval  --n_views 24 
```



## Evaluation
You can just run the following script to evaluate the model.  

```
python metrics.py --source_path dataset/nerf_llff_data/horns/  --model_path  output/horns --iteration 10000
```

