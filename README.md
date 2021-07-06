# cil_road_segmentation 

Train and predict:

- python main.py -c config/unet.json

Quick test train and predict

- python main.py -c config/unet.json -t 


On leonhard:
- bsub -n 16 -W 2:00 -o unet.%J -R "rusage[mem=1024, ngpus_excl_p=1]" -J unet -B -N python main.py -c config/unet.json 
