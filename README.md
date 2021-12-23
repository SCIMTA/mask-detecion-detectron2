# Mask detection using Detectron2 - base on DatacompFPT 2021

### Project: Neural network from scratch
- Dev by [SCIMTA](https://github.com/SCIMTA) EX-team:
    - [trongthanht3](https://github.com/trongthanht3)  
    - [minhk17khmt](https://github.com/minhkids)  
    - [tienduongnguyen](https://github.com/tienduongnguyen)  
  
- Thanks to: 
    - Mentor: [Ms. Phan Thi Hai Hong - HVKTQS](haihonglyk6@yahoo.com)  
    - [DGMaxime](https://github.com/DGMaxime) for [detectron2-windows](https://github.com/DGMaxime/detectron2-windows)
    
- Reference:
    - [Detectron2](https://github.com/facebookresearch/detectron2)
    - [Road Damage Detection and Classification with Detectron2 and Faster R-CNN](https://ieeexplore.ieee.org/document/9378027)
    - [Mask detection using detectron2](https://medium.com/mlearning-ai/mask-detection-using-detectron2-225383c7d069)
    
# How to run this???
```angular2html
  # Clone source code
  git clone https://github.com/trongthanht3/mask-detecion-detectron2
```  
Put this [model](https://drive.google.com/file/d/13wmkXhTXZxMoxKuELQcDT9W2XymeSlXa/view?usp=sharing) to `model/`

```
  cd mask-detecion-detectron2  
  pip install -r requirements.txt
  
  git clone https://github.com/DGMaxime/detectron2-windows
  cd detectron2-windows
  pip install -e .
  cd ..
  
  python main_windows.py
  ---- click Start ----  
```