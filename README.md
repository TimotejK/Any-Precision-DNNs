This project combines code from [Any-precision](https://github.com/SHI-Labs/Any-Precision-DNNs) and [Slimmable neural networks](https://github.com/JiahuiYu/slimmable_networks) repositories.

To run the tests you have to save data from [Human Activity Recognition Using Smartphones Data Set](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) to data subfolder.

Example command for training slimmable mobilenet:
```
python3 train.py --model slimmableMobileNetV2 --dataset activityRecognition --train_split train --lr 0.001 --lr_decay "100,140,180" --epochs 200 --optimizer adam --weight-decay 1e-2 --results-dir results/mobilenet_slimmable --bit_width_list "0.35,0.5,0.75,1.0"
```

Example command for training quantized mobilenet:
```
python3 train.py --model mobileNetV2s --dataset activityRecognition --train_split train --lr 0.001 --lr_decay "50,70,90" --epochs 100 --optimizer adam --weight-decay 1e-2 --results-dir results/mobilenet --bit_width_list "1,2,4,8,32"
```

Example command for training quantized resnet-50:
```
python3 train.py --model resnet50q --dataset activityRecognition --train_split train --lr 0.001 --lr_decay "50,70,90" --epochs 100 --optimizer adam --weight-decay 1e-2 --results-dir results/activityRecognition50 --bit_width_list "1,2,4,8,32"
```

Example command for testing quantization selection:
```
python3 test_optimization_level_selection.py --model mobileNetV2s --dataset activityRecognition --train_split test --lr 0.001 --lr_decay "100,150,180" --epochs 1 --batch-size 256 --pretrain results/mobilenet/ckpt/model_latest.pth.tar --optimizer adam --weight-decay 1e-50 --results-dir results/mobilenet --bit_width_list "1,2,4,8,32"
```
