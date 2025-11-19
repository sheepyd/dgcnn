# Dynamic Graph CNN for Learning on Point Clouds (PyTorch)

## Point Cloud Classification
* Run the training script:


``` 1024 points
python main.py --exp_name=dgcnn_1024 --model=dgcnn --num_points=1024 --k=20 --use_sgd=True
```

``` 2048 points
python main.py --exp_name=dgcnn_2048 --model=dgcnn --num_points=2048 --k=40 --use_sgd=True
```

* Run the evaluation script after training finished:

``` 1024 points
python main.py --exp_name=dgcnn_1024_eval --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=checkpoints/dgcnn_1024/models/model.t7
```

``` 2048 points
python main.py --exp_name=dgcnn_2048_eval --model=dgcnn --num_points=2048 --k=40 --use_sgd=True --eval=True --model_path=checkpoints/dgcnn_2048/models/model.t7
```

* Run the evaluation script with pretrained models:

``` 1024 points
python main.py --exp_name=dgcnn_1024_eval --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=pretrained/model.1024.t7
```

``` 2048 points
python main.py --exp_name=dgcnn_2048_eval --model=dgcnn --num_points=2048 --k=40 --use_sgd=True --eval=True --model_path=pretrained/model.2048.t7
```

## Tomato Stem/Leaf Segmentation
1. Place the processed scans described in `dataset/tomato/README.md` under `dataset/tomato` (or point `--tomato_root` elsewhere).
2. Launch segmentation training (example below trains on 2048 points and 3 classes: earth, stem, leaf):

```
python main.py --exp_name=tomato_seg --dataset=tomato --task=seg --model=dgcnn \
  --num_points=2048 --k=30 --batch_size=8 --test_batch_size=8 --epochs=200 \
  --num_classes=3 --ignore_label=0
```

The optional `--ignore_label` masks a class (e.g., earth=0) when computing the loss and metrics.

To evaluate a saved checkpoint on the validation split:

```
python main.py --exp_name=tomato_seg_eval --dataset=tomato --task=seg \
  --eval=True --model_path=checkpoints/tomato_seg/models/model.t7
```

To generate predictions on the processed test split (results saved under `predictions/test` by default):

```
python main.py --exp_name=tomato_seg_infer --dataset=tomato --task=seg \
  --predict --model_path=checkpoints/tomato_seg/models/model.t7 \
  --num_points=2048 --k=30 --num_classes=3
```
