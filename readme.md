# 1. prepare dataset

prepare shift dataset as follows:

```Bash
SHIFT2D/
	checkpoints/
    data/   
        pretrain_models/   
        shift/
            continuous/
            distrete/
    mmdet/
...
```



# 2. infer on continuous

单卡
```bash
python test.py my_shift_cfg/shift_dino_swins_1x.py checkpoints/iter_105792.pth
```
多卡
```bash
bash dist_test.sh my_shift_cfg/shift_dino_swins_1x.py checkpoints/iter_105792.pth
```

# 2. adapt on continuous

tent 单卡

```bash
python tools/tta_test.py my_shift_cfg/tent/shift_dino_swins_1x_continous_tent.py
```

tent 多卡

```bash
bash tools/tta_dist_test.py my_shift_cfg/tent/shift_dino_swins_1x_continous_tent.py 2
```

mean teacher 单卡

```bash
python tools/tta_test.py my_shift_cfg/meanteacher/shift_dino_swins_1x_continous_tent.py
```

mean teacher 多卡

```bash
bash tools/tta_dist_test.py my_shift_cfg/meanteacher/shift_dino_swins_1x_continous_tent.py 2
```



