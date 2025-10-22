# AdaGS

**Adaptive Gaussian Splatting** combines 3D and 4D Gaussians to model dynamic scenes efficiently.


## 📦 Installation

```shell
git clone https://github.com/sudarshan-iyengar/adags.git
cd adags
conda env create --file environment.yml
conda activate adags
```

## 📁 Data preparation
### Neural 3D Video Dataset
Download the dataset [here](https://github.com/facebookresearch/Neural_3D_Video).
After downloading the data, preprocess it using:
```shell
python scripts/n3v2blender.py $path_to_dataset
```

## 🏃‍♂️ Training
Single sequence training:
```shell
python main.py --config configs/n3v/default.yaml --model_path <model save path> --source_path <dataset path>
```
Train all sequences:
```shell
bash train.sh
```
Don't forget to adjust dataset paths in train.sh.

## 🧪 Testing / Evaluation

```shell
python main.py --config configs/n3v/default.yaml --model_path <model path> --source_path <dataset path> --start_checkpoint <model_path>/chkpnt6000.pth --val
```

## 🙏 Acknowledgement
This project builds upon:
- [Hybrid 3D-4DGS](https://github.com/ohsngjun/3D-4DGS)
- [Real-time 4D Gaussian Splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)
- [Ex4DGS](https://github.com/juno181/Ex4DGS)
- [4D-Rotor Gaussians](https://github.com/weify627/4D-Rotor-Gaussians) (data preprocessing)
- [@sorceressyidi](https://github.com/sorceressyidi) (visualization code)


```
