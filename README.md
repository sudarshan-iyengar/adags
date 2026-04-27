# AdaGS

**Adaptive Gaussian Splatting** combines 3D and 4D Gaussians to model dynamic scenes efficiently.


## 📦 Installation

```shell
git clone https://github.com/sudarshan-iyengar/adags.git
cd adags
conda env create --file environment.yml
conda activate adags
pip install wandb
```

`weave` is not required for experiment tracking in this repo. `wandb` is sufficient for logging runs, metrics, and configs.

## W&B authentication

Use `WANDB_API_KEY` from your shell or job environment instead of putting secrets in code or config files.

Linux/macOS:
```shell
export WANDB_API_KEY=your_api_key
```

PowerShell:
```powershell
$env:WANDB_API_KEY="your_api_key"
```

For local dry runs without an API key, use `--wandb_mode offline`.

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

Single sequence training with W&B:
```shell
python main.py --config configs/n3v/default.yaml --model_path <model save path> --source_path <dataset path> --use_wandb --wandb_project adags
```

Train all sequences:
```shell
bash train.sh
```
Don't forget to adjust dataset paths in train.sh.

You can also place the W&B settings in your YAML config as top-level fields:
```yaml
use_wandb: true
wandb_project: adags
wandb_group: n3v
wandb_tags: [baseline, n3v]
wandb_mode: online
```

## 🧪 Testing / Evaluation

```shell
python main.py --config configs/n3v/default.yaml --model_path <model path> --source_path <dataset path> --start_checkpoint <model_path>/chkpnt6000.pth --val
```

## HPC usage

For HPC jobs, inject `WANDB_API_KEY` through the scheduler or container environment instead of storing it in the repo. For example, in a job/container spec:

```yaml
environment:
  environment_variables:
    - WANDB_API_KEY=${WANDB_API_KEY}
```

If outbound connectivity is restricted on the cluster, run with `--wandb_mode offline` and sync later from the run directory.

## 🙏 Acknowledgement
This project builds upon:
- [Hybrid 3D-4DGS](https://github.com/ohsngjun/3D-4DGS)
- [Real-time 4D Gaussian Splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)
- [Ex4DGS](https://github.com/juno181/Ex4DGS)
- [4D-Rotor Gaussians](https://github.com/weify627/4D-Rotor-Gaussians) (data preprocessing)
- [@sorceressyidi](https://github.com/sorceressyidi) (visualization code)


```
