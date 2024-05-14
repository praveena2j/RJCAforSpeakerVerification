In this work, we present Recursive fusion of Joint Cross-Attention across audio and visual modalities for person verification. 

## References
If you find this work useful in your research, please consider citing our work :pencil: and giving a star :star2: :
```bibtex
@article{praveen2024audio,
  title={Audio-Visual Person Verification based on Recursive Fusion of Joint Cross-Attention},
  author={Praveen, R Gnana and Alam, Jahangir},
  journal={arXiv preprint arXiv:2403.04654},
  year={2024}
}
```

There are three major blocks in this repository to reproduce the results of our paper. This code uses Mixed Precision Training (torch.cuda.amp). The dependencies and packages required to reproduce the environment of this repository can be found in the `environment.yml` file. 

### Creating the environment
Create an environment using the `environment.yml` file

`conda env create -f environment.yml`

### Models and Text Files
The pre-trained models of audio and visual backbones are obtained [here](https://drive.google.com/drive/u/0/folders/1bXyexxgspeOi6gFiP177pM-KhwSeVsTq) 

The fusion models trained using our fusion approach can be found [here](https://drive.google.com/drive/u/0/my-drive?q=after:2024-05-13%20parent:0AF1R2CTIrvG8Uk9PVA)

The text files can be found [here](https://drive.google.com/drive/u/0/folders/1NJicFlj9CeNzxvtrOHRIHy6HnoTszro7)
```
train_list :  Train list
val_trials :  Validation trials list
val_list : Validation list
test_trials : VoX1-O trials list
test_list : Vox 1-O list

```


# Table of contents <a name="Table_of_Content"></a>

+ [Preprocessing](#DP) 
    + [Step One: Download the dataset](#PD)
    + [Step Two: Preprocess the visual modality](#PV) 
+ [Training](#Training) 
    + [Training the fusion model](#TE) 
+ [Inference](#R)
    + [Generating the results](#GR)
 
## Preprocessing <a name="DP"></a>
[Return to Table of Content](#Table_of_Content)

### Step One: Download the dataset <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)
Please download the following.
  + The images of Voxceleb1 dataset can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/) 

### Step Two: Preprocess the visual modality <a name="PV"></a>
[Return to Table of Content](#Table_of_Content)
  + The downloaded images are not properly aligned. So the images are aligned using [Insightface](https://github.com/TadasBaltrusaitis/OpenFace/releases). 

## Training <a name="TE"></a>
[Return to Table of Content](#Table_of_Content)
  + sbatch run_train.sh 

## Inference <a name="GR"></a>
[Return to Table of Content](#Table_of_Content)
  + sbatch run_eval.sh



### 👍 Acknowledgments
Our code is based on [AVCleanse](https://github.com/TaoRuijie/AVCleanse)
