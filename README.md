# Deep Learning-Based Medical Image Registration

## Project Approach

1. CT - CT Medical Image Registration
2. 간 Segmentation Mask를 기준으로 ROI Crop (현재 간의 가장 꼭대기 부분부터 200만큼 내려서 잘랐음, 거기서부터 200이 안되는 이미지는 crop 안됨)
3. 혈관이나 장기의 크기가 바뀌면 안되기 때문에 Affine transformation에서 Scaling Factor를 제외한 Matrix를 뽑아내는 것이 최종 출력
4. 학습에 쓰이는 Loss는 Matrix로 moving image를 warping하고 fixed image와 비교해서 계산(MSE, NCC, NMI)
5. Moving Image : Artery / Fixed Image : Vein

## Code Frame

Christodoulidis Stergios et al. “*Linear and Deformable Image Registration with 3D Convolutional Neural Networks”, 2018 MICCAI*

[https://github.com/kumarshreshtha/image-registration-cnn](https://github.com/kumarshreshtha/image-registration-cnn)

<aside>
💡 원래 해당 Repository는 2019년이 마지막 commit이었고 제대로 구현하지 못해서 참고만 하라고 쓰여있었는데 22년 8월 25일 기준 3일 전에 commit을 했고 제대로 implementation을 한 것 같습니다…. 추가로 개발할 때는 이 Repo를 좀 더 적극적으로 참고하셔도 좋을 것 같습니다..ㅠ

</aside>


## Requirements

- Python : 3.8.12
- Torch : 1.11.0a0+17540c5
- wandb : 0.12.16
- MONAI : 0.8.1+181.ga676e387

## Experiment Configurations

- Learning Rate : 0.0001
- GPU : NVIDIA Titan XP (x8, 12GB)
- Batch size : 8 (one data at one GPU)
- Visualization : Weight and Bias


## Code Explanation

> 대부분 주석으로 설명을 써 놓긴 했지만 자세하게 써 놓진 못해서...
> 

### 1. Utils

- data_MONAI.py : Data path를 정의하고 해당 path를 통해 dictionary로 만들어주는 함수와 transformation을 정의해주는 함수가 있다.
- ddp.py : Argument Parser가 정의되어 있다.
- earlystopping.py : Frame에서 Clone할 때부터 있던 파일인데, Val loss가 전보다 내려가지 않는 현상이 여러 번 지속되면 학습을 종료 시켜버리는 기능과 checkpoint를 만들어 model을 save 해주는 기능이 있습니다. 저는 checkpoint가 너무 많으면 용량이 커져서 5 epoch마다 저장 시키는 것으로 바꿨다.

### 2. Models

- encoder.py : concat되어 들어온 moving과 fixed image를 5개의 3D conv layer로 연산해서 feature map을 만들어 주는 module
- affine_decoder.py : Encoder의 feature map을 6 DoF로 출력해주는 module
- deform_decoder.py : Deformation을 예측하기 위한 Decoder이다. SqueezeExcitation과 3D conv로 이루어져 있는데 코드는 전혀 건들지 않았다.
- register_3d.py : 전체 module을 하나로 합친 Class이다.

### 3. Metrics

- Github에서 implement되어 있는 여러 registration metric을 모아둔 파일인데, 거의 DDF를 위한 loss들이라(혹은 segmentation label을 같이 넣어주는) 당장 적용 가능한 것은 몇 개 없다. MIND_loss와 MILoss를 주로 사용했다.

### 4. .out

- preprocess.out : Liver segmentation mask를 이용해 crop할 때의 로그를 남겨둔 .out 파일이다.  mid1과 mid2가 각각 move와 fix의 liver segmention의 꼭대기 z slice이다. (여기서 200만큼 밑으로 잘랐음.)

### 5. Preprocess.py

- 위에서 설명한 Crop Preprocessing을 진행한 코드이다. MONAI의 SpatialCrop 함수를 사용해서 roi의 범위를 지정해주어 잘랐다.

### 6. Train.py

- Block처럼 나누어서 주석을 달아 놨습니다.