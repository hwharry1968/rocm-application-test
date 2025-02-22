# rocm-application-test

<font color="red" size=15><b>Rocm Application</b></font>
ROCm Version: 6.0.0
Ubuntu：22.04.03
Pytorch: 2.0.1

- [Intoduction](#intoduction)
- [Deploy ROCm](#deploy-rocm)
  -[Prerequisites](#prerequisites)
  -[ROCm Installation](#rocm-installation)
  -[Pytorch Installation for ROCm (optional)](#pytorch-installation-for-rocm-optional)
- [ROCm Examples (GitHub examples)](#rocm-examples-github-examples)
 - [Prerequisites](#prerequisites-1)
 - [Building the examples](#building-the-examples)
 - [Run the examples](#run-the-examples)
- [AI Tutorials](#ai-tutorials)
 - [Deep Learning summary](#deep-learning-summary)
 - [Machine Learning Notebooks](#machine-learning-notebooks)
- [AI Application](#ai-application)
 - [Inception V3 with PyTorch](#inception-v3-with-pytorch)
 - [PointNet with PyTorch](#pointnet-with-pytorch)
 - [Download data](#download-data)
 - [Training](#training)
 - [Inference Performance](#inference-performance)
- [AIGC Application](#aigc-application)
 - [LocalGPT](#localgpt)
 - [Environment Setup](#environment-setup)
 - [Run test](#run-test)
 - [ChatGLM-6B](#chatglm-6b)
 - [Environment Setup](#environment-setup-1)
 - [Download model](#download-model)
 - [Model deployment](#model-deployment)
 - [AI Painting: Stable-Diffusion](#ai-painting-stable-diffusion)
 - [Environment Setup](#environment-setup-2)
 - [Model deployment](#model-deployment-1)
 - [Model download](#model-download)


# Intoduction
ROCm is an open-source stack, composed primarily of open-source software, designed for graphics processing unit (GPU) computation. ROCm consists of a collection of drivers, development tools, and APIs that enable GPU programming from low-level kernel to end-user applications.

All the details can be found in [ROCm Documation](https://rocm.docs.amd.com/en/latest/what-is-rocm.html).

# Deploy ROCm
## Prerequisites
Confirm that your kernel version matches the [System requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

```shell
$ uname -m && cat /etc/*release
$ uname -srmv
```

## ROCm Installation
In this section, you should intall gpu drive and rocm libraries.

[Quick-start intall on Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html)
Ubuntu 22.04:
```bash
$ sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
# See prerequisites. Adding current user to Video and Render groups
$ sudo usermod -a -G render,video $LOGNAME
$ sudo apt update
$ sudo apt install amdgpu-dkms
$ sudo apt install rocm-hip-libraries
$ echo Please reboot system for all settings to take effect.
```

1. [Driver setup](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html#installation):

```shell
$ sudo apt update
$ wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
$ sudo apt install ./amdgpu-install_6.0.60000-1_all.deb
```

2. [ROCm installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html#installing-rocm-packages):
The upgrade procedure with the installer script is exactly the same as installing for first time use.
```bash
$ sudo amdgpu-install --usecase=rocm
```

3. Verification

```bash
$ rocm-smi 
$ rocminfo
$ clinfo 
```

## Pytorch Installation for ROCm (optional)
For documentation on how to install PyTorch please refer to the following [Pytorch Installation](#https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/3rd-party/pytorch-install.html#installing-pytorch-for-rocm).

Install torch, torchvision, and torchaudio, as specified in the [installation matrix](#https://pytorch.org/get-started/locally/).
[Return Content](#content)

# ROCm Examples ([GitHub examples](#https://github.com/amd/rocm-examples))
## Prerequisites
**Linux**:
 * CMake (at least version 3.21)
 * ROCm (at least version 5.x.x)

## Building the examples
**CMake**
```bash
$ git clone https://github.com/amd/rocm-examples.git
$ cd rocm-examples
$ cmake -S . -B build
$ cmake --build build
$ cmake --install build --prefix install
```

## Run the examples 
Find the output of building files in `rocm-examples/build/`.
you can excute the examples in corresponding file:
```bash
$ cd /rocm-examples/build/HIP-Basic/hello_world/
$ ./hip_hello_world
```


# AI Tutorials
This part contains notes and summaries on DeepLearning.ai and Machine Learning Notebooks. 

## Deep Learning summary
[DeepLearning.ai](#https://github.com/mbadry1/DeepLearning.ai-Summary/tree/master) contains five courses which can be taken on Coursera. The five courses titles are:

- Neural Networks and Deep Learning.
- Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization.
- Structuring Machine Learning Projects.
- Convolutional Neural Networks.
- Sequence Models.

In these courses, you will learn the foundations of Deep Learning, understand how to build neural networks, and learn how to lead successful machine learning projects.  

## [Machine Learning Notebooks](#https://github.com/dair-ai/ML-Notebooks/tree/main)
This repo (https://github.com/dair-ai/ML-Notebooks/tree/main) contains machine learning notebooks for different tasks and applications. Through these examples, you can quickly familiarize with the  process and key knowledge of Machine learning.

# AI Application 
## Inception V3 with PyTorch
You should install pytorch first [(_Pytorch installation_)](#pytorch-installation-for-rocm-optional)

1. [Evaluating a pre-trained model](#https://github.com/pytorch/hub/blob/master/pytorch_vision_inception_v3.md)
2. [Training Inception V3](#https://rocm.docs.amd.com/en/latest/conceptual/ai-pytorch-inception.html#training-inception-v3)

## PointNet with PyTorch
This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in pytorch. The code is in [PointNet.pytorch](#https://github.com/fxia22/pointnet.pytorch/tree/master).

Point cloud is an important type of geometric data structure. PointNet, provides a unified architecture for applications ranging from object classification, part segmentation, to scene semantic parsing.

### Download data
Clone code
```bash
$ git clone https://github.com/fxia22/pointnet.pytorch
$ cd pointnet.pytorch
$ pip install -e .
```

Download and build visualization tool
```bash
$ cd scripts
$ bash build.sh #build C++ code for visualization
$ bash download.sh #download dataset
```

### Training 
```bash
$ cd utils
$ python3 train_classification.py --dataset <dataset path> --nepoch=<number epochs> --dataset_type <modelnet40 | shapenet>
$ python3 train_segmentation.py --dataset <dataset path> --nepoch=<number epochs> 
```
Use `--feature_transform` to use feature transform.
Examples:
```bash 
$ nohup python3 train_classification.py --dataset ../shapenetcore_partanno_segmentation_benchmark_v0 --nepoch=200 --dataset_type shapenet
$ nohup python3 train_segmentation.py --dataset ../shapenetcore_partanno_segmentation_benchmark_v0 --nepoch=200
```
```nohup``` can execute code in the background if you need.


### Inference Performance
1. Classification performance
```bash
$ cd utils
$ python3 show_cls.py --model <model path>  
```

2. Segmentation performance
```bash
$ cd utils
$ python3 show_seg.py --model <model path> --dataset <dataset path>
```
Examples
```bash
$ cd utils
$ python3 show_cls.py --model ./cls/cls_model_50.pth
$ python3 show_seg.py --model ./seg/seg_model_Chair_5.pth --dataset ../shapenetcore_partanno_segmentation_benchmark_v0
```

# AIGC Application 
AIGC（Artificial Intelligence Generated Content).
AI utilizes its understanding, imagination, and creativity to create various contents based on specified needs and styles: **articles, short stories, reports, music, images, and even videos**. 
The emergence of AIGC has opened up a whole new world of creativity, providing people with countless possibilities.

There are some local and open-source AIGC applications, you can be assured that no data ever leaves your computer.
## LocalGPT
LocalGPT (https://github.com/PromtEngineer/localGPT) is an open-source initiative that allows you to converse with your documents without compromising your privacy. 
### Environment Setup
**Clone the repo using git:**
```bash
$ git clone https://github.com/PromtEngineer/localGPT.git
```
Note for PermissionError: run `sudo chmod 777 ./localGPT/` 
**Use Conda**
1. Install conda for virtual environment management. Create and activate a new virtual environment.
```bash
$ conda create -n localGPT python=3.10.0
$ conda activate localGPT
```
2. Install the dependencies using pip
```bash
$ pip install -r requirements.txt
```
3. Installing LLAMA-CPP :
    To install with hipBLAS / ROCm support for AMD cards, set the LLAMA_HIPBLAS=on environment variable before installing:
```
$ CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python
```

**Use Docker (alternative)**
As an alternative to Conda, you can use Docker
```bash
$ docker build -t localgpt .
$ docker run -it --mount src="$HOME/.cache",target=/root/.cache,type=bind --gpus=all localgpt
```

### Run test 
1. Put files in `SOURCE_DOCUMENTS` folder. 
    You can put multiple folders within the `SOURCE_DOCUMENTS` folder and the code will recursively read your files. LocalGPT currently supports the following file formats:  ".txt",".md",".py",".pdf",".csv",".xls",".xlsx",".docx",".doc".
2. Ingest data
    If you have cuda setup on your system
    ```bash
 $ python ingest.py
    ```
    To run on cpu
    ```bash
 $ python ingest.py --device_type cpu
    ```
    Note: ValueError: Dependencies for InstructorEmbedding not found.
    ```bash
 $ pip install InstructorEmbedding sentence_transformers huggingface-hub
    ```
3. Chat with your documents
    You can also specify the device type just like ingest.py
    ```bash
 $ python run_localGPT.py --device_type <device type>
    ```
  This will load the ingested vector store and embedding model. You will be presented with a prompt:
```
> Enter a query:
```

## ChatGLM-6B
ChatGLM-6B (https://github.com/Fanstuck/ChatGLM-6B) is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters. With the quantization technique, users can deploy locally on consumer-grade graphics cards (only 6GB of GPU memory is required at the INT4 quantization level).

### Environment Setup
**Clone the repo using git:**
```bash
$ git clone https://github.com/Fanstuck/ChatGLM-6B
$ cd ChatGLM-6B/
```
Note for PermissionError: run `sudo chmod 777 ChatGLM-6B/` 

**Use Conda**
1. Install conda for virtual environment management. Create and activate a new virtual environment.
```bash
$ conda create -n ChatGLM python=3.10.0
$ conda activate ChatGLM
```
2. Install the dependencies using pip in `ChatGLM-6B/`
```bash
$ pip install -r requirements.txt
```
### Download model
We directly download the pretrained model and deploy it on a personal PC.
- **Download from tsinghua cloud (alternative)**
    ```bash
 $ GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b
    ```
    Then manually download the files from [here](#https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/) and replace them in `THUDM/chatglm-6b`. You should manually create a new folder `THUDM/`, otherwise files will be downloaded repeatedly.
- **Download from Hugging Face Hub directly**
    Install [Git LFS](#https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=linux) first，and then: 
    ```bash
 $ git clone https://huggingface.co/THUDM/chatglm-6b
    ```

### Model deployment
- **CLI Demo**
    Run cli_demo.py in the repo:
    ```bash
 $ python cli_demo.py
    ```

- **WEB Demo**
    Install Gradio 
    ```bash
 pip install gradio==3.50.0 #latest version may have some problems
    ```
 1. run web_demo.py:
    ```bash
 python web_demo.py 
    ```
 2. streamlit based web_demo 
    This version can be smoother and more visually appealing.
    ```bash
 $ pip install streamlit
 $ pip install streamlit-chat
 $ streamlit run web_demo2.py --server.port 6006
    ```
 3. Run cli_demo.py in the repo:
    ```bash
 $ python cli_demo.py
    ```
## AI Painting: Stable-Diffusion

**Stable Diffusion** is an open-source machine learning model that can generate images from text, modify images based on text, or fill in details on low-resolution or low-detail images. It has been trained on billions of images and can produce results that are comparable to the ones you'd get from DALL-E 2 and MidJourney. It's developed by Stability AI and was first publicly released on August 22, 2022.

### Environment Setup

**Requirements:**
- Working version of docker on 64-bit **Linux** (Ubuntu 22.04). Need at least kernel 5.10 for AMD ROCm support.
- **AMD ROCm** modules drivers loaded. Recommended 6.0 to match current release.
- 12+ GB of system RAM recommended.  
- 8GB of VRAM to produce 512x512 images. 

```bash
$ sudo apt update && sudo apt install -y curl git vim ffmpeg gfortran libstdc++-12-dev cockpit openssh-server
```

**Use Conda**
1. Install conda for virtual environment management. Create and activate a new virtual environment.
```bash
$ conda create -n sd python=3.11.5 -y
$ conda activate sd
```
2. Intall torch on virtual environment
```bash
$ pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.7
```
Verify the torch installation and GPU import status
```bash
$ python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
$ python3 -c 'import torch; print(torch.cuda.is_available())'
$ python3 -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))"
$ python3 -m torch.utils.collect_env 
```

3. Install the dependencies using pip in `stable-diffusion-webui/`
```bash
$ cd ./stable-diffusion-webui
$ pip install -r requirements.txt
```

### Model deployment

```bash
$ python launch.py --listen
```
Accessing through local browser http://0.0.0.0:7860 Or other remote devices accessing IP:7860.

### Model download
Recommended model download websites:
- https://huggingface.co/models
- https://cyberes.github.io/stable-diffusion-models/
- https://civitai.com/
- https://www.liandange.com/models

If you want to install the `.ckpt` model or the ckpt model with the `.safesensors` suffix, please place it in the following directory: `~/stable-diffusion-webui/models/Stable-diffusion`.
