# Google Summer of code application

## Project Title
* Package LLM Inference Libraries

## Personal Information
- Name: Kohei Sendai 
- Email: kouhei.sendai@gmail.com
- GitHub: [k1000dai](https://github.com/k1000dai)
- University: The University of Tokyo
- Major: Engineering, system innovation
- Year: 3rd grade undergraduate student
- Country: Japan

## Background
 I am a third grade undergraduate student of the University of Tokyo. Now, I am studing computer science at KTH in Sweden as an exchange student. I have used Debian family and am familiar with Linux OS. Also, I worked for AI Startup Company for two years and I do a part-time job at a Matsuo Laboratory which did research related to AI, so I have a good experience with Python, especially PyTorch and another AI-related library. I'm really passioate 

## Application Tasks
### 1. Setup development environment
At first, I try to set up the development environment. To understand the Debian packaging system and to more familiar with the Debian system, I try to build by sbuild on my local machine. 
I mainly use m1mac book air so, I use docker container and use debian image. 

I used the following command to build the package.
1. Create a docker container
```bash
docker run -it --platform linux/amd64 --privileged --name=debian-amd64 debian:latest
```
2. Install the necessary package
```bash
apt update
apt install sbuild debootstrap schroot build-essential
```
3. Create a chroot environment
```bash
sudo sbuild-createchroot --arch=amd64 buster /srv/chroot/buster-amd64 http://deb.debian.org/debian
```
4. get the source code

4.1 modify the sources.list

Add the following line to /etc/apt/sources.list
```bash
deb http://deb.debian.org/debian buster main contrib non-free
deb-src http://deb.debian.org/debian buster main contrib non-free
```

4.2 get the source code

```bash
apt update
apt source <package-name>
```

I select vim for testing the package. 

5. Build the package
```bash
sbuild -d buster -c buster-amd64-sbuild <package-name>.dsc
```

6. test the package 
I create the another docker container and try to install the package. 
```bash 
dpkg -i <package-name>.deb
apt-get install -f
```

### 2. Analyze how PyTorch is packaged in Debian, including how the CUDA variant of PyTorch is prepared

I try to read the code of the debian packaging of pytorch here(https://salsa.debian.org/deeplearning-team/pytorch).

I found that they have some specific file for cuda variant.
Such as 
- debian/cudabuild.sh
- debian/control.cuda  
And in the debian/rules file, they have the following if statement to check the cuda variant.
```bash
ifneq (,$(shell command -v nvcc))
```
By checking the nvcc, they can check the cuda variant.

At first, cudabuild.sh is used to set up the environment for the cuda variant. 
In this process control.cuda file link to the control file and decide the cuda variant dependency. It includes such as  
* nvidia-cuda-toolkit,
* nvidia-cuda-toolkit-gcc,
* nvidia-cudnn (>= 8.7.0.84~cuda11.8),
* libcudnn-frontend-dev
which is really important packages for using cuda.

And, in the debian/rules file, nvcc command is used to check the cuda variant. If the nvcc command is available, it means that the cuda variant is available.

Also, through this process, I learned the idea of Debian Free Software Guideline(DFSG). 

### 3. setup vLLM locally using pip or uv

#### 3.1 setup vLLM using uv and run the test code
I mainly use m1mac book air so, I don't have a GPU. So, I try to use vLLM in the Google Cloud Platform. I created the instance with T4 GPU and try to install the vLLM in the instance.

The instance configuration is as follows.
- Machine type: n1-highmem-2 (2 vCPUs, 13 GB Memory)
- GPU: NVIDIA Tesla T4
- OS: Debian 11, Python 3.10. With CUDA 12.4 preinstalled.

I mainly use uv for the installation. 
```bash
uv init -p 3.12 local_vllm
cd local_vllm
uv add vllm
```

After the installation, I try to run the vLLM with the following code.
```python
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

And I got the following output. (some of the output is omitted because of the length)
```bash
Prompt: 'Hello, my name is', Generated text: " Sherry and I'm a 35-year-old woman from"
Prompt: 'The president of the United States is', Generated text: ' elected through the Electoral College, not through a direct popular vote. In the'
Prompt: 'The capital of France is', Generated text: ' Paris. The city is known for its rich history, cultural heritage, and'
Prompt: 'The future of AI is', Generated text: ' bright and exciting, with many potential applications across various industries. Here are'
```

#### 3.2 some of the issues I faced

1. I faced the issue with the installation of the vllm on my laptop. They only offer Pre-built wheels for the nvidia GPUs and we need to build from source. If we need to build for multiplatfrom, it seems to have some challenges.

2. Cuda version is also the important. The vllm provide vLLM binaries compiled with CUDA 12.1, 11.8, and public PyTorch release versions so we need to think about this compatibility.

### 4. Analyze the dependency tree of transformers package

I try to analyze the dependency tree of the transformers package. I mainly use the pipdeptree package to analyze the dependency tree. 

The raw output is as follows.
```bash
transformers==4.50.0.dev0
├── filelock [required: Any, installed: 3.17.0]
├── huggingface-hub [required: >=0.26.0,<1.0, installed: 0.29.3]
│   ├── filelock [required: Any, installed: 3.17.0]
│   ├── fsspec [required: >=2023.5.0, installed: 2025.3.0]
│   ├── packaging [required: >=20.9, installed: 24.2]
│   ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   ├── requests [required: Any, installed: 2.32.3]
│   │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   ├── tqdm [required: >=4.42.1, installed: 4.67.1]
│   └── typing_extensions [required: >=3.7.4.3, installed: 4.12.2]
├── numpy [required: >=1.17, installed: 2.2.3]
├── packaging [required: >=20.0, installed: 24.2]
├── PyYAML [required: >=5.1, installed: 6.0.2]
├── regex [required: !=2019.12.17, installed: 2024.11.6]
├── requests [required: Any, installed: 2.32.3]
│   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   ├── idna [required: >=2.5,<4, installed: 3.10]
│   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
├── tokenizers [required: >=0.21,<0.22, installed: 0.21.1]
│   └── huggingface-hub [required: >=0.16.4,<1.0, installed: 0.29.3]
│       ├── filelock [required: Any, installed: 3.17.0]
│       ├── fsspec [required: >=2023.5.0, installed: 2025.3.0]
│       ├── packaging [required: >=20.9, installed: 24.2]
│       ├── PyYAML [required: >=5.1, installed: 6.0.2]
│       ├── requests [required: Any, installed: 2.32.3]
│       │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│       │   ├── idna [required: >=2.5,<4, installed: 3.10]
│       │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│       │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│       ├── tqdm [required: >=4.42.1, installed: 4.67.1]
│       └── typing_extensions [required: >=3.7.4.3, installed: 4.12.2]
├── safetensors [required: >=0.4.1, installed: 0.5.3]
└── tqdm [required: >=4.27, installed: 4.67.1]
```

However, most of the dependencies are already included in the Debian repository. 

python3-filelock[https://packages.debian.org/sid/python3-filelock]

python3-numpy[https://packages.debian.org/sid/python3-numpy]

python3-packaging[https://packages.debian.org/sid/python3-packaging]

python3-pyyaml[https://packages.debian.org/sid/python3-yaml]

python3-regex[https://packages.debian.org/sid/python3-regex]

python3-requests[https://packages.debian.org/sid/python3-requests]

python3-tqdm[https://packages.debian.org/sid/python3-tqdm]

python3-safetensors[https://packages.debian.org/sid/python3-safetensors]

Then, the remaining dependencies are as follows.
```bash
transformers==4.50.0.dev0
├── huggingface-hub [required: >=0.26.0,<1.0, installed: 0.29.3]
├── tokenizers [required: >=0.21,<0.22, installed: 0.21.1]
```

I already create a package for huggingface-hub debian package in the next section, so the remaining package is tokenizers.

And tokenizers only has the huggingface-hub dependencies.
```bash
tokenizers==0.21.1
└── huggingface-hub [required: >=0.16.4,<1.0, installed: 0.29.3]
    ├── filelock [required: Any, installed: 3.17.0]
    ├── fsspec [required: >=2023.5.0, installed: 2025.3.0]
    ├── packaging [required: >=20.9, installed: 24.2]
    ├── PyYAML [required: >=5.1, installed: 6.0.2]
    ├── requests [required: Any, installed: 2.32.3]
    │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
    │   ├── idna [required: >=2.5,<4, installed: 3.10]
    │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
    │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
    ├── tqdm [required: >=4.42.1, installed: 4.67.1]
    └── typing_extensions [required: >=3.7.4.3, installed: 4.12.2]
```
So, after creating the huggingface-hub package, I think tokenizers package can be installed without any problem. Then, I can create the transformers package.

### 5. Create a Debian package for huggingface-hub

The source code of the huggingface-hub is here(https://github.com/huggingface/huggingface_hub).

And I create the debian package for huggingface-hub here(
https://github.com/k1000dai/python3-huggingface-hub).

huggingface-hub only has the following dependencies.
```bash
huggingface-hub==0.29.3
├── filelock [required: Any, installed: 3.17.0]
├── fsspec [required: >=2023.5.0, installed: 2025.3.0]
├── packaging [required: >=20.9, installed: 24.2]
├── PyYAML [required: >=5.1, installed: 6.0.2]
├── requests [required: Any, installed: 2.32.3]
│   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   ├── idna [required: >=2.5,<4, installed: 3.10]
│   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
├── tqdm [required: >=4.42.1, installed: 4.67.1]
└── typing_extensions [required: >=3.7.4.3, installed: 4.12.2]
```
and every dependency is already included in the Debian repository. So, I just modify the debian/control file and build the package with sbuild.

I checked that following code is working.
```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="tiiuae/falcon-7b-instruct", filename="config.json")
```

## Project Details
### Implementation Plan

~ 5/8 : Analyze the dependency tree of the vllm package. Setup the development environment and add every dependency which already included in the Debian repository.

5/8 ~ 6/1 : Starts with some of the packages which are not included in the Debian repository. I will start with the architecture-independent package and non cuda variant. I already create the package for huggingface-hub so I think this is a good start and don't take much time. I will get the feedback from the mentor and continue to the next step.

6/1 ~ 7/18 : Tackle the cuda variant. I will try to build the vLLM package with cuda variant on my local machine. Some packages which are not included in the Debian repository are needed and required for the cuda variant. I will try to create the package for the required package and build the vLLM package with cuda variant. Now, I already checked that vllm works on CUDA 12.4 so I will try to build the package with CUDA 12.4. I will get the feedback from the mentor and continue to the next step.

7/18 : Mid-term evaluation

7/19 ~ 7/30 : Try to build the vLLM package with multiplatform. I will try to build the package with different CUDA version and different architecture. I will get the feedback from the mentor and continue to the next step.

8/1 ~ 8/25 : I understand project always have some issues and need more time than expected. So, I will try to finish the project as soon as possible and try to fix the issue. I will finish the project and get the feedback from the mentor.

## Availability
My school finishes at the end of May so I can start the project from the beginning of June. I will go back to Japan after school ends. University of Tokyo starts at October so I can work full-time on the project during the summer. 
