# Google Summer of code : proposal

## Project Title
* Package LLM Inference Libraries

## Description of the project (cited from the project idea)
 Package Large Language Model (LLM) inference libraries, in particular vLLM. It is needless to explain how LLMs are important. Currently, in the Debian archive, we only have ?PyTorch, but downstream applications are still missing. One of the most promising downstream applications is LLM inference. There are already people working on llama.cpp and Ollama, but vLLM still lacks lots of dependencies to land onto Debian. For multi-GPU inference and concurrency, vLLM has its advantages over llama.cpp. The missing packages are, for instance, transformers, huggingface-hub, etc. We would like to trim the dependency tree a little bit at the beginning until we get a minimum working instance of vLLM. Such, this project involves the Debian packaging work for vLLM and its dependencies that are missing from Debian, as well as fixing issues (if there is any) in existing packages to make vLLM work.

## Personal Information
- Name: Kohei Sendai 
- Email: kouhei.sendai@gmail.com
- GitHub: [k1000dai](https://github.com/k1000dai)
- University: The University of Tokyo
- Major: Engineering, system innovation
- Year: 3rd grade undergraduate student
- Country: Japan

## Background
 I am a third grade undergraduate student of the University of Tokyo. Now, I am studing computer science at KTH in Sweden as an exchange student. I have used Debian family and am familiar with Linux OS. Also, I worked for AI Startup Company for two years and I do a part-time job at a Matsuo Laboratory which did research related to AI, so I have a good experience with Python, especially PyTorch and another AI-related library. 

## Why I am interested in this project
I used linux OS and apt package manager for a long time and I am interested in the Debian packaging system. Also, I have experience with PyTorch and another AI-related library, so I am interested in the LLM inference libraries. I think this project is a good opportunity to learn more about the Debian packaging system and AI-related library which sometimes have complex dependencies.
Also, through application tasks, I feel that debian community is really welcoming and I want to contribute to the community.
Additionally, I think local LLM inference is really important because local llm becomes more and more sophisticated nowadays. I hope I can contribute to the project and can create a good package for the local LLM inference.

## Application Tasks
### 1. Setup development environment
At first, I try to set up the development environment. To understand the Debian packaging system and to more familiar with the Debian system, I try to build by sbuild.

I used the following command to build the package.
1. Create a docker container or use aws instance
```bash
docker run --shm-size 2g -it --platform linux/amd64 --privileged --name=debian-amd64 debian:sid
```

I use the docker container for the testing. Also I use aws instance for the testing with amd64 architecture.

2. Install the necessary package
```bash
apt update
apt install sbuild debootstrap schroot build-essential
```
3. Create a chroot environment
```bash
sudo sbuild-createchroot --arch=amd64 sid /srv/chroot/sid-amd64 http://deb.debian.org/debian
```
4. get the source code

I use the hugingface-hub package which is one of the dependencies of the vLLM package. I get the source code from the github repository(https://salsa.debian.org/deeplearning-team/huggingface_hub)

Also for the autopkgtest, I create the following file.
 ~/.config/sbuild/config.pl

and add the following content.
```
$run_autopkgtest = 1;
```

5. Build the package

Then, I build the package with the following command.

```bash
cd huggingface_hub
mk-build-deps -i -r
sbuild -d sid -c sid-amd64-sbuild 
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
https://salsa.debian.org/k1000dai/huggingface_hub).

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

Also I add some autopkgtest for the testing.

## Project Details
### Implementation Plan

~ 5/8 : Analyze the dependency tree of the vllm package.(I already did in the next section.) Setup the development environment and add every dependency which already included in the Debian repository. Start with the architecture-independent package and non cuda variant. I will get the feedback from the mentor and continue to the next step.

5/8 ~ 6/1 : Starts with some of the packages which are not included in the Debian repository. I will start with the architecture-independent package and non cuda variant. I already create the package for huggingface-hub so I think this is a good start and don't take much time. By this time, I will finish the package for tokenizers and research more about transformers package. I will get the feedback from the mentor and continue to the next step.

6/1 ~ 6/30 : Create the package for the transformers package. After I could complete the transformers package, I think I could get some knowledge about the Debian packaging system with cuda variant. Then I will finish rest of the package for the vLLM package. I will get the feedback from the mentor and continue to the next step.

6/30 ~ 7/18 : I will try to build the vLLM package with cuda variant. Now, I already checked that vllm works on CUDA 12.4 so I will try to build the package with CUDA 12.4 with specific GPU(Maybe T4?). I will get the feedback from the mentor and continue to the next step.

7/18 : Mid-term evaluation

7/19 ~ 7/30 : Try to build the vLLM package with multiplatform. I will try to build the package with different CUDA version and different architecture, including multi-GPU environment. I will get the feedback from the mentor and continue to the next step.

8/1 ~ 8/25 : I understand project always have some issues and need more time than expected. So, I will try to finish the project as soon as possible and try to fix the issue. I will finish the project and get the feedback from the mentor.

If I have some more time, I will continue SGLang packaging.

8/25 - 9/1 : submit final work product.  

### Analyze the dependency tree of the vllm package
Here is the dependency tree of the vllm package.
```bash
vllm==0.8.1
├── cachetools [required: Any, installed: 5.5.2]
├── psutil [required: Any, installed: 7.0.0]
├── sentencepiece [required: Any, installed: 0.2.0]
├── numpy [required: <2.0.0, installed: 1.26.4]
├── requests [required: >=2.26.0, installed: 2.32.3]
│   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   ├── idna [required: >=2.5,<4, installed: 3.10]
│   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
├── tqdm [required: Any, installed: 4.67.1]
├── blake3 [required: Any, installed: 1.0.4]
├── py-cpuinfo [required: Any, installed: 9.0.0]
├── transformers [required: >=4.48.2, installed: 4.50.0]
│   ├── filelock [required: Any, installed: 3.18.0]
│   ├── huggingface-hub [required: >=0.26.0,<1.0, installed: 0.29.3]
│   │   ├── filelock [required: Any, installed: 3.18.0]
│   │   ├── fsspec [required: >=2023.5.0, installed: 2025.3.0]
│   │   ├── packaging [required: >=20.9, installed: 24.2]
│   │   ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   │   ├── requests [required: Any, installed: 2.32.3]
│   │   │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   │   ├── tqdm [required: >=4.42.1, installed: 4.67.1]
│   │   └── typing_extensions [required: >=3.7.4.3, installed: 4.12.2]
│   ├── numpy [required: >=1.17, installed: 1.26.4]
│   ├── packaging [required: >=20.0, installed: 24.2]
│   ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   ├── regex [required: !=2019.12.17, installed: 2024.11.6]
│   ├── requests [required: Any, installed: 2.32.3]
│   │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   ├── tokenizers [required: >=0.21,<0.22, installed: 0.21.1]
│   │   └── huggingface-hub [required: >=0.16.4,<1.0, installed: 0.29.3]
│   │       ├── filelock [required: Any, installed: 3.18.0]
│   │       ├── fsspec [required: >=2023.5.0, installed: 2025.3.0]
│   │       ├── packaging [required: >=20.9, installed: 24.2]
│   │       ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   │       ├── requests [required: Any, installed: 2.32.3]
│   │       │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │       │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │       │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │       │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   │       ├── tqdm [required: >=4.42.1, installed: 4.67.1]
│   │       └── typing_extensions [required: >=3.7.4.3, installed: 4.12.2]
│   ├── safetensors [required: >=0.4.3, installed: 0.5.3]
│   └── tqdm [required: >=4.27, installed: 4.67.1]
├── tokenizers [required: >=0.19.1, installed: 0.21.1]
│   └── huggingface-hub [required: >=0.16.4,<1.0, installed: 0.29.3]
│       ├── filelock [required: Any, installed: 3.18.0]
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
├── protobuf [required: Any, installed: 6.30.1]
├── fastapi [required: >=0.115.0, installed: 0.115.11]
│   ├── starlette [required: >=0.40.0,<0.47.0, installed: 0.46.1]
│   │   └── anyio [required: >=3.6.2,<5, installed: 4.9.0]
│   │       ├── idna [required: >=2.8, installed: 3.10]
│   │       ├── sniffio [required: >=1.1, installed: 1.3.1]
│   │       └── typing_extensions [required: >=4.5, installed: 4.12.2]
│   ├── pydantic [required: >=1.7.4,<3.0.0,!=2.1.0,!=2.0.1,!=2.0.0,!=1.8.1,!=1.8, installed: 2.10.6]
│   │   ├── annotated-types [required: >=0.6.0, installed: 0.7.0]
│   │   ├── pydantic_core [required: ==2.27.2, installed: 2.27.2]
│   │   │   └── typing_extensions [required: >=4.6.0,!=4.7.0, installed: 4.12.2]
│   │   └── typing_extensions [required: >=4.12.2, installed: 4.12.2]
│   └── typing_extensions [required: >=4.8.0, installed: 4.12.2]
├── aiohttp [required: Any, installed: 3.11.14]
│   ├── aiohappyeyeballs [required: >=2.3.0, installed: 2.6.1]
│   ├── aiosignal [required: >=1.1.2, installed: 1.3.2]
│   │   └── frozenlist [required: >=1.1.0, installed: 1.5.0]
│   ├── attrs [required: >=17.3.0, installed: 25.3.0]
│   ├── frozenlist [required: >=1.1.1, installed: 1.5.0]
│   ├── multidict [required: >=4.5,<7.0, installed: 6.2.0]
│   ├── propcache [required: >=0.2.0, installed: 0.3.0]
│   └── yarl [required: >=1.17.0,<2.0, installed: 1.18.3]
│       ├── idna [required: >=2.0, installed: 3.10]
│       ├── multidict [required: >=4.0, installed: 6.2.0]
│       └── propcache [required: >=0.2.0, installed: 0.3.0]
├── openai [required: >=1.52.0, installed: 1.68.2]
│   ├── anyio [required: >=3.5.0,<5, installed: 4.9.0]
│   │   ├── idna [required: >=2.8, installed: 3.10]
│   │   ├── sniffio [required: >=1.1, installed: 1.3.1]
│   │   └── typing_extensions [required: >=4.5, installed: 4.12.2]
│   ├── distro [required: >=1.7.0,<2, installed: 1.9.0]
│   ├── httpx [required: >=0.23.0,<1, installed: 0.28.1]
│   │   ├── anyio [required: Any, installed: 4.9.0]
│   │   │   ├── idna [required: >=2.8, installed: 3.10]
│   │   │   ├── sniffio [required: >=1.1, installed: 1.3.1]
│   │   │   └── typing_extensions [required: >=4.5, installed: 4.12.2]
│   │   ├── certifi [required: Any, installed: 2025.1.31]
│   │   ├── httpcore [required: ==1.*, installed: 1.0.7]
│   │   │   ├── certifi [required: Any, installed: 2025.1.31]
│   │   │   └── h11 [required: >=0.13,<0.15, installed: 0.14.0]
│   │   └── idna [required: Any, installed: 3.10]
│   ├── jiter [required: >=0.4.0,<1, installed: 0.9.0]
│   ├── pydantic [required: >=1.9.0,<3, installed: 2.10.6]
│   │   ├── annotated-types [required: >=0.6.0, installed: 0.7.0]
│   │   ├── pydantic_core [required: ==2.27.2, installed: 2.27.2]
│   │   │   └── typing_extensions [required: >=4.6.0,!=4.7.0, installed: 4.12.2]
│   │   └── typing_extensions [required: >=4.12.2, installed: 4.12.2]
│   ├── sniffio [required: Any, installed: 1.3.1]
│   ├── tqdm [required: >4, installed: 4.67.1]
│   └── typing_extensions [required: >=4.11,<5, installed: 4.12.2]
├── pydantic [required: >=2.9, installed: 2.10.6]
│   ├── annotated-types [required: >=0.6.0, installed: 0.7.0]
│   ├── pydantic_core [required: ==2.27.2, installed: 2.27.2]
│   │   └── typing_extensions [required: >=4.6.0,!=4.7.0, installed: 4.12.2]
│   └── typing_extensions [required: >=4.12.2, installed: 4.12.2]
├── prometheus_client [required: >=0.18.0, installed: 0.21.1]
├── pillow [required: Any, installed: 11.1.0]
├── prometheus-fastapi-instrumentator [required: >=7.0.0, installed: 7.1.0]
│   ├── prometheus_client [required: >=0.8.0,<1.0.0, installed: 0.21.1]
│   └── starlette [required: >=0.30.0,<1.0.0, installed: 0.46.1]
│       └── anyio [required: >=3.6.2,<5, installed: 4.9.0]
│           ├── idna [required: >=2.8, installed: 3.10]
│           ├── sniffio [required: >=1.1, installed: 1.3.1]
│           └── typing_extensions [required: >=4.5, installed: 4.12.2]
├── tiktoken [required: >=0.6.0, installed: 0.9.0]
│   ├── regex [required: >=2022.1.18, installed: 2024.11.6]
│   └── requests [required: >=2.26.0, installed: 2.32.3]
│       ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│       ├── idna [required: >=2.5,<4, installed: 3.10]
│       ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│       └── certifi [required: >=2017.4.17, installed: 2025.1.31]
├── lm-format-enforcer [required: >=0.10.11,<0.11, installed: 0.10.11]
│   ├── interegular [required: >=0.3.2, installed: 0.3.3]
│   ├── packaging [required: Any, installed: 24.2]
│   ├── pydantic [required: >=1.10.8, installed: 2.10.6]
│   │   ├── annotated-types [required: >=0.6.0, installed: 0.7.0]
│   │   ├── pydantic_core [required: ==2.27.2, installed: 2.27.2]
│   │   │   └── typing_extensions [required: >=4.6.0,!=4.7.0, installed: 4.12.2]
│   │   └── typing_extensions [required: >=4.12.2, installed: 4.12.2]
│   └── PyYAML [required: Any, installed: 6.0.2]
├── outlines [required: ==0.1.11, installed: 0.1.11]
│   ├── interegular [required: Any, installed: 0.3.3]
│   ├── Jinja2 [required: Any, installed: 3.1.6]
│   │   └── MarkupSafe [required: >=2.0, installed: 3.0.2]
│   ├── lark [required: Any, installed: 1.2.2]
│   ├── nest-asyncio [required: Any, installed: 1.6.0]
│   ├── numpy [required: Any, installed: 1.26.4]
│   ├── cloudpickle [required: Any, installed: 3.1.1]
│   ├── diskcache [required: Any, installed: 5.6.3]
│   ├── pydantic [required: >=2.0, installed: 2.10.6]
│   │   ├── annotated-types [required: >=0.6.0, installed: 0.7.0]
│   │   ├── pydantic_core [required: ==2.27.2, installed: 2.27.2]
│   │   │   └── typing_extensions [required: >=4.6.0,!=4.7.0, installed: 4.12.2]
│   │   └── typing_extensions [required: >=4.12.2, installed: 4.12.2]
│   ├── referencing [required: Any, installed: 0.36.2]
│   │   ├── attrs [required: >=22.2.0, installed: 25.3.0]
│   │   ├── rpds-py [required: >=0.7.0, installed: 0.23.1]
│   │   └── typing_extensions [required: >=4.4.0, installed: 4.12.2]
│   ├── jsonschema [required: Any, installed: 4.23.0]
│   │   ├── attrs [required: >=22.2.0, installed: 25.3.0]
│   │   ├── jsonschema-specifications [required: >=2023.03.6, installed: 2024.10.1]
│   │   │   └── referencing [required: >=0.31.0, installed: 0.36.2]
│   │   │       ├── attrs [required: >=22.2.0, installed: 25.3.0]
│   │   │       ├── rpds-py [required: >=0.7.0, installed: 0.23.1]
│   │   │       └── typing_extensions [required: >=4.4.0, installed: 4.12.2]
│   │   ├── referencing [required: >=0.28.4, installed: 0.36.2]
│   │   │   ├── attrs [required: >=22.2.0, installed: 25.3.0]
│   │   │   ├── rpds-py [required: >=0.7.0, installed: 0.23.1]
│   │   │   └── typing_extensions [required: >=4.4.0, installed: 4.12.2]
│   │   └── rpds-py [required: >=0.7.1, installed: 0.23.1]
│   ├── requests [required: Any, installed: 2.32.3]
│   │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   ├── tqdm [required: Any, installed: 4.67.1]
│   ├── typing_extensions [required: Any, installed: 4.12.2]
│   ├── pycountry [required: Any, installed: 24.6.1]
│   ├── airportsdata [required: Any, installed: 20250224]
│   ├── torch [required: Any, installed: 2.6.0]
│   │   ├── filelock [required: Any, installed: 3.18.0]
│   │   ├── typing_extensions [required: >=4.10.0, installed: 4.12.2]
│   │   ├── networkx [required: Any, installed: 3.4.2]
│   │   ├── Jinja2 [required: Any, installed: 3.1.6]
│   │   │   └── MarkupSafe [required: >=2.0, installed: 3.0.2]
│   │   ├── fsspec [required: Any, installed: 2025.3.0]
│   │   ├── nvidia-cuda-nvrtc-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cuda-runtime-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cuda-cupti-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cudnn-cu12 [required: ==9.1.0.70, installed: 9.1.0.70]
│   │   │   └── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│   │   ├── nvidia-cublas-cu12 [required: ==12.4.5.8, installed: 12.4.5.8]
│   │   ├── nvidia-cufft-cu12 [required: ==11.2.1.3, installed: 11.2.1.3]
│   │   ├── nvidia-curand-cu12 [required: ==10.3.5.147, installed: 10.3.5.147]
│   │   ├── nvidia-cusolver-cu12 [required: ==11.6.1.9, installed: 11.6.1.9]
│   │   │   ├── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│   │   │   ├── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   │   └── nvidia-cusparse-cu12 [required: Any, installed: 12.3.1.170]
│   │   │       └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   ├── nvidia-cusparse-cu12 [required: ==12.3.1.170, installed: 12.3.1.170]
│   │   │   └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   ├── nvidia-cusparselt-cu12 [required: ==0.6.2, installed: 0.6.2]
│   │   ├── nvidia-nccl-cu12 [required: ==2.21.5, installed: 2.21.5]
│   │   ├── nvidia-nvtx-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-nvjitlink-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── triton [required: ==3.2.0, installed: 3.2.0]
│   │   ├── setuptools [required: Any, installed: 77.0.3]
│   │   └── sympy [required: ==1.13.1, installed: 1.13.1]
│   │       └── mpmath [required: >=1.1.0,<1.4, installed: 1.3.0]
│   └── outlines_core [required: ==0.1.26, installed: 0.1.26]
│       ├── interegular [required: Any, installed: 0.3.3]
│       └── jsonschema [required: Any, installed: 4.23.0]
│           ├── attrs [required: >=22.2.0, installed: 25.3.0]
│           ├── jsonschema-specifications [required: >=2023.03.6, installed: 2024.10.1]
│           │   └── referencing [required: >=0.31.0, installed: 0.36.2]
│           │       ├── attrs [required: >=22.2.0, installed: 25.3.0]
│           │       ├── rpds-py [required: >=0.7.0, installed: 0.23.1]
│           │       └── typing_extensions [required: >=4.4.0, installed: 4.12.2]
│           ├── referencing [required: >=0.28.4, installed: 0.36.2]
│           │   ├── attrs [required: >=22.2.0, installed: 25.3.0]
│           │   ├── rpds-py [required: >=0.7.0, installed: 0.23.1]
│           │   └── typing_extensions [required: >=4.4.0, installed: 4.12.2]
│           └── rpds-py [required: >=0.7.1, installed: 0.23.1]
├── lark [required: ==1.2.2, installed: 1.2.2]
├── xgrammar [required: ==0.1.16, installed: 0.1.16]
│   ├── pydantic [required: Any, installed: 2.10.6]
│   │   ├── annotated-types [required: >=0.6.0, installed: 0.7.0]
│   │   ├── pydantic_core [required: ==2.27.2, installed: 2.27.2]
│   │   │   └── typing_extensions [required: >=4.6.0,!=4.7.0, installed: 4.12.2]
│   │   └── typing_extensions [required: >=4.12.2, installed: 4.12.2]
│   ├── sentencepiece [required: Any, installed: 0.2.0]
│   ├── tiktoken [required: Any, installed: 0.9.0]
│   │   ├── regex [required: >=2022.1.18, installed: 2024.11.6]
│   │   └── requests [required: >=2.26.0, installed: 2.32.3]
│   │       ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │       ├── idna [required: >=2.5,<4, installed: 3.10]
│   │       ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │       └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   ├── torch [required: >=1.10.0, installed: 2.6.0]
│   │   ├── filelock [required: Any, installed: 3.18.0]
│   │   ├── typing_extensions [required: >=4.10.0, installed: 4.12.2]
│   │   ├── networkx [required: Any, installed: 3.4.2]
│   │   ├── Jinja2 [required: Any, installed: 3.1.6]
│   │   │   └── MarkupSafe [required: >=2.0, installed: 3.0.2]
│   │   ├── fsspec [required: Any, installed: 2025.3.0]
│   │   ├── nvidia-cuda-nvrtc-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cuda-runtime-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cuda-cupti-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cudnn-cu12 [required: ==9.1.0.70, installed: 9.1.0.70]
│   │   │   └── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│   │   ├── nvidia-cublas-cu12 [required: ==12.4.5.8, installed: 12.4.5.8]
│   │   ├── nvidia-cufft-cu12 [required: ==11.2.1.3, installed: 11.2.1.3]
│   │   ├── nvidia-curand-cu12 [required: ==10.3.5.147, installed: 10.3.5.147]
│   │   ├── nvidia-cusolver-cu12 [required: ==11.6.1.9, installed: 11.6.1.9]
│   │   │   ├── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│   │   │   ├── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   │   └── nvidia-cusparse-cu12 [required: Any, installed: 12.3.1.170]
│   │   │       └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   ├── nvidia-cusparse-cu12 [required: ==12.3.1.170, installed: 12.3.1.170]
│   │   │   └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   ├── nvidia-cusparselt-cu12 [required: ==0.6.2, installed: 0.6.2]
│   │   ├── nvidia-nccl-cu12 [required: ==2.21.5, installed: 2.21.5]
│   │   ├── nvidia-nvtx-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-nvjitlink-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── triton [required: ==3.2.0, installed: 3.2.0]
│   │   ├── setuptools [required: Any, installed: 77.0.3]
│   │   └── sympy [required: ==1.13.1, installed: 1.13.1]
│   │       └── mpmath [required: >=1.1.0,<1.4, installed: 1.3.0]
│   ├── transformers [required: >=4.38.0, installed: 4.50.0]
│   │   ├── filelock [required: Any, installed: 3.18.0]
│   │   ├── huggingface-hub [required: >=0.26.0,<1.0, installed: 0.29.3]
│   │   │   ├── filelock [required: Any, installed: 3.18.0]
│   │   │   ├── fsspec [required: >=2023.5.0, installed: 2025.3.0]
│   │   │   ├── packaging [required: >=20.9, installed: 24.2]
│   │   │   ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   │   │   ├── requests [required: Any, installed: 2.32.3]
│   │   │   │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   │   │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   │   │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   │   │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   │   │   ├── tqdm [required: >=4.42.1, installed: 4.67.1]
│   │   │   └── typing_extensions [required: >=3.7.4.3, installed: 4.12.2]
│   │   ├── numpy [required: >=1.17, installed: 1.26.4]
│   │   ├── packaging [required: >=20.0, installed: 24.2]
│   │   ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   │   ├── regex [required: !=2019.12.17, installed: 2024.11.6]
│   │   ├── requests [required: Any, installed: 2.32.3]
│   │   │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   │   ├── tokenizers [required: >=0.21,<0.22, installed: 0.21.1]
│   │   │   └── huggingface-hub [required: >=0.16.4,<1.0, installed: 0.29.3]
│   │   │       ├── filelock [required: Any, installed: 3.18.0]
│   │   │       ├── fsspec [required: >=2023.5.0, installed: 2025.3.0]
│   │   │       ├── packaging [required: >=20.9, installed: 24.2]
│   │   │       ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   │   │       ├── requests [required: Any, installed: 2.32.3]
│   │   │       │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   │       │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   │       │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   │       │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   │   │       ├── tqdm [required: >=4.42.1, installed: 4.67.1]
│   │   │       └── typing_extensions [required: >=3.7.4.3, installed: 4.12.2]
│   │   ├── safetensors [required: >=0.4.3, installed: 0.5.3]
│   │   └── tqdm [required: >=4.27, installed: 4.67.1]
│   └── ninja [required: Any, installed: 1.11.1.4]
├── typing_extensions [required: >=4.10, installed: 4.12.2]
├── filelock [required: >=3.16.1, installed: 3.18.0]
├── partial-json-parser [required: Any, installed: 0.2.1.1.post5]
├── pyzmq [required: Any, installed: 26.3.0]
├── msgspec [required: Any, installed: 0.19.0]
├── gguf [required: ==0.10.0, installed: 0.10.0]
│   ├── numpy [required: >=1.17, installed: 1.26.4]
│   ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   └── tqdm [required: >=4.27, installed: 4.67.1]
├── importlib_metadata [required: Any, installed: 8.6.1]
│   └── zipp [required: >=3.20, installed: 3.21.0]
├── mistral_common [required: >=1.5.4, installed: 1.5.4]
│   ├── jsonschema [required: >=4.21.1, installed: 4.23.0]
│   │   ├── attrs [required: >=22.2.0, installed: 25.3.0]
│   │   ├── jsonschema-specifications [required: >=2023.03.6, installed: 2024.10.1]
│   │   │   └── referencing [required: >=0.31.0, installed: 0.36.2]
│   │   │       ├── attrs [required: >=22.2.0, installed: 25.3.0]
│   │   │       ├── rpds-py [required: >=0.7.0, installed: 0.23.1]
│   │   │       └── typing_extensions [required: >=4.4.0, installed: 4.12.2]
│   │   ├── referencing [required: >=0.28.4, installed: 0.36.2]
│   │   │   ├── attrs [required: >=22.2.0, installed: 25.3.0]
│   │   │   ├── rpds-py [required: >=0.7.0, installed: 0.23.1]
│   │   │   └── typing_extensions [required: >=4.4.0, installed: 4.12.2]
│   │   └── rpds-py [required: >=0.7.1, installed: 0.23.1]
│   ├── numpy [required: >=1.25, installed: 1.26.4]
│   ├── pillow [required: >=10.3.0, installed: 11.1.0]
│   ├── pydantic [required: >=2.7,<3.0, installed: 2.10.6]
│   │   ├── annotated-types [required: >=0.6.0, installed: 0.7.0]
│   │   ├── pydantic_core [required: ==2.27.2, installed: 2.27.2]
│   │   │   └── typing_extensions [required: >=4.6.0,!=4.7.0, installed: 4.12.2]
│   │   └── typing_extensions [required: >=4.12.2, installed: 4.12.2]
│   ├── requests [required: >=2.0.0, installed: 2.32.3]
│   │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   ├── sentencepiece [required: >=0.2.0, installed: 0.2.0]
│   ├── tiktoken [required: >=0.7.0, installed: 0.9.0]
│   │   ├── regex [required: >=2022.1.18, installed: 2024.11.6]
│   │   └── requests [required: >=2.26.0, installed: 2.32.3]
│   │       ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │       ├── idna [required: >=2.5,<4, installed: 3.10]
│   │       ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │       └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   └── typing_extensions [required: >=4.11.0, installed: 4.12.2]
├── PyYAML [required: Any, installed: 6.0.2]
├── six [required: >=1.16.0, installed: 1.17.0]
├── setuptools [required: >=74.1.1, installed: 77.0.3]
├── einops [required: Any, installed: 0.8.1]
├── compressed-tensors [required: ==0.9.2, installed: 0.9.2]
│   ├── torch [required: >=1.7.0, installed: 2.6.0]
│   │   ├── filelock [required: Any, installed: 3.18.0]
│   │   ├── typing_extensions [required: >=4.10.0, installed: 4.12.2]
│   │   ├── networkx [required: Any, installed: 3.4.2]
│   │   ├── Jinja2 [required: Any, installed: 3.1.6]
│   │   │   └── MarkupSafe [required: >=2.0, installed: 3.0.2]
│   │   ├── fsspec [required: Any, installed: 2025.3.0]
│   │   ├── nvidia-cuda-nvrtc-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cuda-runtime-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cuda-cupti-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cudnn-cu12 [required: ==9.1.0.70, installed: 9.1.0.70]
│   │   │   └── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│   │   ├── nvidia-cublas-cu12 [required: ==12.4.5.8, installed: 12.4.5.8]
│   │   ├── nvidia-cufft-cu12 [required: ==11.2.1.3, installed: 11.2.1.3]
│   │   ├── nvidia-curand-cu12 [required: ==10.3.5.147, installed: 10.3.5.147]
│   │   ├── nvidia-cusolver-cu12 [required: ==11.6.1.9, installed: 11.6.1.9]
│   │   │   ├── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│   │   │   ├── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   │   └── nvidia-cusparse-cu12 [required: Any, installed: 12.3.1.170]
│   │   │       └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   ├── nvidia-cusparse-cu12 [required: ==12.3.1.170, installed: 12.3.1.170]
│   │   │   └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   ├── nvidia-cusparselt-cu12 [required: ==0.6.2, installed: 0.6.2]
│   │   ├── nvidia-nccl-cu12 [required: ==2.21.5, installed: 2.21.5]
│   │   ├── nvidia-nvtx-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-nvjitlink-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── triton [required: ==3.2.0, installed: 3.2.0]
│   │   ├── setuptools [required: Any, installed: 77.0.3]
│   │   └── sympy [required: ==1.13.1, installed: 1.13.1]
│   │       └── mpmath [required: >=1.1.0,<1.4, installed: 1.3.0]
│   ├── transformers [required: Any, installed: 4.50.0]
│   │   ├── filelock [required: Any, installed: 3.18.0]
│   │   ├── huggingface-hub [required: >=0.26.0,<1.0, installed: 0.29.3]
│   │   │   ├── filelock [required: Any, installed: 3.18.0]
│   │   │   ├── fsspec [required: >=2023.5.0, installed: 2025.3.0]
│   │   │   ├── packaging [required: >=20.9, installed: 24.2]
│   │   │   ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   │   │   ├── requests [required: Any, installed: 2.32.3]
│   │   │   │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   │   │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   │   │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   │   │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   │   │   ├── tqdm [required: >=4.42.1, installed: 4.67.1]
│   │   │   └── typing_extensions [required: >=3.7.4.3, installed: 4.12.2]
│   │   ├── numpy [required: >=1.17, installed: 1.26.4]
│   │   ├── packaging [required: >=20.0, installed: 24.2]
│   │   ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   │   ├── regex [required: !=2019.12.17, installed: 2024.11.6]
│   │   ├── requests [required: Any, installed: 2.32.3]
│   │   │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   │   ├── tokenizers [required: >=0.21,<0.22, installed: 0.21.1]
│   │   │   └── huggingface-hub [required: >=0.16.4,<1.0, installed: 0.29.3]
│   │   │       ├── filelock [required: Any, installed: 3.18.0]
│   │   │       ├── fsspec [required: >=2023.5.0, installed: 2025.3.0]
│   │   │       ├── packaging [required: >=20.9, installed: 24.2]
│   │   │       ├── PyYAML [required: >=5.1, installed: 6.0.2]
│   │   │       ├── requests [required: Any, installed: 2.32.3]
│   │   │       │   ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│   │   │       │   ├── idna [required: >=2.5,<4, installed: 3.10]
│   │   │       │   ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│   │   │       │   └── certifi [required: >=2017.4.17, installed: 2025.1.31]
│   │   │       ├── tqdm [required: >=4.42.1, installed: 4.67.1]
│   │   │       └── typing_extensions [required: >=3.7.4.3, installed: 4.12.2]
│   │   ├── safetensors [required: >=0.4.3, installed: 0.5.3]
│   │   └── tqdm [required: >=4.27, installed: 4.67.1]
│   └── pydantic [required: >=2.0, installed: 2.10.6]
│       ├── annotated-types [required: >=0.6.0, installed: 0.7.0]
│       ├── pydantic_core [required: ==2.27.2, installed: 2.27.2]
│       │   └── typing_extensions [required: >=4.6.0,!=4.7.0, installed: 4.12.2]
│       └── typing_extensions [required: >=4.12.2, installed: 4.12.2]
├── depyf [required: ==0.18.0, installed: 0.18.0]
│   ├── astor [required: Any, installed: 0.8.1]
│   └── dill [required: Any, installed: 0.3.9]
├── cloudpickle [required: Any, installed: 3.1.1]
├── watchfiles [required: Any, installed: 1.0.4]
│   └── anyio [required: >=3.0.0, installed: 4.9.0]
│       ├── idna [required: >=2.8, installed: 3.10]
│       ├── sniffio [required: >=1.1, installed: 1.3.1]
│       └── typing_extensions [required: >=4.5, installed: 4.12.2]
├── python-json-logger [required: Any, installed: 3.3.0]
├── scipy [required: Any, installed: 1.15.2]
│   └── numpy [required: >=1.23.5,<2.5, installed: 1.26.4]
├── ninja [required: Any, installed: 1.11.1.4]
├── numba [required: ==0.60.0, installed: 0.60.0]
│   ├── llvmlite [required: >=0.43.0dev0,<0.44, installed: 0.43.0]
│   └── numpy [required: >=1.22,<2.1, installed: 1.26.4]
├── ray [required: >=2.43.0, installed: 2.44.0]
│   ├── click [required: >=7.0, installed: 8.1.8]
│   ├── filelock [required: Any, installed: 3.18.0]
│   ├── jsonschema [required: Any, installed: 4.23.0]
│   │   ├── attrs [required: >=22.2.0, installed: 25.3.0]
│   │   ├── jsonschema-specifications [required: >=2023.03.6, installed: 2024.10.1]
│   │   │   └── referencing [required: >=0.31.0, installed: 0.36.2]
│   │   │       ├── attrs [required: >=22.2.0, installed: 25.3.0]
│   │   │       ├── rpds-py [required: >=0.7.0, installed: 0.23.1]
│   │   │       └── typing_extensions [required: >=4.4.0, installed: 4.12.2]
│   │   ├── referencing [required: >=0.28.4, installed: 0.36.2]
│   │   │   ├── attrs [required: >=22.2.0, installed: 25.3.0]
│   │   │   ├── rpds-py [required: >=0.7.0, installed: 0.23.1]
│   │   │   └── typing_extensions [required: >=4.4.0, installed: 4.12.2]
│   │   └── rpds-py [required: >=0.7.1, installed: 0.23.1]
│   ├── msgpack [required: >=1.0.0,<2.0.0, installed: 1.1.0]
│   ├── packaging [required: Any, installed: 24.2]
│   ├── protobuf [required: >=3.15.3,!=3.19.5, installed: 6.30.1]
│   ├── PyYAML [required: Any, installed: 6.0.2]
│   ├── aiosignal [required: Any, installed: 1.3.2]
│   │   └── frozenlist [required: >=1.1.0, installed: 1.5.0]
│   ├── frozenlist [required: Any, installed: 1.5.0]
│   └── requests [required: Any, installed: 2.32.3]
│       ├── charset-normalizer [required: >=2,<4, installed: 3.4.1]
│       ├── idna [required: >=2.5,<4, installed: 3.10]
│       ├── urllib3 [required: >=1.21.1,<3, installed: 2.3.0]
│       └── certifi [required: >=2017.4.17, installed: 2025.1.31]
├── torch [required: ==2.6.0, installed: 2.6.0]
│   ├── filelock [required: Any, installed: 3.18.0]
│   ├── typing_extensions [required: >=4.10.0, installed: 4.12.2]
│   ├── networkx [required: Any, installed: 3.4.2]
│   ├── Jinja2 [required: Any, installed: 3.1.6]
│   │   └── MarkupSafe [required: >=2.0, installed: 3.0.2]
│   ├── fsspec [required: Any, installed: 2025.3.0]
│   ├── nvidia-cuda-nvrtc-cu12 [required: ==12.4.127, installed: 12.4.127]
│   ├── nvidia-cuda-runtime-cu12 [required: ==12.4.127, installed: 12.4.127]
│   ├── nvidia-cuda-cupti-cu12 [required: ==12.4.127, installed: 12.4.127]
│   ├── nvidia-cudnn-cu12 [required: ==9.1.0.70, installed: 9.1.0.70]
│   │   └── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│   ├── nvidia-cublas-cu12 [required: ==12.4.5.8, installed: 12.4.5.8]
│   ├── nvidia-cufft-cu12 [required: ==11.2.1.3, installed: 11.2.1.3]
│   ├── nvidia-curand-cu12 [required: ==10.3.5.147, installed: 10.3.5.147]
│   ├── nvidia-cusolver-cu12 [required: ==11.6.1.9, installed: 11.6.1.9]
│   │   ├── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│   │   ├── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   └── nvidia-cusparse-cu12 [required: Any, installed: 12.3.1.170]
│   │       └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   ├── nvidia-cusparse-cu12 [required: ==12.3.1.170, installed: 12.3.1.170]
│   │   └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   ├── nvidia-cusparselt-cu12 [required: ==0.6.2, installed: 0.6.2]
│   ├── nvidia-nccl-cu12 [required: ==2.21.5, installed: 2.21.5]
│   ├── nvidia-nvtx-cu12 [required: ==12.4.127, installed: 12.4.127]
│   ├── nvidia-nvjitlink-cu12 [required: ==12.4.127, installed: 12.4.127]
│   ├── triton [required: ==3.2.0, installed: 3.2.0]
│   ├── setuptools [required: Any, installed: 77.0.3]
│   └── sympy [required: ==1.13.1, installed: 1.13.1]
│       └── mpmath [required: >=1.1.0,<1.4, installed: 1.3.0]
├── torchaudio [required: ==2.6.0, installed: 2.6.0]
│   └── torch [required: ==2.6.0, installed: 2.6.0]
│       ├── filelock [required: Any, installed: 3.18.0]
│       ├── typing_extensions [required: >=4.10.0, installed: 4.12.2]
│       ├── networkx [required: Any, installed: 3.4.2]
│       ├── Jinja2 [required: Any, installed: 3.1.6]
│       │   └── MarkupSafe [required: >=2.0, installed: 3.0.2]
│       ├── fsspec [required: Any, installed: 2025.3.0]
│       ├── nvidia-cuda-nvrtc-cu12 [required: ==12.4.127, installed: 12.4.127]
│       ├── nvidia-cuda-runtime-cu12 [required: ==12.4.127, installed: 12.4.127]
│       ├── nvidia-cuda-cupti-cu12 [required: ==12.4.127, installed: 12.4.127]
│       ├── nvidia-cudnn-cu12 [required: ==9.1.0.70, installed: 9.1.0.70]
│       │   └── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│       ├── nvidia-cublas-cu12 [required: ==12.4.5.8, installed: 12.4.5.8]
│       ├── nvidia-cufft-cu12 [required: ==11.2.1.3, installed: 11.2.1.3]
│       ├── nvidia-curand-cu12 [required: ==10.3.5.147, installed: 10.3.5.147]
│       ├── nvidia-cusolver-cu12 [required: ==11.6.1.9, installed: 11.6.1.9]
│       │   ├── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│       │   ├── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│       │   └── nvidia-cusparse-cu12 [required: Any, installed: 12.3.1.170]
│       │       └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│       ├── nvidia-cusparse-cu12 [required: ==12.3.1.170, installed: 12.3.1.170]
│       │   └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│       ├── nvidia-cusparselt-cu12 [required: ==0.6.2, installed: 0.6.2]
│       ├── nvidia-nccl-cu12 [required: ==2.21.5, installed: 2.21.5]
│       ├── nvidia-nvtx-cu12 [required: ==12.4.127, installed: 12.4.127]
│       ├── nvidia-nvjitlink-cu12 [required: ==12.4.127, installed: 12.4.127]
│       ├── triton [required: ==3.2.0, installed: 3.2.0]
│       ├── setuptools [required: Any, installed: 77.0.3]
│       └── sympy [required: ==1.13.1, installed: 1.13.1]
│           └── mpmath [required: >=1.1.0,<1.4, installed: 1.3.0]
├── torchvision [required: ==0.21.0, installed: 0.21.0]
│   ├── numpy [required: Any, installed: 1.26.4]
│   ├── torch [required: ==2.6.0, installed: 2.6.0]
│   │   ├── filelock [required: Any, installed: 3.18.0]
│   │   ├── typing_extensions [required: >=4.10.0, installed: 4.12.2]
│   │   ├── networkx [required: Any, installed: 3.4.2]
│   │   ├── Jinja2 [required: Any, installed: 3.1.6]
│   │   │   └── MarkupSafe [required: >=2.0, installed: 3.0.2]
│   │   ├── fsspec [required: Any, installed: 2025.3.0]
│   │   ├── nvidia-cuda-nvrtc-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cuda-runtime-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cuda-cupti-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-cudnn-cu12 [required: ==9.1.0.70, installed: 9.1.0.70]
│   │   │   └── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│   │   ├── nvidia-cublas-cu12 [required: ==12.4.5.8, installed: 12.4.5.8]
│   │   ├── nvidia-cufft-cu12 [required: ==11.2.1.3, installed: 11.2.1.3]
│   │   ├── nvidia-curand-cu12 [required: ==10.3.5.147, installed: 10.3.5.147]
│   │   ├── nvidia-cusolver-cu12 [required: ==11.6.1.9, installed: 11.6.1.9]
│   │   │   ├── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
│   │   │   ├── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   │   └── nvidia-cusparse-cu12 [required: Any, installed: 12.3.1.170]
│   │   │       └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   ├── nvidia-cusparse-cu12 [required: ==12.3.1.170, installed: 12.3.1.170]
│   │   │   └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
│   │   ├── nvidia-cusparselt-cu12 [required: ==0.6.2, installed: 0.6.2]
│   │   ├── nvidia-nccl-cu12 [required: ==2.21.5, installed: 2.21.5]
│   │   ├── nvidia-nvtx-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── nvidia-nvjitlink-cu12 [required: ==12.4.127, installed: 12.4.127]
│   │   ├── triton [required: ==3.2.0, installed: 3.2.0]
│   │   ├── setuptools [required: Any, installed: 77.0.3]
│   │   └── sympy [required: ==1.13.1, installed: 1.13.1]
│   │       └── mpmath [required: >=1.1.0,<1.4, installed: 1.3.0]
│   └── pillow [required: >=5.3.0,!=8.3.*, installed: 11.1.0]
└── xformers [required: ==0.0.29.post2, installed: 0.0.29.post2]
    ├── numpy [required: Any, installed: 1.26.4]
    └── torch [required: ==2.6.0, installed: 2.6.0]
        ├── filelock [required: Any, installed: 3.18.0]
        ├── typing_extensions [required: >=4.10.0, installed: 4.12.2]
        ├── networkx [required: Any, installed: 3.4.2]
        ├── Jinja2 [required: Any, installed: 3.1.6]
        │   └── MarkupSafe [required: >=2.0, installed: 3.0.2]
        ├── fsspec [required: Any, installed: 2025.3.0]
        ├── nvidia-cuda-nvrtc-cu12 [required: ==12.4.127, installed: 12.4.127]
        ├── nvidia-cuda-runtime-cu12 [required: ==12.4.127, installed: 12.4.127]
        ├── nvidia-cuda-cupti-cu12 [required: ==12.4.127, installed: 12.4.127]
        ├── nvidia-cudnn-cu12 [required: ==9.1.0.70, installed: 9.1.0.70]
        │   └── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
        ├── nvidia-cublas-cu12 [required: ==12.4.5.8, installed: 12.4.5.8]
        ├── nvidia-cufft-cu12 [required: ==11.2.1.3, installed: 11.2.1.3]
        ├── nvidia-curand-cu12 [required: ==10.3.5.147, installed: 10.3.5.147]
        ├── nvidia-cusolver-cu12 [required: ==11.6.1.9, installed: 11.6.1.9]
        │   ├── nvidia-cublas-cu12 [required: Any, installed: 12.4.5.8]
        │   ├── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
        │   └── nvidia-cusparse-cu12 [required: Any, installed: 12.3.1.170]
        │       └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
        ├── nvidia-cusparse-cu12 [required: ==12.3.1.170, installed: 12.3.1.170]
        │   └── nvidia-nvjitlink-cu12 [required: Any, installed: 12.4.127]
        ├── nvidia-cusparselt-cu12 [required: ==0.6.2, installed: 0.6.2]
        ├── nvidia-nccl-cu12 [required: ==2.21.5, installed: 2.21.5]
        ├── nvidia-nvtx-cu12 [required: ==12.4.127, installed: 12.4.127]
        ├── nvidia-nvjitlink-cu12 [required: ==12.4.127, installed: 12.4.127]
        ├── triton [required: ==3.2.0, installed: 3.2.0]
        ├── setuptools [required: Any, installed: 77.0.3]
        └── sympy [required: ==1.13.1, installed: 1.13.1]
            └── mpmath [required: >=1.1.0,<1.4, installed: 1.3.0]
```

It is a long list of dependencies, but thanks for the debian community, most of them are already packaged in the debian repository.

* python3-cachetools (https://packages.debian.org/sid/python3-cachetools)
* python3-aiohttp (https://packages.debian.org/sid/python3-aiohttp)
* python3-cloudpickle (https://packages.debian.org/sid/python3-cloudpickle)
* python3-torch (https://packages.debian.org/sid/python3-torch)
* python3-pydantic (https://packages.debian.org/sid/python3-pydantic)
* python3-huggingface-hub (application task)
* python3-filelock (https://packages.debian.org/sid/python3-filelock)
* python3-numpy (https://packages.debian.org/sid/python3-numpy)
* python3-packaging (https://packages.debian.org/sid/python3-packaging)
* python3-yaml (https://packages.debian.org/sid/python3-yaml)
* python3-regex (https://packages.debian.org/sid/python3-regex)
* python3-requests (https://packages.debian.org/sid/python3-requests)
* python3-fsspec (https://packages.debian.org/sid/python3-fsspec)
* python3-typing-extensions (https://packages.debian.org/sid/python3-typing-extensions)
* python3-tqdm (https://packages.debian.org/sid/python3-tqdm)
* python3-fastapi (https://packages.debian.org/sid/python3-fastapi)
* python3-openai (https://packages.debian.org/sid/python3-openai)
* python3-astor (https://packages.debian.org/sid/python3-astor)
* python3-dill (https://packages.debian.org/sid/python3-dill)
* python3-einops (https://packages.debian.org/sid/python3-einops)
* python3-zipp (https://packages.debian.org/sid/python3-zipp)
* python3-lark (https://packages.debian.org/sid/python3-lark)
* python3-cloudpickle (https://packages.debian.org/sid/python3-cloudpickle)
* python3-jsonschema (https://packages.debian.org/sid/python3-jsonschema)
* python3-pil (https://packages.debian.org/sid/python3-pil)
* python3-sentencepiece (https://packages.debian.org/sid/python3-sentencepiece)
* python3-tiktoken (https://packages.debian.org/sid/python3-tiktoken)
* python3-msgspec (https://packages.debian.org/sid/python3-msgspec)
* python3-diskcache (https://packages.debian.org/sid/python3-diskcache)
* python3-jinja2 (https://packages.debian.org/sid/python3-jinja2)
* python3-nest-asyncio (https://packages.debian.org/sid/python3-nest-asyncio)
* python3-jsonschema-specifications (https://packages.debian.org/sid/python3-jsonschema-specifications)
* python3-referencing (https://packages.debian.org/sid/python3-referencing)
* python3-rpds-py (https://packages.debian.org/sid/python3-rpds-py)
* python3-pycountry (https://packages.debian.org/sid/python3-pycountry)
* python3-starlette (https://packages.debian.org/sid/python3-starlette)
* python3-anyio (https://packages.debian.org/sid/python3-anyio)
* python3-idna (https://packages.debian.org/sid/python3-idna)
* python3-sniffio (https://packages.debian.org/sid/python3-sniffio)
* python3-protobuf (https://packages.debian.org/sid/python3-protobuf)
* python3-psutil (https://packages.debian.org/sid/python3-psutil)
* py-cpuinfo (https://packages.debian.org/sid/py-cpuinfo)
* python3-aiosignal (https://packages.debian.org/sid/python3-aiosignal)
* python3-msgpack (https://packages.debian.org/sid/python3-msgpack)
* python3-click (https://packages.debian.org/sid/python3-click)
* python3-frozenlist (https://packages.debian.org/sid/python3-frozenlist)
* python3-certifi (https://packages.debian.org/sid/python3-certifi)
* python3-charset-normalizer (https://packages.debian.org/sid/python3-charset-normalizer)
* python3-urllib3 (https://packages.debian.org/sid/python3-urllib3)
* python3-setuptools (https://packages.debian.org/sid/python3-setuptools)
* python3-six (https://packages.debian.org/sid/python3-six)
* python3-torchaudio (https://packages.debian.org/sid/python3-torchaudio)
* python3-torchvision (https://packages.debian.org/sid/python3-torchvision)
* python3-uvicorn (https://packages.debian.org/sid/python3-uvicorn)
* python3-networkx (https://packages.debian.org/sid/python3-networkx)
* python3-triton (https://packages.debian.org/sid/python3-triton)
* python3-sympy   (https://packages.debian.org/sid/python3-sympy)
* python3-pybind11 (https://packages.debian.org/sid/python3-pybind11)
* python3-pytest (https://packages.debian.org/sid/python3-pytest)
* python3-scipy (https://packages.debian.org/sid/python3-scipy)
* python3-numba (https://packages.debian.org/sid/python3-numba)
* python3-safetensors (https://packages.debian.org/sid/python3-safetensors)

```bash
vllm==0.8.1
├── blake3 [required: Any, installed: 1.0.4]
├── transformers [required: >=4.48.2, installed: 4.50.0]
│   ├── tokenizers [required: >=0.21,<0.22, installed: 0.21.1]
├── tokenizers [required: >=0.19.1, installed: 0.21.1]
├── prometheus_client [required: >=0.18.0, installed: 0.21.1]
├── prometheus-fastapi-instrumentator [required: >=7.0.0, installed: 7.1.0]
│   ├── prometheus_client [required: >=0.8.0,<1.0.0, installed: 0.21.1]
├── lm-format-enforcer [required: >=0.10.11,<0.11, installed: 0.10.11]
│   ├── interegular [required: >=0.3.2, installed: 0.3.3]
├── outlines [required: ==0.1.11, installed: 0.1.11]
│   ├── interegular [required: Any, installed: 0.3.3]
│   ├── airportsdata [required: Any, installed: 20250224]
│   └── outlines_core [required: ==0.1.26, installed: 0.1.26]
│       ├── interegular [required: Any, installed: 0.3.3]
├── xgrammar [required: ==0.1.16, installed: 0.1.16]
│   ├── transformers [required: >=4.38.0, installed: 4.50.0]
│   │   ├── tokenizers [required: >=0.21,<0.22, installed: 0.21.1]
│   └── ninja [required: Any, installed: 1.11.1.4]
├── partial-json-parser [required: Any, installed: 0.2.1.1.post5]
├── pyzmq [required: Any, installed: 26.3.0]
├── gguf [required: ==0.10.0, installed: 0.10.0]
├── importlib_metadata [required: Any, installed: 8.6.1]
├── mistral_common [required: >=1.5.4, installed: 1.5.4]
├── compressed-tensors [required: ==0.9.2, installed: 0.9.2]
│   ├── transformers [required: Any, installed: 4.50.0]
│   │   ├── tokenizers [required: >=0.21,<0.22, installed: 0.21.1]
├── depyf [required: ==0.18.0, installed: 0.18.0]
├── watchfiles [required: Any, installed: 1.0.4]
├── python-json-logger [required: Any, installed: 3.3.0]
├── ninja [required: Any, installed: 1.11.1.4]
├── ray [required: >=2.43.0, installed: 2.44.0]
│   ├── frozenlist [required: Any, installed: 1.5.0]
└── xformers [required: ==0.0.29.post2, installed: 0.0.29.post2]
```

Dependency tree becomes simpler, but still, there are some dependencies that are not packaged in the debian repository.

First of all, I will try packaging tokenizers, then I will try to package transformers, because tokenizers is a dependency of transformers and transformers is a dependency of many other packages. I think this is the most important part of the project because transformers is a very popular library and many other libraries depend on it.

At the same time, I will try to package the following packages:
*  depyf [required: ==0.18.0, installed: 0.18.0]
*  watchfiles [required: Any, installed: 1.0.4]
*  python-json-logger [required: Any, installed: 3.3.0]
*  ninja [required: Any, installed: 1.11.1.4]
*  partial-json-parser [required: Any, installed: 0.2.1.1.post5]
*  pyzmq [required: Any, installed: 26.3.0]
*  gguf [required: ==0.10.0, installed: 0.10.0]
*  importlib_metadata [required: Any, installed: 8.6.1]
*  mistral_common [required: >=1.5.4, installed: 1.5.4]
*  frozenlist [required: Any, installed: 1.5.0]
*  interegular [required: Any, installed: 0.3.3]
*  airportsdata [required: Any, installed: 20250224]
*  prometheus_client [required: >=0.8.0,<1.0.0, installed: 0.21.1]
*  xformers [required: ==0.0.29.post2, installed: 0.0.29.post2]
which looks like to have a simple dependency tree. Especially, package which don't have gpu dependency is easy to package and good start for me.

Then, I will finish the project by packaging the following packages:
* prometheus-fastapi-instrumentator [required: >=7.0.0, installed: 7.1.0]
* lm-format-enforcer [required: >=0.10.11,<0.11, installed: 0.10.11]
* outlines_core [required: ==0.1.26, installed: 0.1.26]
* outlines [required: ==0.1.11, installed: 0.1.11] 
* xgrammar [required: ==0.1.16, installed: 0.1.16] 
* compressed-tensors [required: ==0.9.2, installed: 0.9.2]
* ray [required: >=2.43.0, installed: 2.44.0]
which have a more complex dependency tree.

## Availability
My school finishes at the end of May so I can start the project from the beginning of June. I will go back to Japan after school ends. University of Tokyo starts at October so I can work full-time on the project during the summer. 
