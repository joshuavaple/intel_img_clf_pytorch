# intel-image-classification-pytorch
 Pytorch Implementation of CNN for image classification on Intel dataset

# Initial Setup
## Virtual Environment - PowerShell
- Note that this repo used python 3.9.13
- Create a virtual environment in the root folder: `<local machine path to desired python version>\python.exe -m venv myvenv` or simply `python -m venv venv`if you are sure about your installed/default python version.
- Activate the environment: `myvenv\Scripts\activate.ps1`
- If there is no GPU, install the requirements: `pip install -r requirements.txt`
- If there is GPU:
    - Go to `https://pytorch.org/get-started/locally/`, configure your OS and method of installing. Run the generated command in the terminal. E.g., `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116`
    - Install `pip install -r requirements_no_torch.txt`. This will install all remaining required libraries except torch and torchvision above
    - Check if CUDA is available: 
        `import torch`
        `print(torch.backends.cudnn.enabled)`
        `print(torch.cuda.is_available())`
    - Check current CUDA version and GPU info:
        `nvidia-smi`
