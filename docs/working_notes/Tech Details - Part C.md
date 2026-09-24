For this replication, I recommend scikit-learn + PyTorch, with NumPy/SciPy for the custom estimators and statsmodels for inference. Your existing environment already covers much of the data work.
I checked the workspace machine: it reports an NVIDIA RTX 5080, approximately 16 GB VRAM, and driver 610.74. The CUDA instructions below target that machine. If you mean a different laptop, its GPU and driver need to be checked separately.
1. Packages to add
Package	Purpose in this project
scikit-learn	PCR/PLS components, linear-model benchmarks, random forests, regression-tree building blocks
torch	NN1–NN5, Adam, batch normalization, GPU computation
statsmodels	Factor regressions, alpha estimates, Newey–West/HAC inference
pyarrow	Efficient Parquet storage for the stock-month panel
openpyxl	Read the Goyal macroeconomic .xlsx workbook
matplotlib	Research figures
seaborn	Convenient heatmaps and statistical plots
tqdm	Progress reporting for annual fits and tuning loops
pytest	Tests for important calculations
ipykernel	Optional: use this environment in notebooks


Statsmodels supplies HAC covariance estimators; pandas supports Parquet through PyArrow and Excel through openpyxl. Statsmodels documentation, pandas dependencies.
Keep your existing numpy, scipy, pandas, and WRDS packages. Since NumPy and SciPy will be direct dependencies of our research code, explicitly record them in the project too.
From the project root, run:
uv add numpy scipy scikit-learn statsmodels pyarrow openpyxl matplotlib seaborn tqdm
uv add --dev pytest
If you use notebooks:
uv add --dev ipykernel
I recommend uv add because your project already has pyproject.toml: it records dependencies and updates the lockfile. uv pip install alone does not record them as project dependencies. uv dependency management.
2. Installing packages does not automatically reproduce GKX
Our likely implementation mapping is:
GKX method	Implementation approach
OLS	NumPy/SciPy or scikit-learn
PCR	PCA plus linear regression
PLS	Scikit-learn implementation, checked against the paper
Random forest	Scikit-learn
Huber linear regression / ENet+H	Custom objective and optimizer where needed
Spline GLM+H with group lasso	Custom basis and proximal optimization
GBRT+H	Check the library algorithm against Appendix Algorithm 4; customize if needed
NN1–NN5	PyTorch


For example, scikit-learn’s HuberRegressor includes a scale parameter and an $L_2$ penalty. It is not automatically the same estimator as GKX’s Huber specification. HuberRegressor documentation.
We do not need TensorFlow, XGBoost, LightGBM, or a separate group-lasso package to get started. The listed stack is sufficient to implement the paper’s models.
3. What “installing CUDA” means here
For ordinary PyTorch training, the setup consists of:
Component	Required?	How it is supplied
NVIDIA GPU driver	Yes	Installed in Windows
CUDA-enabled PyTorch	Yes	Installed into your Python environment
CUDA runtime and cuDNN libraries	Yes	Supplied by the PyTorch binary installation
Standalone CUDA Toolkit, including nvcc	Generally no	Needed for compiling CUDA code/extensions
Visual Studio C++ build tools	Generally no	Needed for relevant source builds


You do not need to separately install CUDA Toolkit or cuDNN for these neural networks. PyTorch’s prebuilt binaries provide their runtime dependencies; a compatible NVIDIA driver is needed. PyTorch maintainer explanation.
Also, the CUDA version displayed by nvidia-smi describes driver compatibility. It does not prove that a standalone CUDA Toolkit is installed. NVIDIA explanation.
4. Check the target Windows machine
Run:
nvidia-smi
python --version
For the detected RTX 5080 machine, I recommend PyTorch 2.13.0 with its CUDA 13.0 build. This combination is listed in PyTorch’s official Windows installation instructions, and your detected driver is sufficiently recent. Python 3.12 is supported. PyTorch builds, Windows Python support, NVIDIA compatibility.
If the target laptop has an NVIDIA GPU but its driver is missing or outdated, install the appropriate Windows notebook driver from NVIDIA’s driver page, restart, and run nvidia-smi again.
For the machine I detected, a driver update is unnecessary for the proposed build.
5. Install CUDA-enabled PyTorch reproducibly with uv
First, append this configuration to your project’s pyproject.toml. If these sections already exist, merge the entries rather than creating duplicate sections:
[tool.uv.sources]
torch = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
Then run:
uv add "torch==2.13.0"
This directs torch to the CUDA 13.0 package index while keeping ordinary dependencies on PyPI. explicit = true restricts the special index to explicitly assigned packages. uv index configuration.
Only torch is needed here; torchvision and torchaudio are unnecessary for tabular return prediction.
6. Verify actual GPU computation
Paste this entire block into PowerShell from the project root:
@'
import sys
import torch
from torch import nn

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "PyTorch cannot access CUDA."
print("GPU:", torch.cuda.get_device_name(0))

model = nn.Sequential(
    nn.Linear(920, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Linear(32, 1),
).to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x = torch.randn(256, 920, device="cuda")
y = torch.randn(256, 1, device="cuda")

optimizer.zero_grad()
loss = (model(x) - y).square().mean()
loss.backward()
optimizer.step()
torch.cuda.synchronize()

print("GPU forward/backward/optimizer step passed.")
print("Loss:", loss.item())
'@ | uv run python -
This checks more than GPU detection: it runs a small network’s forward pass, gradient calculation, and Adam update on the GPU. It is an installation check, not the complete GKX training procedure.
Then check package consistency:
uv pip check
If CUDA is unavailable:
Observation	Likely issue
torch.version.cuda is None	CPU-only PyTorch was installed
nvidia-smi fails	Driver, GPU availability, or executable-path issue
Python executable points outside the project	Wrong environment
CUDA is detected but computation reports an unsupported kernel	GPU architecture and installed PyTorch build mismatch


7. Preserve the environment
Keep pyproject.toml and uv.lock in Git. On subsequent setups, use:
uv sync --locked
The GPU will primarily accelerate the PyTorch neural networks. Installing CUDA does not automatically move pandas operations or scikit-learn random forests onto the GPU. Scikit-learn GPU support.
I have not installed packages or changed your project files.