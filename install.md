## Installation

The code is developed using Python 3.6 with PyTorch 1.0.0. The code is developed and tested using 2 GPU cards.

1. **Clone this repo.**

   ```shell
   $ git clone https://github.com/charlesCXK/TorchSSC.git
   $ cd TorchSSC
   ```

2. **Install dependencies.**

   **(1) Create a conda environment:**

   ```shell
   $ SET CONDA_RESTORE_FREE_CHANNEL=1
   $ conda env create -f ssc.yaml
   $ conda activate ssc
   ```

   **(2) Install apex 0.1(needs CUDA)**

   ```shell
   $ cd ./furnace/apex
   $ python setup.py install --cpp_ext --cuda_ext
   ```

​
