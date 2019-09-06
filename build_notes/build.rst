Conda requirement file for readthedocs
--------------------------------------

onda create -n pytorch367_cpu python==3.6.7
conda activate pytorch367_cpu
conda install pytorch torchvision cpuonly -c pytorch
cd D:\devel\trw
pip install -e .
pip install -r requirements-dev.txt
conda env export > environment.yml