Set up conda environment
------------

Set up directly using `environment.yml`: 

    conda env create -f environment.yml
    conda activate IWAEhighdim

Alternatively, set up using the following commands: 
    
    conda create --name IWAEhighdim python=3.8
    conda activate IWAEhighdim
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
    conda install scipy pandas tqdm matplotlib seaborn
    conda install -c conda-forge pytorch-lightning=1.6.4
    conda install -c conda-forge scikit-learn hydra-core wandb arviz
    pip install slurm-gpustat hydra-submitit-launcher


Using the code
---------------------

MNIST:

1. Single run: 
`python vae.py loss_name=alpha_iwae_dreg_loss_v6 alpha=0 N=100`
    
    To turn off wandb logging, use the option `logger=CSV`

2. Sweep across parameters (on slurm clusters):
`python vae.py loss_name=alpha_iwae_dreg_loss_v6 alpha=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 N=100 -m`