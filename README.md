# RL for Symbolic Integration

Our code contains training and testing scripts for RL and supervised learning based integration. 

- **RL Integrator** (PPO-trained actor/critic transformer policy).
- **Transformer Integrator** (rule classifier + integration routine).
- **Manual Integrator** Sympy's baseline.

These modules must be cloned directly into the repository -
https://github.com/NVIDIA/apex
https://github.com/facebookresearch/SymbolicMathematics.git
https://github.com/sympy/sympy.git

To prepare datasets extract ALL_data.zip and RUBI_dataset.zip, and run json_conversion.py and rubi_conversion.py

Supervised model training can be performed by running transformer_training.py, RL models can be trained by running rl_integrate.py directly. 

To test models on different datasets run model_testing.py using the --dataset argument followed by the dataset path.