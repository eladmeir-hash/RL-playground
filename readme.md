Elad Meir - SAC implementation for 2D navigation (Line5)
--------------------------------
-------------------------

Requirement are found on the `reqs` file (feel free to install in any other way the required libraries)

Training
---------
python train.py
with optional flags for     '--n_envs' (default=12)
                            '--total_timesteps' (default=1_000_000)

Saving (automatic)
-----------------
Models are saved in simulation/models folder with a unique timestamp - <model_name>
In the <model_name> folder there will be 
1. <tensorboard> folder (feel free to take a look at the tensorboard metrics, they are valid)
2. the model's weights - model.zip 
3. the model's normalization parameters - model.pkl (VecNormalize)
4. some important parameters in model.json


Evaluation
----------
python eval --checkpoint <directory path>
with optional flags for     '--n_trials' (default: 1000. Number of simulations for metrics evaluation)
                            '--render' (default: None. True for simulation visualization)
                            '--n_render' (default: 10. number of simulations for visualization (default 10), only valid if --render True)
