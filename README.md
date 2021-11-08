# VIGN_
Variational Integrator Graph Networks for Learning Energy Conserving Dynamical Systems (https://arxiv.org/abs/2004.13688)

We run an extensive ablation across both graph and non-graph innovations in learning dynamics from noisy, energy conserving trajectories.

Requirements:

```
pip install graph_nets "tensorflow>=1.15,<2" "dm-sonnet<2" "tensorflow_probability<0.9"
```

For each system investigated, the parameter sweep is defined in its respective .sh file.

Use INFERENCE_FINAL.ipynb to visualize the state rollouts and use visualizer.ipynb to plot the average results.

Note: The code is not optimized, please email shaandesai@live.com for further questions.
