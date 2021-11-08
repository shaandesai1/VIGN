#!/bin/bash

#pendulum
for noise_var in 0.05 0
do
 for deltat in 0.1
 do
  for INTEG_STEP in 10
  do
    for INTEG in rk4 vi4
    do
      for NONLIN in softplus
      do
        python train.py -ni 5000 -long_range 1 -nonlinearity $NONLIN -hidden_dims 300 -num_hdims 2 -lr_iters 10000 -n_test_traj 25 -n_train_traj 20 -tmax 20 -dt $deltat -srate $deltat -num_nodes 2 -dname n_grav -noise_std $noise_var -integrator $INTEG -fname expt_a -integ_step $INTEG_STEP
      done
    done
  done
 done
done
