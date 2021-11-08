"""
Author: ****
code to build graph based models for VIGN

Some aspects adopted from: https://github.com/steindoringi/Variational_Integrator_Networks/blob/master/models.py
"""
from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt
import tensorflow as tf
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow.keras as tfk
from tensorflow_probability import distributions as tfd
import os


def create_loss_ops(true, predicted):
    """MSE loss"""
    loss_ops = tf.reduce_mean((true - predicted) ** 2)
    return loss_ops


def log_likelihood_y(y, y_rec, log_noise_var):
    """ noise loss"""
    noise_var = tf.nn.softplus(log_noise_var) * tf.ones_like(y_rec)
    py = tfd.Normal(y_rec, noise_var)
    log_py = py.log_prob(y)
    log_py = tf.reduce_sum(log_py, [0])
    log_lik = tf.reduce_mean(log_py)
    return log_lik


def choose_integrator(method):
    """
    returns integrator for dgn/hnn from utils
    args:
     method (str): 'rk1' or 'rk4'
    """
    if method == 'rk1':
        return rk1

    elif method == 'rk2':
        return rk2

    elif method == 'rk3':
        return rk3

    elif method == 'rk4':
        return rk4

    elif method == 'vi1':
        return vi1

    elif method == 'vi2':
        return vi2

    elif method == 'vi3':
        return vi3

    elif method == 'vi4':
        return vi4


def choose_integrator_nongraph(method):
    """
    returns integrator for dgn/hnn from utils
    args:
     method (str): 'rk1' or 'rk4'
    """
    if method == 'rk1':
        return rk1ng
    elif method == 'rk2':
        return rk2ng
    elif method == 'rk3':
        return rk3ng
    elif method == 'rk4':
        return rk4ng
    elif method == 'vi1':
        return vi1ng
    elif method == 'vi2':
        return vi2ng
    elif method == 'vi3':
        return vi3ng
    elif method == 'vi4':
        return vi4ng


class nongraph_model(object):

    def __init__(self, sess, deriv_method, num_nodes, BS, integ_meth, expt_name, lr,
                 noisy, spatial_dim, dt, num_hdims=2, hidden_dims=256, lr_iters=10000, nonlinearity='softplus',
                 long_range=False,integ_step=2):
        """
            Builds a tensorflow classic non-graph model object
            Args:
                sess (tf.session): instantiated session
                deriv_method (str): one of hnn,dgn,vin_rk1,vin_rk4,vin_rk1_lr,vin_rk4_lr
                num_nodes (int): number of particles
                BS (int): batch size
                integ_method (str): rk1 or rk4 for now, though vign has higher order integrators
                expt_name (str): identifier for specific experiment
                lr (float): learning rate
                is_noisy (bool): flag for noisy data
                spatial_dim (int): the dimension of state vector for 1 particle (e.g. 2, [q,qdot] in spring system)
                dt (float): sampling rate
            """

        self.hidden_dims = hidden_dims
        self.nonlinearity = nonlinearity
        self.long_range = long_range
        self.lr_iters = lr_iters
        self.num_hdims = num_hdims
        self.lr_scale = 0.5
        self.integ_step = integ_step

        self.sess = sess
        self.deriv_method = deriv_method
        self.num_nodes = num_nodes
        self.BS = BS
        self.BS_test = 1
        self.integ_method = integ_meth
        self.expt_name = expt_name
        self.lr = lr
        self.spatial_dim = spatial_dim
        self.dt = dt
        self.is_noisy = noisy
        self.log_noise_var = None
        if self.num_nodes == 1:
            self.activate_sub = False
        else:
            self.activate_sub = True
        self.output_plots = False
        self.M = tf.transpose(self.permutation_tensor(self.spatial_dim * self.num_nodes))
        self._build_net()

    def _build_net(self):
        """
        initializes all tf placeholders/networks/losses
        """

        if self.is_noisy:
            self.log_noise_var = tf.Variable([0.], dtype=tfk.backend.floatx())

        if self.nonlinearity == 'softplus':
            self.nonlin = tf.nn.softplus
        elif self.nonlinearity == 'tanh':
            self.nonlin = tf.nn.tanh
        elif self.nonlinearity == 'cos':
            self.nonlin = tf.nn.cos
        elif self.nonlinearity == 'sin':
            self.nonlin == tf.nn.sin
        else:
            raise ValueError('The suggested nonlinearity is not defined')

        self.mlp_base = snt.nets.MLP([self.hidden_dims] * self.num_hdims, use_bias=True, activation=self.nonlin,
                                     activate_final=True, name='mlp')

        # self.h1 = snt.Linear(output_size=self.hidden_dims, use_bias=True, name='h1')
        # self.h2 = snt.Linear(output_size=self.hidden_dims, use_bias=True, name='h2')

        if self.deriv_method == 'dn':
            self.h3 = snt.Linear(output_size=self.spatial_dim * self.num_nodes, use_bias=False, name='h3')
        else:
            self.h3 = snt.Linear(output_size=1, use_bias=False, name='h3')

        self.mlp = snt.Sequential([
            self.mlp_base,
            self.h3
        ])

        self.input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, self.spatial_dim * self.num_nodes])
        self.test_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, self.spatial_dim * self.num_nodes])
        self.ground_truth_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.integ_step-1,None, self.spatial_dim * self.num_nodes])

        integ = choose_integrator_nongraph(self.integ_method)

        if self.deriv_method == 'dn':
            if self.long_range:
                self.next_step = self.future_pred(integ, self.deriv_fun_dn, self.input_ph, self.dt)
            else:
                self.next_step = integ(self.deriv_fun_dn, self.input_ph, self.dt)
            self.test_next_step = integ(self.deriv_fun_dn, self.test_ph, self.dt)

        elif self.deriv_method == 'hnn':
            if self.long_range:
                self.next_step = self.future_pred(integ, self.deriv_fun_hnn, self.input_ph, self.dt)
            else:
                self.next_step = integ(self.deriv_fun_hnn, self.input_ph, self.dt)
            self.test_next_step = integ(self.deriv_fun_hnn, self.test_ph, self.dt)

        elif self.deriv_method == 'pnn':
            if self.long_range:
                self.next_step = self.future_pred(integ, self.deriv_fun_pnn, self.input_ph, self.dt)
            else:
                self.next_step = integ(self.deriv_fun_pnn, self.input_ph, self.dt)
            self.test_next_step = integ(self.deriv_fun_pnn, self.test_ph, self.dt)

        else:
            raise ValueError("the derivative generator is incorrect, should be dn,hnn or pn")

        if self.is_noisy:
            self.loss_op_tr = -log_likelihood_y(self.next_step, self.ground_truth_ph, self.log_noise_var)
            self.loss_op_test = -log_likelihood_y(self.next_step, self.ground_truth_ph, self.log_noise_var)
        else:
            self.loss_op_tr = self.create_loss_ops(self.next_step, self.ground_truth_ph)
            self.loss_op_test = self.create_loss_ops(self.next_step, self.ground_truth_ph)

        global_step = tf.Variable(0, trainable=False)
        rate = self.lr#tf.compat.v1.train.exponential_decay(self.lr, global_step, self.lr_iters, self.lr_scale, staircase=False)
        # tf.train.AdamW()
        optimizer = tf.train.AdamOptimizer(rate)
        self.step_op = optimizer.minimize(self.loss_op_tr, global_step=global_step)

    def future_pred(self, integ, deriv_fun, state_init, dt):
        accum = []
        first_dim = state_init.shape[0]
        q_init = state_init
        print(f'qinitshape {q_init.shape}')
        accum.append(q_init)
        # print(self.BS)
        for _ in range(self.BS):
            xtp1 = integ(deriv_fun, accum[-1], dt)

            accum.append(xtp1)

        yhat = tf.stack(accum, 0)
        print(f'yhatshape {yhat.shape}')
        return yhat[1:]

    def create_loss_ops(self, true, predicted):
        """MSE loss"""
        loss_ops = tf.reduce_mean((true - predicted) ** 2)
        return loss_ops

    def deriv_fun_dn(self, xt):
        output_nodes = self.mlp(xt)
        return output_nodes

    def deriv_fun_hnn(self, xt):
        with tf.GradientTape() as g:
            g.watch(xt)
            output_nodes = self.mlp(xt)
        dH = g.gradient(output_nodes, xt)
        return tf.concat(
            [dH[:, int(self.spatial_dim * self.num_nodes / 2):], -dH[:, :int(self.spatial_dim * self.num_nodes / 2)]],
            1)

    def deriv_fun_pnn(self, xt):
        qvals = xt[:, :int(self.spatial_dim * self.num_nodes / 2)]
        pvals = xt[:, int(self.spatial_dim * self.num_nodes / 2):]
        with tf.GradientTape() as g:
            g.watch(qvals)
            output_nodes = self.mlp(qvals)
        dH = g.gradient(output_nodes, qvals)
        return tf.concat([pvals, -dH], 1)

    def permutation_tensor(self, n):
        M = None
        M = tf.eye(n)
        M = tf.concat([M[n // 2:], -M[:n // 2]], 0)
        return M

    def train_step(self, input_batch, true_batch):

        train_feed = {self.input_ph: input_batch,
                      self.ground_truth_ph: true_batch,
                      }
        train_ops = [self.loss_op_tr, self.next_step, self.step_op]
        loss, next_pred, _ = self.sess.run(train_ops, feed_dict=train_feed)
        return loss, next_pred

    def valid_step(self, input_batch, true_batch):

        train_feed = {self.input_ph: input_batch,
                      self.ground_truth_ph: true_batch,
                      }
        train_ops = [self.loss_op_test, self.next_step]
        loss, next_pred = self.sess.run(train_ops, feed_dict=train_feed)

        return loss, next_pred

    def test_step(self, input_batch, true_batch, steps):
        # figures relegated to jupyter notebook infengine
        stored_states = [input_batch.astype(np.float32)]
        for i in range(steps):
            test_feed = {self.test_ph: stored_states[-1],
                         }
            test_ops = [self.test_next_step]

            yhat = self.sess.run(test_ops, feed_dict=test_feed)
            stored_states.append(yhat[0])

        preds = tf.stack(stored_states, 0).eval(session=self.sess)

        error = mean_squared_error(preds[1:].flatten(), true_batch.flatten())

        return error, preds[1:, :]


class graph_model(object):
    """
    Builds a tensorflow graph model object
    Args:
        sess (tf.session): instantiated session
        deriv_method (str): one of hnn,dgn,vin_rk1,vin_rk4,vin_rk1_lr,vin_rk4_lr
        num_nodes (int): number of particles
        BS (int): batch size
        integ_method (str): rk1 or rk4 for now, though vign has higher order integrators
        expt_name (str): identifier for specific experiment
        lr (float): learning rate
        is_noisy (bool): flag for noisy data
        spatial_dim (int): the dimension of state vector for 1 particle (e.g. 2, [q,qdot] in spring system)
        dt (float): sampling rate
        eflag (bool): whether to use extra input in building graph (default=True)
    """


    def __init__(self, sess, deriv_method, num_nodes, BS, integ_meth, expt_name, lr,
                 noisy, spatial_dim, dt, num_hdims=2, hidden_dims=32, lr_iters=10000, nonlinearity='softplus',
                 long_range=False,integ_step=2,num_test_traj=25):

        self.hidden_dims = hidden_dims
        self.nonlinearity = nonlinearity
        self.long_range = long_range
        self.lr_iters = lr_iters
        self.num_hdims = num_hdims
        self.lr_scale = 0.5
        self.integ_step = integ_step

        self.test_traj = num_test_traj
        self.sess = sess
        self.deriv_method = deriv_method
        self.num_nodes = num_nodes
        self.BS = BS
        self.BS_test = 1
        self.integ_method = integ_meth
        self.expt_name = expt_name
        self.lr = lr
        self.spatial_dim = spatial_dim
        self.dt = dt
        self.eflag = False
        self.is_noisy = noisy
        self.log_noise_var = None
        if self.num_nodes == 1:
            self.activate_sub = False
        else:
            self.activate_sub = False
        self.output_plots = False
        self._build_net()

    def _build_net(self):
        """
        initializes all tf placeholders/graph networks/losses
        """

        if self.is_noisy:
            self.log_noise_var = tf.Variable([0.], dtype=tfk.backend.floatx())


        if self.nonlinearity == 'softplus':
            self.nonlin = tf.nn.softplus
        elif self.nonlinearity == 'tanh':
            self.nonlin = tf.nn.tanh
        elif self.nonlinearity == 'cos':
            self.nonlin = tf.nn.cos
        elif self.nonlinearity == 'sin':
            self.nonlin == tf.nn.sin
        else:
            raise ValueError('The suggested nonlinearity is not defined')



        self.out_to_global = snt.Linear(output_size=1, use_bias=False, name='out_to_global')
        self.out_to_node = snt.Linear(output_size=self.spatial_dim, use_bias=True, name='out_to_node')

        self.graph_network = modules.GraphNetwork(
            edge_model_fn=lambda: snt.nets.MLP([self.hidden_dims]*self.num_hdims, activation=self.nonlin, activate_final=True),
            node_model_fn=lambda: snt.nets.MLP([self.hidden_dims]*self.num_hdims, activation=self.nonlin, activate_final=True),
            global_model_fn=lambda: snt.nets.MLP([self.hidden_dims]*self.num_hdims, activation=self.nonlin, activate_final=True),
        )

        self.base_graph_tr = tf.compat.v1.placeholder(tf.float32,
                                                      shape=[self.num_nodes * self.BS, self.spatial_dim])
        self.ks_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS, self.num_nodes])
        self.ms_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS, self.num_nodes])

        self.true_dq_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.integ_step-1,self.BS*self.num_nodes, self.spatial_dim])

        self.test_graph_ph = tf.compat.v1.placeholder(tf.float32,
                                                      shape=[self.num_nodes * self.BS_test, self.spatial_dim])
        self.test_ks_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS_test, self.num_nodes])
        self.test_ms_ph = tf.compat.v1.placeholder(tf.float32, shape=[self.BS_test, self.num_nodes])

        integ = choose_integrator(self.integ_method)

        if self.deriv_method == 'dgn':
            if self.long_range:
                self.next_step = self.future_pred(integ, self.deriv_fun_dgn, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                   self.num_nodes)
            else:
                self.next_step = integ(self.deriv_fun_dgn, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                   self.num_nodes)
            self.test_next_step = integ(self.deriv_fun_dgn, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                        self.dt, 1, self.num_nodes)
        elif self.deriv_method == 'hogn':
            if self.long_range:
                self.next_step = self.future_pred(integ, self.deriv_fun_hogn, self.base_graph_tr, self.ks_ph, self.ms_ph,
                                                  self.dt, self.BS,
                                                  self.num_nodes)
            else:
                self.next_step = integ(self.deriv_fun_hogn, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                   self.num_nodes)
            self.test_next_step = integ(self.deriv_fun_hogn, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                        self.dt, 1, self.num_nodes)
        elif self.deriv_method == 'pgn':
            if self.long_range:
                self.next_step = self.future_pred(integ, self.deriv_fun_pgn, self.base_graph_tr, self.ks_ph, self.ms_ph,
                                                  self.dt, self.BS,
                                                  self.num_nodes)
            else:
                self.next_step = integ(self.deriv_fun_pgn, self.base_graph_tr, self.ks_ph, self.ms_ph, self.dt, self.BS,
                                   self.num_nodes)
            self.test_next_step = integ(self.deriv_fun_pgn, self.test_graph_ph, self.test_ks_ph, self.test_ms_ph,
                                        self.dt, 1, self.num_nodes)
        else:
            raise ValueError("the derivative generator is incorrect, should be dgn,hogn or pgn")

        if self.is_noisy:
            self.loss_op_tr = -log_likelihood_y(self.next_step, self.true_dq_ph, self.log_noise_var)
            self.loss_op_test = -log_likelihood_y(self.next_step, self.true_dq_ph, self.log_noise_var)
        else:
            self.loss_op_tr = self.create_loss_ops(self.next_step, self.true_dq_ph)
            self.loss_op_test = self.create_loss_ops(self.next_step, self.true_dq_ph)

        global_step = tf.Variable(0, trainable=False)
        rate = self.lr#tf.compat.v1.train.exponential_decay(self.lr, global_step, self.lr_iters, self.lr_scale, staircase=False)
        optimizer = tf.train.AdamOptimizer(rate)
        self.step_op = optimizer.minimize(self.loss_op_tr, global_step=global_step)

    def future_pred(self, integ, deriv_fun, state_init, ks, ms, dt, bs, num_nodes):
        """
        only used with long range rollout (i.e. neuralODE without adjoint method) - future step predictions
        """
        accum = []

        q_init = state_init
        accum.append(q_init)

        for _ in range(self.integ_step-1):
            xtp1 = integ(deriv_fun, accum[-1], ks, ms, dt, bs, num_nodes)
            accum.append(xtp1)

        yhat = tf.stack(accum, 0)
        print(f'yhatshape {yhat.shape}')

        return yhat[1:]

    def create_loss_ops(self, true, predicted):
        """MSE loss"""
        loss_ops = tf.reduce_mean((true - predicted) ** 2)
        return loss_ops

    def base_graph(self, input_features, ks, ms, num_nodes):
        """builds graph for every group of particles"""
        # Node features for graph 0.
        if self.eflag:
            nodes_0 = tf.concat([input_features, tf.reshape(ms, [num_nodes, 1]), tf.reshape(ks, [num_nodes, 1])], 1)
        else:
            nodes_0 = input_features

        senders_0 = []
        receivers_0 = []
        # edges_0 = []
        an = np.arange(0, num_nodes, 1)
        for i in range(len(an)):
            for j in range(i + 1, len(an)):
                senders_0.append(i)
                senders_0.append(j)
                receivers_0.append(j)
                receivers_0.append(i)

        data_dict_0 = {
            "nodes": nodes_0,
            "senders": senders_0,
            "receivers": receivers_0
        }

        return data_dict_0

    def deriv_fun_dgn(self, xt, ks, ms, bs, n_nodes):
        if self.activate_sub == True:
            sub_vecs = self.sub_mean(xt, bs)
        else:
            sub_vecs = xt

        input_vec = tf.concat(sub_vecs, 0)
        vec2g = [self.base_graph(input_vec[n_nodes * i:n_nodes * (i + 1)], ks[i], ms[i], n_nodes) for i in range(bs)]
        vec2g = utils_tf.data_dicts_to_graphs_tuple(vec2g)
        vec2g = utils_tf.set_zero_global_features(vec2g, 1)
        vec2g = utils_tf.set_zero_edge_features(vec2g, 1)
        output_graphs = self.graph_network(vec2g)
        new_node_vals = self.out_to_node(output_graphs.nodes)
        return new_node_vals

    def deriv_fun_hogn(self, xt, ks, ms, bs, n_nodes):
        if self.activate_sub == True:
            sub_vecs = self.sub_mean(xt, bs)
        else:
            sub_vecs = xt

        input_vec = tf.concat(sub_vecs, 0)
        with tf.GradientTape() as g:
            g.watch(input_vec)
            vec2g = [self.base_graph(input_vec[n_nodes * i:n_nodes * (i + 1)], ks[i], ms[i], n_nodes) for i in
                     range(bs)]
            vec2g = utils_tf.data_dicts_to_graphs_tuple(vec2g)
            vec2g = utils_tf.set_zero_global_features(vec2g, 1)
            vec2g = utils_tf.set_zero_edge_features(vec2g, 1)
            output_graphs = self.graph_network(vec2g)
            global_vals = self.out_to_global(output_graphs.globals)
        dUdq = g.gradient(global_vals, input_vec)

        dqdt = dUdq[:, int(self.spatial_dim / 2):]
        dpdt = -dUdq[:, :int(self.spatial_dim / 2)]
        dHdin = tf.concat([dqdt, dpdt], 1)
        return dHdin

    def sub_mean(self, xt, bs):
        init_x = xt[:, :int(self.spatial_dim / 2)]
        # means = tf.reduce_mean(init_x, 0)
        # new_means = tf.transpose(tf.reshape(tf.repeat(means, init_x.shape[0]), (int(self.spatial_dim / 2), -1)))
        # return tf.concat([init_x - new_means, xt[:, int(self.spatial_dim / 2):]], 1)
        init_x = tf.reshape(init_x, (bs, self.num_nodes, int(self.spatial_dim / 2)))
        means = tf.reshape(tf.reduce_mean(init_x, 1), (-1, int(self.spatial_dim / 2)))
        new_means = tf.repeat(means, self.num_nodes, 0)
        # print(new_means.shape)
        return tf.concat([xt[:, :int(self.spatial_dim / 2)] - new_means, xt[:, int(self.spatial_dim / 2):]], 1)

    def deriv_fun_pgn(self, xt, ks, ms, bs, n_nodes):
        if self.activate_sub == True:
            sub_vecs = self.sub_mean(xt, bs)
        else:
            sub_vecs = xt

        input_vec = tf.concat(sub_vecs, 0)
        q = input_vec[:, :int(self.spatial_dim / 2)]
        p = input_vec[:, int(self.spatial_dim / 2):]

        with tf.GradientTape() as g:
            g.watch(q)
            # if bs == 1:
            #     vec2g = [self.base_graph(q[n_nodes * i:n_nodes * (i + 1)], ks, ms, n_nodes) for i in range(bs)]
            # else:
            vec2g = [self.base_graph(q[n_nodes * i:n_nodes * (i + 1)], ks[i], ms[i], n_nodes) for i in
                     range(bs)]

            vec2g = utils_tf.data_dicts_to_graphs_tuple(vec2g)
            vec2g = utils_tf.set_zero_global_features(vec2g, 1)
            vec2g = utils_tf.set_zero_edge_features(vec2g, 1)
            output_graphs = self.graph_network(vec2g)
            global_vals = self.out_to_global(output_graphs.globals)

        dUdq = g.gradient(global_vals, q)

        return tf.concat([p, -dUdq], 1)

    def train_step(self, input_batch, true_batch, ks, mass):

        train_feed = {self.base_graph_tr: input_batch,
                      self.true_dq_ph: true_batch,
                      self.ks_ph: ks,
                      self.ms_ph: mass}
        train_ops = [self.loss_op_tr, self.next_step, self.step_op]
        loss, next_pred, _ = self.sess.run(train_ops, feed_dict=train_feed)

        return loss, next_pred

    def valid_step(self, input_batch, true_batch, ks, mass):
        train_feed = {self.base_graph_tr: input_batch,
                      self.true_dq_ph: true_batch,
                      self.ks_ph: ks,
                      self.ms_ph: mass}
        train_ops = [self.loss_op_test,self.next_step]
        loss,next_pred = self.sess.run(train_ops, feed_dict=train_feed)

        return loss, next_pred

    def test_step(self, input_batch, true_batch, ks, mass, steps):
        # figures relegated to jupyter notebook infengine
        stored_states = [input_batch.astype(np.float32)]
        for i in range(steps):
            test_feed = {self.test_graph_ph: stored_states[-1],
                         self.test_ks_ph: ks,
                         self.test_ms_ph: mass}
            test_ops = [self.test_next_step]

            yhat = self.sess.run(test_ops, feed_dict=test_feed)
            stored_states.append(yhat[0])

        preds = tf.stack(stored_states, 0).eval(session=self.sess)

        error = ((preds[1:, :] - true_batch[:, :])**2).mean()

        if self.output_plots is True:
            data_dir = 'data/plots/' + self.expt_name + '/' + str(self.lr) + '/' + str(self.integ_method) + '/'

            if not os.path.exists(data_dir):
                print('non existent')
                os.makedirs(data_dir)

            plt.figure(figsize=(15, 10))
            nv = preds[:, :2]
            gt = true_batch[:, :2]
            plt.scatter(nv[:, 0], nv[:, 1], label=self.deriv_method, c='blue')
            plt.scatter(gt[:, 0], gt[:, 1], label='gt', c='black', alpha=0.5)
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(self.deriv_method + str(self.lr) + ' graphic space evolution')
            plt.savefig(data_dir + 'graphic' + self.deriv_method)

            plt.figure(figsize=(15, 10))
            nv = preds[:, :2]
            gt = true_batch[:, :2]
            plt.scatter(nv[::5, 0], nv[::5, 1], label=self.deriv_method, c='blue')
            plt.scatter(gt[::5, 0], gt[::5, 1], label='gt', c='black', alpha=0.5)
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(self.deriv_method + str(self.lr) + ' graphic space evolution')
            plt.savefig(data_dir + 'graphic' + self.deriv_method + 'onetraj')

        return error, preds[1:]
