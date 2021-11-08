"""
Author: ***
Code to build simulated physics datasets

Adapted,in part, from https://github.com/greydanus/hamiltonian-nn
"""

import pickle
from scipy.integrate import solve_ivp as rk
import autograd
from autograd.numpy import cos, sin
solve_ivp = rk
import numpy as np
from utils import *

def get_dataset(data_name, expt_name, num_samples, num_particles, T_max, dt, srate, noise_std=0, seed=0, pixels=False):
    """
    dataset_name, expt_name, n_test_traj, num_nodes, T_max, dt, srate, 0, 11
    Args:
        data_name: str, from list 'mass_spring','n_spring','n_grav','pendulum','dpendulum','heinon'
        expt_name: directory to save dataset
        num_samples: total number of initial conditions to sample
        num_particles: number of particles in given system
        T_max: maximimum time to integrate to
        dt: integration time step
        srate: subsampling rate
        noise_std: noise scaling on gaussian
        seed: random seed for numpy
        pixels: generate pixel data instead of vectors (for future work)
    """

    dataset_list = ['mass_spring', 'n_spring', 'n_grav', 'pendulum', 'heinon','three_body']
    if data_name not in dataset_list:
        raise ValueError('data name not in data list')

    if data_name == 'mass_spring':
        return mass_spring(expt_name, num_samples, num_particles, T_max, dt, srate, noise_std, seed)
    if data_name == 'n_spring':
        return spring_particle(expt_name, num_samples, num_particles, T_max, dt, srate, noise_std, seed)
    if data_name == 'n_grav':
        return grav_n(expt_name, num_samples, num_particles, T_max, dt, srate, noise_std, seed)
    if data_name == 'pendulum':
        return pendulum(expt_name, num_samples, num_particles, T_max, dt, srate, noise_std, seed)
    if data_name == 'heinon':
        return heinon_heiles(expt_name, num_samples, num_particles, T_max, dt, srate, noise_std, seed)
    if data_name == 'three_body':
        return three_body(expt_name, num_samples, num_particles, T_max, dt, srate, noise_std, seed)


def heinon_heiles(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate, noise_std, seed):
    """heinon heiles data generator"""

    def hamiltonian_fn(coords):
        x, y, px, py = np.split(coords, 4)
        lambda_ = 1
        H = 0.5 * px ** 2 + 0.5 * py ** 2 + 0.5 * (x ** 2 + y ** 2) + lambda_ * (
                (x ** 2) * y - (y ** 3) / 3)
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dxdt, dydt, dpxdt, dpydt = np.split(dcoords, 4)
        S = np.concatenate([dpxdt, dpydt, -dxdt, -dydt], axis=-1)
        return S

    def get_trajectory(t_span=[0, 3], timescale=0.01, ssr=sub_sample_rate, radius=None, y0=None, noise_std=0.1,
                       **kwargs):

        # get initial state
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        px = np.random.uniform(-.5, .5)
        py = np.random.uniform(-.5, .5)

        y0 = np.array([x, y, px, py])

        spring_ivp = rk(lambda t, y: dynamics_fn(t, y), t_span, y0,
                        t_eval=np.arange(0, t_span[1], timescale),
                        rtol=1e-12, method='DOP853')
        accum = spring_ivp.y.T
        ssr = int(ssr / timescale)
        accum = accum[::ssr]
        daccum = [dynamics_fn(None, accum[i]) for i in range(accum.shape[0])]
        energies = []
        for i in range(accum.shape[0]):
            energies.append(np.sum(hamiltonian_fn(accum[i])))

        return accum, np.array(daccum), energies

    def get_dataset(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate, seed=seed, test_split=0.5,
                    **kwargs):
        data = {'meta': locals()}

        # randomly sample inputs
        np.random.seed(seed)
        data = {}
        ssr = int(sub_sample_rate / dt)

        xs, dxs, energies, ks, ms = [], [], [], [], []
        for s in range(num_trajectories):
            x, dx, energy = get_trajectory(t_span=[0, T_max], timescale=dt, ssr=sub_sample_rate)

            x += np.random.randn(*x.shape) * noise_std
            dx += np.random.randn(*dx.shape) * noise_std

            xs.append(x)
            dxs.append(dx)
            energies.append(energy)
            ks.append([1])
            ms.append([1])

        data['x'] = np.concatenate(xs)
        data['dx'] = np.concatenate(dxs)
        data['energy'] = np.concatenate(energies)
        data['ks'] = np.concatenate(ks)
        data['mass'] = np.concatenate(ms)

        f = open(name + ".pkl", "wb")
        pickle.dump(data, f)
        f.close()

        return data

    return get_dataset(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate)


def spring_particle(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate, noise_std, seed):
    """n-body system with spring forces between particles """
    num_particles = NUM_PARTS
    collater = {}

    def diffeq_hyper(t, q, k, m, nparts):
        num_particles = nparts
        vels = q[2 * num_particles:]
        xs = q[:2 * num_particles]
        xs = xs.reshape(-1, 2)
        forces = np.zeros(xs.shape)
        new_k = np.repeat(k, num_particles) * np.tile(k, num_particles)
        new_k = np.repeat(new_k, 2).reshape(-1, 2)
        dx = np.repeat(xs, num_particles, axis=0) - np.tile(xs, (num_particles, 1))
        resu = -new_k * dx
        forces = np.add.reduceat(resu, np.arange(0, nparts * nparts, nparts)).ravel()

        return np.concatenate([vels / np.repeat(m, 2), forces]).ravel()

    def hamiltonian(vec, m, k, num_particles):
        num_particles = num_particles
        x = vec[:num_particles * 2]
        p = vec[2 * num_particles:]
        xs = x.reshape(-1, 2)
        ps = p.reshape(-1, 2)
        U1 = 0
        K = 0
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                U1 += .5 * k[i] * k[j] * ((xs[i] - xs[j]) ** 2).sum()
            K += 0.5 * ((ps[i] ** 2).sum()) / m[i]
        return K, U1

    theta = []
    dtheta = []
    energy = []
    mass_arr = []
    ks_arr = []
    lagrangian = []
    np.random.seed(seed)

    for traj in range(num_trajectories):
        ks = np.ones(NUM_PARTS)#np.random.uniform(.5, 1, size=(NUM_PARTS))
        positions = np.random.uniform(-1, 1, size=(NUM_PARTS, 2))
        velocities = np.random.uniform(-3, 3, size=(NUM_PARTS, 2))
        masses = np.ones(NUM_PARTS)#np.random.uniform(0.1, 1, size=NUM_PARTS)
        momentum = np.multiply(velocities, np.repeat(masses, 2).reshape(-1, 2))
        q = np.concatenate([positions, momentum]).ravel()
        qnrk = rk(lambda t, y: diffeq_hyper(t, y, ks, masses, num_particles), (0, T_max), q,
                  t_eval=np.arange(0, T_max, dt),
                  rtol=1e-12, method='DOP853')
        accum = qnrk.y.T
        ssr = int(sub_sample_rate / dt)
        accum = accum[::ssr]
        daccum = np.array([diffeq_hyper(0, accum[i], ks, masses, num_particles) for i in range(accum.shape[0])])
        energies = []
        lags = []
        for i in range(accum.shape[0]):
            ktmp, utmp = hamiltonian(accum[i], masses, ks, NUM_PARTS)
            energies.append(ktmp + utmp)
            lags.append(ktmp - utmp)

        accum += np.random.randn(*accum.shape) * noise_std
        daccum += np.random.randn(*daccum.shape) * noise_std

        theta.append(accum)
        dtheta.append(daccum)
        energy.append(energies)
        mass_arr.append(masses)
        ks_arr.append(ks)
        lagrangian.append(lags)

    collater['x'] = np.concatenate(theta)
    collater['dx'] = np.concatenate(dtheta)
    collater['energy'] = np.concatenate(energy)
    collater['lagrangian'] = np.concatenate(lagrangian)

    collater['mass'] = mass_arr
    collater['ks'] = ks_arr

    f = open(name + ".pkl", "wb")
    pickle.dump(collater, f)
    f.close()

    return collater


def mass_spring(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate, noise_std, seed):
    """1-body mass spring system"""

    def hamiltonian_fn(coords):
        q, p = np.split(coords, 2)

        H = (p ** 2) / 2 + (q ** 2) / 2  # spring hamiltonian (linear oscillator)
        return H

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dqdt, dpdt = np.split(dcoords, 2)
        S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def get_trajectory(t_span=[0, 3], timescale=0.01, ssr=sub_sample_rate, radius=None, y0=None, noise_std=0.1,
                       **kwargs):

        # get initial state
        if y0 is None:
            y0 = np.random.rand(2) * 2 - 1
        if radius is None:
            radius = np.sqrt(np.random.uniform(0.5, 4.5))
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * (radius)

        spring_ivp = rk(lambda t, y: dynamics_fn(t, y), t_span, y0,
                        t_eval=np.arange(0, t_span[1], timescale),method='DOP853',rtol=1e-12,
                        )

        accum = spring_ivp.y.T
        ssr = int(ssr / timescale)
        accum = accum[::ssr]

        daccum = [dynamics_fn(None, accum[i]) for i in range(accum.shape[0])]
        energies = []
        for i in range(accum.shape[0]):
            energies.append(np.sum(hamiltonian_fn(accum[i])))

        return accum, np.array(daccum), energies

    def get_dataset(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate, seed=seed, test_split=0.5, **kwargs):
        data = {'meta': locals()}

        # randomly sample inputs
        np.random.seed(seed)
        data = {}
        ssr = int(sub_sample_rate / dt)

        xs, dxs, energies, ks, ms = [], [], [], [], []
        for s in range(num_trajectories):
            x, dx, energy = get_trajectory(t_span=[0, T_max], timescale=dt, ssr=sub_sample_rate)

            x += np.random.randn(*x.shape) * noise_std
            dx += np.random.randn(*dx.shape) * noise_std

            xs.append(x)
            dxs.append(dx)
            energies.append(energy)
            ks.append([1])
            ms.append([1])

        data['x'] = np.concatenate(xs)
        data['dx'] = np.concatenate(dxs)
        data['energy'] = np.concatenate(energies)
        data['ks'] = np.concatenate(ks)
        data['mass'] = np.concatenate(ms)

        f = open(name + ".pkl", "wb")
        pickle.dump(data, f)
        f.close()

        return data

    return get_dataset(name, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate)

def three_body(expt_name, num_samples, num_particles, T_max, dt, srate, noise_std, seed):
    """2-body gravitational problem"""

    ##### ENERGY #####
    def potential_energy(state):
        '''U=\sum_i,j>i G m_i m_j / r_ij'''
        tot_energy = np.zeros((1, 1, state.shape[2]))
        for i in range(state.shape[0]):
            for j in range(i + 1, state.shape[0]):
                r_ij = ((state[i:i + 1, 1:3] - state[j:j + 1, 1:3]) ** 2).sum(1, keepdims=True) ** .5
                m_i = state[i:i + 1, 0:1]
                m_j = state[j:j + 1, 0:1]
                tot_energy += m_i * m_j / r_ij
        U = -tot_energy.sum(0).squeeze()
        return U

    def kinetic_energy(state):
        '''T=\sum_i .5*m*v^2'''
        energies = .5 * state[:, 0:1] * (state[:, 3:5] ** 2).sum(1, keepdims=True)
        T = energies.sum(0).squeeze()
        return T

    def total_energy(state):
        return potential_energy(state) + kinetic_energy(state)

    ##### DYNAMICS #####
    def get_accelerations(state, epsilon=0):
        # shape of state is [bodies x properties]
        net_accs = []  # [nbodies x 2]
        for i in range(state.shape[0]):  # number of bodies
            other_bodies = np.concatenate([state[:i, :], state[i + 1:, :]], axis=0)
            displacements = other_bodies[:, 1:3] - state[i, 1:3]  # indexes 1:3 -> pxs, pys
            distances = (displacements ** 2).sum(1, keepdims=True) ** 0.5
            masses = other_bodies[:, 0:1]  # index 0 -> mass
            pointwise_accs = masses * displacements / (distances ** 3 + epsilon)  # G=1
            net_acc = pointwise_accs.sum(0, keepdims=True)
            net_accs.append(net_acc)
        net_accs = np.concatenate(net_accs, axis=0)
        return net_accs

    def update(t, state):
        state = state.reshape(-1, 5)  # [bodies, properties]
        # print(state.shape)
        deriv = np.zeros_like(state)
        deriv[:, 1:3] = state[:, 3:5]  # dx, dy = vx, vy
        deriv[:, 3:5] = get_accelerations(state)
        return deriv.reshape(-1)

    ##### INTEGRATION SETTINGS #####
    def get_orbit(state, update_fn=update, t_points=100, t_span=[0, 2], **kwargs):
        if not 'rtol' in kwargs.keys():
            kwargs['rtol'] = 1e-12
            # kwargs['atol'] = 1e-12
            # kwargs['atol'] = 1e-9

        orbit_settings = locals()

        nbodies = state.shape[0]
        t_eval = np.arange(t_span[0], t_span[1], dt)
        if len(t_eval) != t_points:
            t_eval = t_eval[:-1]
        orbit_settings['t_eval'] = t_eval

        path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(),
                         t_eval=t_eval,method='DOP853', **kwargs)
        orbit = path['y'].reshape(nbodies, 5, t_points)
        return orbit, orbit_settings
        # spring_ivp = rk(update_fn, t_eval, state.reshape(-1), dt)
        # spring_ivp = np.array(spring_ivp)
        # print(spring_ivp.shape)
        # q, p = spring_ivp[:, 0], spring_ivp[:, 1]
        # dydt = [dynamics_fn(y, None) for y in spring_ivp]
        # dydt = np.stack(dydt).T
        # dqdt, dpdt = np.split(dydt, 2)
        # return spring_ivp.reshape(nbodies,5,t_points), 33

        ##### INITIALIZE THE TWO BODIES #####
    def rotate2d(p, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return (R @ p.reshape(2, 1)).squeeze()

    def random_config(nu=2e-1, min_radius=0.9, max_radius=1.2):
        '''This is not principled at all yet'''
        state = np.zeros((3, 5))
        state[:, 0] = 1
        p1 = 2 * np.random.rand(2) - 1
        r = np.random.rand() * (max_radius - min_radius) + min_radius

        p1 *= r / np.sqrt(np.sum((p1 ** 2)))
        p2 = rotate2d(p1, theta=2 * np.pi / 3)
        p3 = rotate2d(p2, theta=2 * np.pi / 3)

        # # velocity that yields a circular orbit
        v1 = rotate2d(p1, theta=np.pi / 2)
        v1 = v1 / r ** 1.5
        v1 = v1 * np.sqrt(
            np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6) ** 2))  # scale factor to get circular trajectories
        v2 = rotate2d(v1, theta=2 * np.pi / 3)
        v3 = rotate2d(v2, theta=2 * np.pi / 3)

        # make the circular orbits slightly chaotic
        v1 *= 1 + nu * (2 * np.random.rand(2) - 1)
        v2 *= 1 + nu * (2 * np.random.rand(2) - 1)
        v3 *= 1 + nu * (2 * np.random.rand(2) - 1)

        state[0, 1:3], state[0, 3:5] = p1, v1
        state[1, 1:3], state[1, 3:5] = p2, v2
        state[2, 1:3], state[2, 3:5] = p3, v3
        return state

    def sample_orbits(timesteps=20, trials=5000, nbodies=3, orbit_noise=2e-1,
                      min_radius=0.9, max_radius=1.2, t_span=[0, 5], verbose=False, **kwargs):
        orbit_settings = locals()
        if verbose:
            print("Making a dataset of near-circular 2-body orbits:")

        x, dx, e, ks, ms = [], [], [], [], []
        # samps_per_trial = np.ceil((T_max / srate))

        # N = samps_per_trial * trials
        np.random.seed(seed)
        for _ in range(trials):
            state = random_config(orbit_noise, min_radius, max_radius)
            orbit, _ = get_orbit(state, t_points=timesteps, t_span=t_span, **kwargs)
            print(orbit.shape)
            batch = orbit.transpose(2, 0, 1).reshape(-1, 15)
            ssr = int(srate / dt)
            # (batch.shape)
            batch = batch[::ssr]
            # print('ssr')
            # print(batch.shape)
            sbx, sbdx, sbe = [], [], []
            for state in batch:
                dstate = update(None, state)
                # reshape from [nbodies, state] where state=[m, qx, qy, px, py]
                # to [canonical_coords] = [qx1, qx2, qy1, qy2, px1,px2,....]
                coords = state.reshape(nbodies, 5).T[1:].flatten()
                dcoords = dstate.reshape(nbodies, 5).T[1:].flatten()
                # print(coords.shape)
                coords += np.random.randn(*coords.shape) * noise_std
                dcoords += np.random.randn(*dcoords.shape) * noise_std

                x.append(coords)
                dx.append(dcoords)

                shaped_state = state.copy().reshape(3, 5, 1)
                e.append(total_energy(shaped_state))

            ks.append(np.ones(num_particles))
            ms.append(np.ones(num_particles))
        # print(len(x))

        #[qx1, qx2,qx3, qy1, qy2,qy3, px1, px2, ....]
        #[0,3,1,4,2,5,6,9,7,10,8,11]

        data = {'x': np.stack(x)[:, [0,3,1,4,2,5,6,9,7,10,8,11]],
                'dx': np.stack(dx)[:, [0,3,1,4,2,5,6,9,7,10,8,11]],
                'energy': np.stack(e),
                'ks': np.stack(ks),
                'mass': np.stack(ms)}
        return data

    return sample_orbits(timesteps=int(np.ceil(T_max / dt)), trials=num_samples, nbodies=3,
                        t_span=[0, T_max], verbose=False)



def grav_n(expt_name, num_samples, num_particles, T_max, dt, srate, noise_std, seed):
    """2-body gravitational problem"""

    ##### ENERGY #####
    def potential_energy(state):
        '''U=sum_i,j>i G m_i m_j / r_ij'''
        tot_energy = np.zeros((1, 1, state.shape[2]))
        for i in range(state.shape[0]):
            for j in range(i + 1, state.shape[0]):
                r_ij = ((state[i:i + 1, 1:3] - state[j:j + 1, 1:3]) ** 2).sum(1, keepdims=True) ** .5
                m_i = state[i:i + 1, 0:1]
                m_j = state[j:j + 1, 0:1]
        tot_energy += m_i * m_j / r_ij
        U = -tot_energy.sum(0).squeeze()
        return U

    def kinetic_energy(state):
        '''T=sum_i .5*m*v^2'''
        energies = .5 * state[:, 0:1] * (state[:, 3:5] ** 2).sum(1, keepdims=True)
        T = energies.sum(0).squeeze()
        return T

    def total_energy(state):
        return potential_energy(state) + kinetic_energy(state)

    ##### DYNAMICS #####
    def get_accelerations(state, epsilon=0):
        # shape of state is [bodies x properties]
        net_accs = []  # [nbodies x 2]
        for i in range(state.shape[0]):  # number of bodies
            other_bodies = np.concatenate([state[:i, :], state[i + 1:, :]], axis=0)
            displacements = other_bodies[:, 1:3] - state[i, 1:3]  # indexes 1:3 -> pxs, pys
            distances = (displacements ** 2).sum(1, keepdims=True) ** 0.5
            masses = other_bodies[:, 0:1]  # index 0 -> mass
            pointwise_accs = masses * displacements / (distances ** 3 + epsilon)  # G=1
            net_acc = pointwise_accs.sum(0, keepdims=True)
            net_accs.append(net_acc)
        net_accs = np.concatenate(net_accs, axis=0)
        return net_accs

    def update(t, state):
        state = state.reshape(-1, 5)  # [bodies, properties]
        # print(state.shape)
        deriv = np.zeros_like(state)
        deriv[:, 1:3] = state[:, 3:5]  # dx, dy = vx, vy
        deriv[:, 3:5] = get_accelerations(state)
        return deriv.reshape(-1)

    ##### INTEGRATION SETTINGS #####
    def get_orbit(state, update_fn=update, t_points=100, t_span=[0, 2], **kwargs):
        if not 'rtol' in kwargs.keys():
            kwargs['rtol'] = 1e-12
            # kwargs['atol'] = 1e-12
            # kwargs['atol'] = 1e-9

        orbit_settings = locals()

        nbodies = state.shape[0]
        t_eval = np.arange(t_span[0], t_span[1], dt)
        if len(t_eval) != t_points:
            t_eval = t_eval[:-1]
        orbit_settings['t_eval'] = t_eval

        path = solve_ivp(fun=update_fn, t_span=t_span, y0=state.flatten(),
                         t_eval=t_eval,method='DOP853', **kwargs)
        orbit = path['y'].reshape(nbodies, 5, t_points)
        return orbit, orbit_settings
        # spring_ivp = rk(update_fn, t_eval, state.reshape(-1), dt)
        # spring_ivp = np.array(spring_ivp)
        # print(spring_ivp.shape)
        # q, p = spring_ivp[:, 0], spring_ivp[:, 1]
        # dydt = [dynamics_fn(y, None) for y in spring_ivp]
        # dydt = np.stack(dydt).T
        # dqdt, dpdt = np.split(dydt, 2)
        # return spring_ivp.reshape(nbodies,5,t_points), 33

    ##### INITIALIZE THE TWO BODIES #####
    def random_config(orbit_noise=5e-2, min_radius=0.5, max_radius=1.5):
        state = np.zeros((2, 5))
        state[:, 0] = 1
        pos = np.random.rand(2) * (max_radius - min_radius) + min_radius
        r = np.sqrt(np.sum((pos ** 2)))

        # velocity that yields a circular orbit
        vel = np.flipud(pos) / (2 * r ** 1.5)
        vel[0] *= -1
        vel *= 1 + orbit_noise * np.random.randn()

        # make the circular orbits SLIGHTLY elliptical
        state[:, 1:3] = pos
        state[:, 3:5] = vel
        state[1, 1:] *= -1
        return state

    ##### HELPER FUNCTION #####
    def coords2state(coords, nbodies=2, mass=1):
        timesteps = coords.shape[0]
        state = coords.T
        state = state.reshape(-1, nbodies, timesteps).transpose(1, 0, 2)
        mass_vec = mass * np.ones((nbodies, 1, timesteps))
        state = np.concatenate([mass_vec, state], axis=1)
        return state

    ##### INTEGRATE AN ORBIT OR TWO #####
    def sample_orbits(timesteps=50, trials=1000, nbodies=2, orbit_noise=5e-2,
                      min_radius=0.5, max_radius=1.5, t_span=[0, 20], verbose=False, **kwargs):
        orbit_settings = locals()
        if verbose:
            print("Making a dataset of near-circular 2-body orbits:")

        x, dx, e, ks, ms = [], [], [], [], []
        # samps_per_trial = np.ceil((T_max / srate))

        # N = samps_per_trial * trials
        np.random.seed(seed)
        for _ in range(trials):
            state = random_config(orbit_noise, min_radius, max_radius)
            orbit, _ = get_orbit(state, t_points=timesteps, t_span=t_span, **kwargs)
            print(orbit.shape)
            batch = orbit.transpose(2, 0, 1).reshape(-1, 10)
            ssr = int(srate / dt)
            # (batch.shape)
            batch = batch[::ssr]
            # print('ssr')
            # print(batch.shape)
            sbx, sbdx, sbe = [], [], []
            for state in batch:
                dstate = update(None, state)
                # reshape from [nbodies, state] where state=[m, qx, qy, px, py]
                # to [canonical_coords] = [qx1, qx2, qy1, qy2, px1,px2,....]
                coords = state.reshape(nbodies, 5).T[1:].flatten()
                dcoords = dstate.reshape(nbodies, 5).T[1:].flatten()
                # print(coords.shape)
                coords += np.random.randn(*coords.shape) * noise_std
                dcoords += np.random.randn(*dcoords.shape) * noise_std

                x.append(coords)
                dx.append(dcoords)

                shaped_state = state.copy().reshape(2, 5, 1)
                e.append(total_energy(shaped_state))

            ks.append(np.ones(num_particles))
            ms.append(np.ones(num_particles))
        # print(len(x))
        data = {'x': np.stack(x)[:, [0, 2, 1, 3, 4, 6, 5, 7]],
                'dx': np.stack(dx)[:, [0, 2, 1, 3, 4, 6, 5, 7]],
                'energy': np.stack(e),
                'ks': np.stack(ks),
                'mass': np.stack(ms)}
        return data

    return sample_orbits(timesteps=int(np.ceil(T_max / dt)), trials=num_samples, nbodies=2, orbit_noise=5e-2,
                         min_radius=0.5, max_radius=1.5, t_span=[0, T_max], verbose=False)


def pendulum(expt_name, num_samples, num_particles, T_max, dt, srate, noise_std, seed,integ_type='rk8'):

    """simple pendulum"""

    def hamiltonian_fn(coords):
        q, p = np.split(coords, 2)
        H = 9.81 * (1 - cos(q)) + 0.5*(p ** 2)   # pendulum hamiltonian
        return H

    def dynamics_fn2(t,coords):
        # print(coords)
        # dcoords = autograd.grad(hamiltonian_fn)(coords)
        # dqdt, dpdt = np.split(dcoords, 2)
        # S = np.concatenate([dpdt, -dqdt], axis=-1)

        S = [coords[1],-9.81*sin(coords[0])]
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt, radius=None, y0=None, **kwargs):
        t_eval = np.arange(t_span[0], t_span[1], timescale)  # int(timescale * (t_span[1] - t_span[0])))

        # get initial state
        if y0 is None:
            y0 = np.random.rand(2) * 2. - 1
        if radius is None:
            radius = np.random.rand() + 1.3  # sample a range of radii
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius  ## set the appropriate radius

        spring_ivp = rk(lambda t, y: dynamics_fn2(t, y), t_span, y0,
                        t_eval=np.arange(0, t_span[1], timescale),method='DOP853',rtol=1e-12,
                        )

        q, p = spring_ivp['y'][0], spring_ivp['y'][1]
        dydt = [dynamics_fn2(None,y) for y in spring_ivp['y'].T]
        dydt = np.stack(dydt).T
        dqdt, dpdt = np.split(dydt, 2)

        # add noise
        q += np.random.randn(*q.shape) * noise_std
        p += np.random.randn(*p.shape) * noise_std
        return q, p, dqdt, dpdt, t_eval

    def get_dataset(seed=0, samples=50, **kwargs):
        data = {'meta': locals()}

        # randomly sample inputs
        np.random.seed(seed)
        xs, dxs = [], []
        ssr = int(srate / dt)
        ms = []
        ks = []
        for s in range(samples):
            x, y, dx, dy, t = get_trajectory(**kwargs)
            # print(x.shape)
            x = x[::ssr]
            y = y[::ssr]
            # print(x.shape)
            ms.append([1.])
            ks.append([1.])
            xs.append(np.stack([x, y]).T)
            dxs.append(np.stack([dx, dy]).T)

        data['x'] = np.concatenate(xs)
        data['dx'] = np.concatenate(dxs).squeeze()
        data['mass'] = np.concatenate(ms)
        data['ks'] = np.concatenate(ks)
        return data

    return get_dataset(seed=seed, samples=num_samples)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #data_name, expt_name, num_samples, num_particles, T_max, dt, srate, noise_std = 0, seed = 0, pixels = False):

    d = get_dataset('heinon', 'temp', 50, 1, 3, 0.1, 0.1,seed=1,noise_std=0)
    # d1 = get_dataset('pendulum', 'temp', 20, 1,2, 0.01, 0.01,seed=0)
    plt.scatter(d['x'][:, 0], d['x'][:, 1],c='blue')
    # plt.hist(d['x'][:,1])
    print(d['x'].shape)
    print(np.mean(np.abs(d['x']),0))
    # print(np.min(np.abs(d['x'][:,0])))
    # plt.hist(d['x'][:,1])
    # print(d['x'].shape)
    # plt.scatter(d['x'][:, 0], d['x'][:, 1],c='red')
    # plt.scatter(d['x'][:, 2], d['x'][:, 3], c='blue')
    # plt.scatter(d['x'][:, 4], d['x'][:, 5], c='green')
    # print(d['energy'])
    # ham = get_hamiltonian('pendulum')
    # epred = ham(d['x'],'classic')
    # print(np.sum(epred,0))
    # spdim = int(d['x'][0].shape[0] / 1)
    # xnow = arrange_data(d, 1, 1, 4, 0.01,0.01,spatial_dim=spdim, nograph=False, samp_size=2)
    # print(xnow.shape)
    # print(xnow[0,[1,2,3],:].shape)
    # true_batch = xnow[1:, :, :].reshape(-1, int(spdim))
    # true_batch = xnow[1:, :, :]
    # hp_gt = ham(true_batch.squeeze(), 'graphic')
    # print(np.sum(hp_gt,0))
    # plt.scatter(d1['x'][:, 0], d1['x'][:, 1],alpha=0.1)

    plt.show()

