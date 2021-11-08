import pickle
from scipy.integrate import solve_ivp
import autograd
# import autograd.numpy as np
from autograd.numpy import cos, sin
import numpy as np


def rk4ng(dx_dt_fn, x_t, dt):
    k1 = dt * dx_dt_fn(x_t)
    # print(x_t + (1 / 2) * k1)
    k2 = dt * dx_dt_fn(x_t + (1. / 2) * k1)
    k3 = dt * dx_dt_fn(x_t + (1. / 2) * k2)
    k4 = dt * dx_dt_fn(x_t + k3)
    x_tp1 = x_t + (1. / 6) * (k1 + k2 * 2 + k3 * 2 + k4)
    return x_tp1


def rk3ng(dx_dt_fn, x_t, dt):
    k1 = dt * dx_dt_fn(x_t)
    k2 = dt * dx_dt_fn(x_t + (1 / 2) * k1)
    k3 = dt * dx_dt_fn(x_t + k2)
    x_tp1 = x_t + (1 / 6) * (k1 + k2 * 4 + k3)
    return x_tp1


def rk1ng(dx_dt_fn, x_t, dt):
    k1 = dt * dx_dt_fn(x_t)
    x_tp1 = x_t + k1
    return x_tp1


def rk2ng(dx_dt_fn, x_t, dt):
    k1 = dt * dx_dt_fn(x_t)
    k2 = dt * dx_dt_fn(x_t + (1 / 2) * k1)
    x_tp1 = x_t + k2
    return x_tp1


def vi1ng(dx_dt_fn, x_t, dt):
    subdim = int(len(x_t) // 2)
    q = x_t[:subdim]
    p = x_t[subdim:]
    q1 = q + dt * dx_dt_fn(np.concatenate([q, p]))[:subdim]
    p1 = p + dt * dx_dt_fn(np.concatenate([q1, p]))[subdim:]

    return np.concatenate([q1, p1])


def vi2ng(dx_dt_fn, x_t, dt):
    subdim = int(len(x_t) // 2)
    q = x_t[:subdim]
    p = x_t[subdim:]
    c1 = 0
    c2 = 1
    d1 = d2 = 0.5

    q1 = q + dt * c1 * dx_dt_fn(np.concatenate([q, p]))[:subdim]
    p1 = p + dt * d1 * dx_dt_fn(np.concatenate([q1, p]))[subdim:]
    q2 = q1 + dt * c2 * dx_dt_fn(np.concatenate([q1, p1]))[:subdim]
    p2 = p1 + dt * d2 * dx_dt_fn(np.concatenate([q2, p1]))[subdim:]

    return np.concatenate([q2, p2])


def vi3ng(dx_dt_fn, x_t, dt):
    subdim = int(len(x_t) // 2)
    q = x_t[:subdim]
    p = x_t[subdim:]

    q1 = q + dt * 1 * dx_dt_fn(np.concatenate([q, p]))[:subdim]
    p1 = p + dt * (-1. / 24) * dx_dt_fn(np.concatenate([q1, p]))[subdim:]
    q2 = q1 + dt * (-2. / 3) * dx_dt_fn(np.concatenate([q1, p1]))[:subdim]
    p2 = p1 + dt * (3. / 4) * dx_dt_fn(np.concatenate([q2, p1]))[subdim:]
    q3 = q2 + dt * (2. / 3) * dx_dt_fn(np.concatenate([q2, p2]))[:subdim]
    p3 = p2 + dt * (7. / 24) * dx_dt_fn(np.concatenate([q3, p2]))[subdim:]

    return np.concatenate([q3, p3])


def vi4ng(dx_dt_fn, x_t, dt):
    subdim = int(len(x_t) // 2)
    q = x_t[:subdim]
    p = x_t[subdim:]

    d1 = 0.515352837431122936
    d2 =  -0.085782019412973646
    d3 = 0.441583023616466524
    d4 = 0.128846158365384185
    c1 = 0.134496199277431089
    c2 =  -0.224819803079420806
    c3 =  0.756320000515668291
    c4 =  0.334003603286321425


    # d1 = d4 = 1. / 6 * (2 + 2 ** (1 / 3) + 2 ** (-1 / 3))
    # d2 = d3 = 1. / 6 * (1 - 2 ** (1 / 3) - 2 ** (-1 / 3))
    # c1 = 0
    # c2 = c4 = 1. / (2 - 2 ** (1 / 3))
    # c3 = 1. / (1 - 2 ** (2 / 3))

    # d1 = d3 = 1. / (2 - 2 ** (1. / 3))
    # d2 = -(2 ** (1. / 3)) / (2 - 2 ** (1. / 3))
    # d4 = 0
    #
    # c1 = c4 = 1. / (2 * (2 - 2 ** (1. / 3)))
    # c2 = c3 = (1 - 2 ** (1. / 3)) / (2 * (2 - 2 ** (1. / 3)))

    q1 = q + dt * c1 * dx_dt_fn(np.concatenate([q, p]))[:subdim]
    p1 = p + dt * d1 * dx_dt_fn(np.concatenate([q1, p]))[subdim:]
    q2 = q1 + dt * c2 * dx_dt_fn(np.concatenate([q1, p1]))[:subdim]
    p2 = p1 + dt * d2 * dx_dt_fn(np.concatenate([q2, p1]))[subdim:]
    q3 = q2 + dt * c3 * dx_dt_fn(np.concatenate([q2, p2]))[:subdim]
    p3 = p2 + dt * d3 * dx_dt_fn(np.concatenate([q3, p2]))[subdim:]
    q4 = q3 + dt * c4 * dx_dt_fn(np.concatenate([q3, p3]))[:subdim]
    p4 = p3 + dt * d4 * dx_dt_fn(np.concatenate([q4, p3]))[subdim:]
    return np.concatenate([q4, p4])


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


def get_system(system_name, integrator_type, num_samples, num_parts, T_max, dt, srate, noise_std=0, seed=3):
    SYSTEMS = {'pendulum': pendulum, 'nspring': spring_particle, 'ngrav': ngrav, 'mass_spring': mass_spring,
               'heinon': heinon_heiles,'three_body':three_body}

    if system_name in SYSTEMS:
        return SYSTEMS[system_name](integrator_type, num_samples, num_parts, T_max, dt, srate, noise_std, seed)


def heinon_heiles(integrator_type, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate, noise_std, seed):
    """heinon heiles data generator"""

    def hamiltonian_fn(coords):
        x, y, px, py = np.split(coords, 4)
        lambda_ = 1
        H = 0.5 * px ** 2 + 0.5 * py ** 2 + 0.5 * (x ** 2 + y ** 2) + lambda_ * (
                (x ** 2) * y - (y ** 3) / 3)
        return H
    def rk(dx_dt_fn, t, y0, dt):
        single_step = choose_integrator_nongraph(integrator_type)
        store = []
        store.append(y0)
        for i in range(len(t)):
            #             print(type(y0))
            ynext = single_step(dx_dt_fn, y0, dt)
            store.append(ynext)
            y0 = ynext

        return store[:-1]

    def dynamics_fn(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dxdt, dydt, dpxdt, dpydt = np.split(dcoords, 4)
        S = np.concatenate([dpxdt, dpydt, -dxdt, -dydt], axis=-1)
        return S

    def dynamics_fn2(coords):
        return dynamics_fn(0, coords)

    def get_trajectory(t_span=[0, T_max], timescale=dt, ssr=sub_sample_rate, radius=None, y0=None, noise_std=noise_std):
        np.random.seed(seed)
        # get initial state
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        px = np.random.uniform(-.5, .5)
        py = np.random.uniform(-.5, .5)

        y0 = np.array([x, y, px, py])
        t_eval = np.arange(0,T_max,dt)
        if integrator_type == 'gt':
            _ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, method='DOP853', rtol=1e-12)
            spring_ivp = _ivp['y'].T
        else:
            spring_ivp = rk(dynamics_fn2, t_eval, y0, dt)

        return spring_ivp

    return get_trajectory()


def mass_spring(integrator_type, num_samples, num_parts, T_max, dt, srate, noise_std, seed):
    """simple pendulum"""

    def hamiltonian_fn(coords):
        q, p = np.split(coords, 2)
        H = (p ** 2) / 2 + (q ** 2) / 2
        return H

    def dynamics_fn(coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dqdt, dpdt = np.split(dcoords, 2)
        S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def dynamics_fn2(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dqdt, dpdt = np.split(dcoords, 2)
        S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt, radius=None, y0=None, **kwargs):
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        if y0 is None:
            y0 = np.random.rand(2) * 2. - 1
        if radius is None:
            radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius

        if integrator_type == 'gt':
            _ivp = solve_ivp(fun=dynamics_fn2, t_span=t_span, y0=y0, t_eval=t_eval, method='DOP853', rtol=1e-12)
            spring_ivp = _ivp['y'].T
        else:
            spring_ivp = rk(dynamics_fn, t_eval, y0, srate)
        return spring_ivp

    def rk(dx_dt_fn, t, y0, dt):
        single_step = choose_integrator_nongraph(integrator_type)
        store = []
        store.append(y0)
        for i in range(len(t)):
            #             print(type(y0))
            ynext = single_step(dx_dt_fn, y0, dt)
            store.append(ynext)
            y0 = ynext

        return store[:-1]

    # randomly sample inputs
    np.random.seed(seed)
    return get_trajectory()


def pendulum(integrator_type, num_samples, num_parts, T_max, dt, srate, noise_std, seed):
    """simple pendulum"""

    def hamiltonian_fn(coords):
        q, p = np.split(coords, 2)
        H = 9 * (1 - cos(q)) + p ** 2 / 2
        return H

    def dynamics_fn(coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dqdt, dpdt = np.split(dcoords, 2)
        S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def dynamics_fn2(t, coords):
        dcoords = autograd.grad(hamiltonian_fn)(coords)
        dqdt, dpdt = np.split(dcoords, 2)
        S = np.concatenate([dpdt, -dqdt], axis=-1)
        return S

    def get_trajectory(t_span=[0, T_max], timescale=dt, radius=None, y0=None, **kwargs):
        t_eval = np.arange(t_span[0], t_span[1], timescale)
        if y0 is None:
            y0 = np.random.rand(2) * 2. - 1
        if radius is None:
            radius = np.random.rand() + 1.3
        y0 = y0 / np.sqrt((y0 ** 2).sum()) * radius

        if integrator_type == 'gt':
            _ivp = solve_ivp(fun=dynamics_fn2, t_span=t_span, y0=y0, t_eval=t_eval, method='DOP853', rtol=1e-12)
            spring_ivp = _ivp['y'].T
        else:
            spring_ivp = rk(dynamics_fn, t_eval, y0, srate)
        return spring_ivp

    def rk(dx_dt_fn, t, y0, dt):
        single_step = choose_integrator_nongraph(integrator_type)
        store = []
        store.append(y0)
        for i in range(len(t)):
            #             print(type(y0))
            ynext = single_step(dx_dt_fn, y0, dt)
            store.append(ynext)
            y0 = ynext

        return store[:-1]

    # randomly sample inputs
    np.random.seed(seed)
    return get_trajectory()


def spring_particle(integrator_type, num_trajectories, NUM_PARTS, T_max, dt, sub_sample_rate, noise_std, seed):
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

    def dynamics_fn(q, k=[1, 1, 1, 1, 1], m=[1, 1, 1, 1, 1], nparts=5):
        return diffeq_hyper(0, q, k, m, nparts)

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

    def rk(dx_dt_fn, t, y0, dt):
        single_step = choose_integrator_nongraph(integrator_type)
        store = []
        store.append(y0)
        for i in range(len(t)):
            #             print(type(y0))
            ynext = single_step(dx_dt_fn, y0, dt)
            store.append(ynext)
            y0 = ynext

        return store[:-1]

    theta = []
    dtheta = []
    energy = []
    mass_arr = []
    ks_arr = []
    lagrangian = []
    np.random.seed(seed)

    for traj in range(num_trajectories):
        ks = np.ones(NUM_PARTS)  # np.random.uniform(.5, 1, size=(NUM_PARTS))
        positions = np.random.uniform(-1, 1, size=(NUM_PARTS, 2))
        velocities = np.random.uniform(-3, 3, size=(NUM_PARTS, 2))
        masses = np.ones(NUM_PARTS)  # np.random.uniform(0.1, 1, size=NUM_PARTS)
        momentum = np.multiply(velocities, np.repeat(masses, 2).reshape(-1, 2))
        q = np.concatenate([positions, momentum]).ravel()

        if integrator_type == 'gt':
            _ivp = solve_ivp(lambda t, y: diffeq_hyper(t, y, ks, masses, num_particles), t_span=[0, T_max], y0=q,
                             t_eval=np.arange(0, T_max, dt), method='DOP853', rtol=1e-12)
            spring_ivp = _ivp['y'].T
        else:
            spring_ivp = rk(dynamics_fn, np.arange(0, T_max, dt), q, dt)

    return spring_ivp


def ngrav(integrator_type, num_samples, num_particles, T_max, dt, srate, noise_std, seed):
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

        qs = state[1:5].reshape(-1, 2)
        ps = state[6:].reshape(-1, 2)
        ms = np.array([1, 1]).reshape(2, 1)
        state = np.concatenate([ms, qs, ps], 1)

        deriv = np.zeros_like(state)
        deriv[:, 1:3] = state[:, 3:5]  # dx, dy = vx, vy
        deriv[:, 3:5] = get_accelerations(state)

        qd = deriv[:, 1:3].ravel()
        pd = deriv[:, 3:5].ravel()
        return np.hstack([0, qd, 0, pd])

    def dynamics_fn(state):
        return update(0, state)

    def rk(dx_dt_fn, t, y0, dt):
        single_step = choose_integrator_nongraph(integrator_type)
        store = []
        store.append(y0)
        for i in range(len(t)):
            #             print(type(y0))
            ynext = single_step(dx_dt_fn, y0, dt)
            store.append(ynext)
            y0 = ynext

        return store[:-1]

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
        # print(state)

        qs = state[:, 1:3].ravel()
        ps = state[:, 3:5].ravel()

        if integrator_type == 'gt':
            path = solve_ivp(fun=update_fn, t_span=t_span, y0=np.hstack([1, qs, 1, ps]),
                             t_eval=t_eval, method='DOP853', **kwargs)
            orbit = path['y'].T
        #         elif 'vi' in integrator_type:
        #             orbit = rk(dynamics_fn, np.arange(0, T_max, dt), state.flatten(), dt)
        else:
            orbit = rk(dynamics_fn, np.arange(0, T_max, dt), np.hstack([1, qs, 1, ps]), dt)

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

        return orbit

    return sample_orbits(timesteps=int(np.ceil(T_max / dt)), trials=num_samples, nbodies=2, orbit_noise=5e-2,
                         min_radius=0.5, max_radius=1.5, t_span=[0, T_max], verbose=False)


def three_body(integrator_type, num_samples, num_particles, T_max, dt, srate, noise_std, seed):
    """3-body gravitational problem"""


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


    def update(t, state):
        qs = state[:int( 2 * 3)].reshape(-1, 2)
        ps = state[int( 2 * 3):].reshape(-1, 2)
        ms = np.array([1, 1, 1]).reshape(3, 1)
        state = np.concatenate([qs, ps], 1)
        nstate = np.concatenate([ms,qs, ps], 1)
        deriv = np.zeros_like(state)
        deriv[:, :2] = state[:, 2:4]  # dx, dy = vx, vy
        deriv[:, 2:4] = get_accelerations(nstate)

        qd = deriv[:, :2].ravel()
        pd = deriv[:, 2:4].ravel()
        return np.hstack([qd, pd])


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
        v1 = v1 * np.sqrt(np.sin(np.pi / 3) / (2 * np.cos(np.pi / 6) ** 2))  # scale factor to get circular trajectories
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


    ##### INTEGRATE AN ORBIT OR TWO #####
    def sample_orbits(timesteps=20, trials=5000, nbodies=3, orbit_noise=2e-1,
                      min_radius=0.9, max_radius=1.2, t_span=[0, 5], verbose=False, **kwargs):
        orbit_settings = locals()
        if verbose:
            print("Making a dataset of near-circular 3-body orbits:")
        np.random.seed(seed)
        state = random_config(nu=orbit_noise, min_radius=min_radius, max_radius=max_radius)
        orbit, settings = get_orbit(state, t_points=timesteps, t_span=t_span, nbodies=nbodies, **kwargs)
        #         print(orbit.shape)
        #         batch = orbit.transpose(2,0,1).reshape(-1,nbodies*5)

        #         for state in batch:
        #             dstate = update(None, state)

        #             # reshape from [nbodies, state] where state=[m, qx, qy, px, py]
        #             # to [canonical_coords] = [qx1, qx2, qy1, qy2, px1,px2,....]
        #             coords = state.reshape(nbodies,5).T[1:].flatten()
        #             dcoords = dstate.reshape(nbodies,5).T[1:].flatten()
        #             x.append(coords)
        #             dx.append(dcoords)

        #             shaped_state = state.copy().reshape(nbodies,5,1)
        #             e.append(total_energy(shaped_state))

        #     data = {'coords': np.stack(x)[:N],
        #             'dcoords': np.stack(dx)[:N],
        #             'energy': np.stack(e)[:N] }
        return orbit


    def get_orbit(state, update_fn=update, t_points=100, t_span=[0, 2], integrator_type=integrator_type, **kwargs):
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
        # print(state)

        qs = state[:, 1:3].ravel()
        ps = state[:, 3:5].ravel()

        if integrator_type == 'gt':
            path = solve_ivp(fun=update_fn, t_span=t_span, y0=np.hstack([ qs, ps]),
                             t_eval=t_eval, method='DOP853', **kwargs)
            orbit = path['y'].T
        #         elif 'vi' in integrator_type:
        #             orbit = rk(dynamics_fn, np.arange(0, T_max, dt), state.flatten(), dt)
        else:
            orbit = rk(dynamics_fn, np.arange(0, T_max, dt), np.hstack([ qs, ps]), dt)

        return orbit, orbit_settings


    def dynamics_fn(state):
        return update(0, state)


    def rk(dx_dt_fn, t, y0, dt):
        single_step = choose_integrator_nongraph(integrator_type)
        store = []
        store.append(y0)
        for i in range(len(t)):
            #             print(type(y0))
            ynext = single_step(dx_dt_fn, y0, dt)
            store.append(ynext)
            y0 = ynext

        return store[:-1]



    return sample_orbits(timesteps=int(np.ceil(T_max / dt)), trials=1, nbodies=3, orbit_noise=5e-2,
                             min_radius=0.9, max_radius=1.2, t_span=[0, T_max], verbose=False)

