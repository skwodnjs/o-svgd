import numpy as np
import torch
import time

from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

from draw import plot_2d_contour
from o_svgd import O_SVGD, O_Langevin


def log_p(x):
    z0 = x[:,0] + x[:,1]**3
    z1 = x[:,1]
    return -(z0*2/2 + z1*2/2)

def g(x):
    return x[:,0] + x[:,1]**3

if __name__ == "__main__":
    # seed = 1000
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    dim = 2
    NUM_PARTICLES = 50

    ## init
    x0 = torch.zeros(NUM_PARTICLES, dim)
    x0[:,1] = torch.randn(NUM_PARTICLES) * 0.5
    x0[:,0] = torch.randn(NUM_PARTICLES) * 0.5

    ## animation
    bound_x = 40
    bound_y = 6

    fig, ax = plt.subplots(figsize=(8, 6))
    artists = []

    p1 = plot_2d_contour(ax, log_p, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
    p2 = plot_2d_contour(ax, g, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100, zero = True)
    p3, = ax.plot(x0[:,0], x0[:,1], '.', alpha=0.8, markersize=5)

    artists.append([p1, p2, p3])

    ## method
    sampler = O_SVGD(log_p, g, stepsize=1e-2,alpha = 100, M = 1000)
    # sampler = O_Langevin(log_p, g, stepsize=1e-2,alpha = 100, M = 1000)

    x = x0
    max_iter = 50
    total_time = 0

    for i in range(max_iter):
        start = time.time()
        x = sampler.step(x)
        step_time = time.time() - start

        p1 = plot_2d_contour(ax, log_p, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
        p2 = plot_2d_contour(ax, g, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100, zero = True)
        p3, = ax.plot(x.detach()[:,0], x.detach()[:,1], '.', alpha=0.8, markersize=5, color='green')

        artists.append([p1, p2, p3])

        print("iter:", i+1)
        print("step time:", step_time)
        total_time += step_time

    # result
    print('runtime:', total_time)
    ani = ArtistAnimation(fig, artists, interval=100, repeat=False)
    ani.save('animation.gif', writer='pillow')
    plt.show()