import torch
import time
import os

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

from draw import plot_2d_contour
from o_langevin import O_Langevin
from o_svgd import O_SVGD


def log_p(x):
    z0 = x[:,0] + x[:,1]**3
    z1 = x[:,1]
    return -(z0**2/2 + z1**2/2)

def g(x):
    return x[:,0] + x[:,1]**3

if __name__ == "__main__":
    # seed = 1000
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    dim = 2
    NUM_PARTICLES = 50

    x0 = torch.zeros(NUM_PARTICLES, dim)
    x0[:,1] = torch.randn(NUM_PARTICLES)
    x0[:,0] = torch.randn(NUM_PARTICLES)

    dir = "frames"

    bound_x = 40
    bound_y = 6

    fig, ax = plt.subplots(figsize=(8, 6))
    artists = []

    p1 = plot_2d_contour(ax, log_p, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
    p2 = plot_2d_contour(ax, g, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100, zero = True)
    p3, = ax.plot(x0[:,0], x0[:,1], '.', alpha=0.8, markersize=5, color='C2')
    artists.append([p1, p2, p3])

    # plt.savefig(f"{dir}/frame0000.png", dpi=300)
    # ax.clear()

    # sampler = O_SVGD(log_p, g, stepsize=1e-2,alpha = 100, M = 1000)
    sampler = O_Langevin(log_p, g, stepsize=1e-2,alpha = 100, M = 1000)

    x = x0
    max_iter = 150
    total_time = 0

    for i in range(max_iter):
        start = time.time()
        x = sampler.step(x)
        step_time = time.time() - start

        p1 = plot_2d_contour(ax, log_p, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
        p2 = plot_2d_contour(ax, g, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100, zero = True)
        p3, = ax.plot(x.detach()[:,0], x.detach()[:,1], '.', alpha=0.8, markersize=5, color='C2')
        artists.append([p1, p2, p3])

        # plt.savefig(f"{dir}/frame{i+1:04d}.png", dpi=300)
        # ax.clear()

        plot_time = time.time() - step_time - start

        print(f"iter: {i+1:04d} \t step time: {step_time:.7f}  \t plot time: {plot_time:.7f}")
        print()
        total_time += step_time

    print(f'runtime: {total_time:.7f}')
    ani = ArtistAnimation(fig, artists, interval= 50, repeat=True)
    plt.show()

    # start = time.time()
    # image_files = sorted([os.path.join(dir, file) for file in os.listdir(dir) if file.endswith(".png")])
    # frames = [Image.open(image) for image in image_files]
    # frames[0].save(
    #     "animation.gif",
    #     save_all=True,
    #     append_images=frames,
    #     duration=50,  # ms
    # )
    # save_time = time.time() - start
    # print(f'save time: {save_time:.7f}')