import os
import numpy as np
from matplotlib import pyplot as plt
import imageio


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    import PIL.Image as Image
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


# ifp = '/home/mmajurski/github/ssl-gmm/usb/saved_models/classic_cv/debug'
ifp = '/home/mmajurski/github/ssl-gmm/usb/saved_models/classic_cv/'
fns = ['denom_lb_hist','denom_ulb_s_hist','denom_ulb_w_hist']

step = 64 #64
st_idx = 0
fps = 4
duration = 1000 * (1.0 / fps)  # ms
step_limit = None #64  # None

for fn in fns:
    with imageio.get_writer(fn + '.gif', mode='I', duration=duration, loop=0) as writer:
        # load numpy array from csv file
        x_dat = np.loadtxt(os.path.join(ifp, fn + '_x.csv'), delimiter=',')
        c_dat = np.loadtxt(os.path.join(ifp, fn + '_c.csv'), delimiter=',')
        if step_limit is None:
            step_limit = x_dat.shape[0]
        step_limit = min(step_limit, x_dat.shape[0])
        sample_idx = list(range(st_idx, step_limit, step))
        for idx in sample_idx:
            print("{}/{}".format(idx, step_limit))

            x = x_dat[idx, :-1]
            y = c_dat[idx, :]

            plt.clf()
            plt.bar(x, y, width=x[1]-x[0])
            plt.title("Train Iteration {}".format(idx))
            # plt.plot(x, y)
            # plt.show()
            image = fig2img(plt.gcf())
            writer.append_data(image)





with imageio.get_writer('denom.gif', mode='I', duration=duration, loop=0) as writer:
    # load numpy array from csv file
    denom_lb_hist_x = np.loadtxt(os.path.join(ifp, 'denom_lb_hist_x.csv'), delimiter=',')
    denom_lb_hist_c = np.loadtxt(os.path.join(ifp, 'denom_lb_hist_c.csv'), delimiter=',')
    denom_ulb_s_hist_x = np.loadtxt(os.path.join(ifp, 'denom_ulb_s_hist_x.csv'), delimiter=',')
    denom_ulb_s_hist_c = np.loadtxt(os.path.join(ifp, 'denom_ulb_s_hist_c.csv'), delimiter=',')
    denom_ulb_w_hist_x = np.loadtxt(os.path.join(ifp, 'denom_ulb_w_hist_x.csv'), delimiter=',')
    denom_ulb_w_hist_c = np.loadtxt(os.path.join(ifp, 'denom_ulb_w_hist_c.csv'), delimiter=',')

    if step_limit is None:
        step_limit = denom_lb_hist_x.shape[0]
    step_limit = min(step_limit, denom_lb_hist_x.shape[0])

    sample_idx = list(range(st_idx, step_limit, step))

    for idx in sample_idx:
        print("{}/{}".format(idx, step_limit))

        plt.clf()
        plt.title("Train Iteration {}".format(idx))

        x = denom_lb_hist_x[idx, :-1]
        y = denom_lb_hist_c[idx, :]
        plt.bar(x, y, width=x[1]-x[0])

        x = denom_ulb_s_hist_x[idx, :-1]
        y = denom_ulb_s_hist_c[idx, :]
        plt.bar(x, y, width=x[1] - x[0])

        x = denom_ulb_w_hist_x[idx, :-1]
        y = denom_ulb_w_hist_c[idx, :]
        plt.bar(x, y, width=x[1] - x[0])

        plt.legend(['lb','ulb_w','ulb_s'])

        image = fig2img(plt.gcf())
        writer.append_data(image)