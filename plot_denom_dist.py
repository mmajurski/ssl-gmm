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


ifp = '/home/mmajurski/github/ssl-gmm/usb/saved_models/classic_cv/debug'
# ifp = '/home/mmajurski/github/ssl-gmm/usb/saved_models/classic_cv/'
fns = ['denom_lb_hist','denom_ulb_s_hist','denom_ulb_w_hist']

step = 2
st_idx = 0
fps = 4
duration = 1000 * (1.0 / fps)  # ms
step_limit = 128

# with imageio.get_writer('pl-acc-vs-denom.gif', mode='I', duration=duration, loop=0) as writer:
#     # load numpy array from csv file
#     d_dat = np.loadtxt(os.path.join(ifp, 'denom_lb_vals.csv'), delimiter=',')
#     a_dat = np.loadtxt(os.path.join(ifp, 'pl_acc.csv'), delimiter=',')
#     if step_limit is None:
#         step_limit = d_dat.shape[0]
#     step_limit = min(step_limit, d_dat.shape[0])
#     sample_idx = list(range(st_idx, step_limit, step))
#     for idx in sample_idx:
#         print("{}/{}".format(idx, step_limit))
#
#         x = d_dat[idx, :]
#         y = a_dat[idx, :]
#
#         plt.clf()
#         plt.plot(x, y, '.')
#         plt.xlim([np.min(d_dat), np.max(d_dat)])
#         plt.title("Train Iteration {}".format(idx))
#         plt.ylabel("PL Accuracy")
#         plt.xlabel("Denominator")
#         # plt.plot(x, y)
#         # plt.show()
#         image = fig2img(plt.gcf())
#         writer.append_data(image)


for fps in [4, 32]:
    if fps == 4:
        step_limit = 128
        step = 2
    else:
        step_limit = 1024
        step = 2
    duration = 1000 * (1.0 / fps)  # ms

    for fn in ['denom_ulb_w', 'denom_ulb_s']:
        with imageio.get_writer('{}-{}fps.gif'.format(fn, fps), mode='I', duration=duration, loop=0) as writer:
            # load numpy array from csv file
            lb_dat = np.loadtxt(os.path.join(ifp, 'denom_lb_vals.csv'), delimiter=',')
            ulb_w_dat = np.loadtxt(os.path.join(ifp, '{}_vals.csv'.format(fn)), delimiter=',')
            if step_limit is None:
                step_limit = lb_dat.shape[0]
            step_limit = min(step_limit, lb_dat.shape[0])
            sample_idx = list(range(st_idx, step_limit, step))
            # minv = min(np.min(lb_dat), np.min(ulb_w_dat))
            # maxv = max(np.max(lb_dat), np.max(ulb_w_dat))
            # bins = np.linspace(minv, maxv, 100)
            minv = None
            maxv = None
            for idx in sample_idx:
                print("{}/{}".format(idx, step_limit))

                min_lb = np.min(lb_dat[idx, :])
                perc10_lb = np.percentile(lb_dat[idx, :], 10)
                perc90_lb = np.percentile(lb_dat[idx, :], 90)
                max_lb = np.max(lb_dat[idx, :])

                x = ulb_w_dat[idx, :]

                nminv = min(np.min(x), np.min(lb_dat[idx, :])) * 0.95
                nmaxv = max(np.max(x), np.max(lb_dat[idx, :])) * 1.25
                if minv is None:
                    minv = nminv
                else:  # ema
                    if fps > 16:
                        minv = 0.95 * minv + 0.05 * nminv
                    else:
                        minv = 0.9 * minv + 0.2 * nminv
                if maxv is None:
                    maxv = nmaxv
                else:  # ema
                    if fps > 16:
                        maxv = 0.95 * maxv + 0.05 * nmaxv
                    else:
                        maxv = 0.8 * maxv + 0.2 * nmaxv

                bins = np.linspace(minv, maxv, 100)
                c, x = np.histogram(x, bins=bins)
                x = x[:-1]
                plt.clf()

                plt.bar(x, c, width=x[1]-x[0])
                plt.axvline(x=min_lb, color='r')  # draws a red vertical line
                plt.axvline(x=perc10_lb, color='m')  # draws a red vertical line
                plt.axvline(x=perc90_lb, color='c')  # draws a red vertical line
                plt.axvline(x=max_lb, color='g')  # draws a red vertical line
                plt.title("Train Iteration {}".format(idx))
                plt.ylabel("Count")
                plt.xlabel("Denominator")
                plt.legend(['lb_min','lb_10th', 'lb_90th', 'lb_max'])
                # plt.plot(x, y)
                # plt.show()
                image = fig2img(plt.gcf())
                writer.append_data(image)


# step = 64 #64
# st_idx = 0
# fps = 4
# duration = 1000 * (1.0 / fps)  # ms
# step_limit = 128  #None #64  # None
#
# for fn in fns:
#     with imageio.get_writer(fn + '.gif', mode='I', duration=duration, loop=0) as writer:
#         # load numpy array from csv file
#         x_dat = np.loadtxt(os.path.join(ifp, fn + '_x.csv'), delimiter=',')
#         c_dat = np.loadtxt(os.path.join(ifp, fn + '_c.csv'), delimiter=',')
#         if step_limit is None:
#             step_limit = x_dat.shape[0]
#         step_limit = min(step_limit, x_dat.shape[0])
#         sample_idx = list(range(st_idx, step_limit, step))
#         for idx in sample_idx:
#             print("{}/{}".format(idx, step_limit))
#
#             x = x_dat[idx, :-1]
#             y = c_dat[idx, :]
#
#             plt.clf()
#             plt.bar(x, y, width=x[1]-x[0])
#             plt.title("Train Iteration {}".format(idx))
#             # plt.plot(x, y)
#             # plt.show()
#             image = fig2img(plt.gcf())
#             writer.append_data(image)





# with imageio.get_writer('denom.gif', mode='I', duration=duration, loop=0) as writer:
#     # load numpy array from csv file
#     denom_lb_hist_x = np.loadtxt(os.path.join(ifp, 'denom_lb_hist_x.csv'), delimiter=',')
#     denom_lb_hist_c = np.loadtxt(os.path.join(ifp, 'denom_lb_hist_c.csv'), delimiter=',')
#     denom_ulb_s_hist_x = np.loadtxt(os.path.join(ifp, 'denom_ulb_s_hist_x.csv'), delimiter=',')
#     denom_ulb_s_hist_c = np.loadtxt(os.path.join(ifp, 'denom_ulb_s_hist_c.csv'), delimiter=',')
#     denom_ulb_w_hist_x = np.loadtxt(os.path.join(ifp, 'denom_ulb_w_hist_x.csv'), delimiter=',')
#     denom_ulb_w_hist_c = np.loadtxt(os.path.join(ifp, 'denom_ulb_w_hist_c.csv'), delimiter=',')
#
#     if step_limit is None:
#         step_limit = denom_lb_hist_x.shape[0]
#     step_limit = min(step_limit, denom_lb_hist_x.shape[0])
#
#     sample_idx = list(range(st_idx, step_limit, step))
#
#     for idx in sample_idx:
#         print("{}/{}".format(idx, step_limit))
#
#         plt.clf()
#         plt.title("Train Iteration {}".format(idx))
#
#         x = denom_lb_hist_x[idx, :-1]
#         y = denom_lb_hist_c[idx, :]
#         plt.bar(x, y, width=x[1]-x[0])
#
#         x = denom_ulb_s_hist_x[idx, :-1]
#         y = denom_ulb_s_hist_c[idx, :]
#         plt.bar(x, y, width=x[1] - x[0])
#
#         x = denom_ulb_w_hist_x[idx, :-1]
#         y = denom_ulb_w_hist_c[idx, :]
#         plt.bar(x, y, width=x[1] - x[0])
#
#         plt.legend(['lb','ulb_w','ulb_s'])
#
#         image = fig2img(plt.gcf())
#         writer.append_data(image)