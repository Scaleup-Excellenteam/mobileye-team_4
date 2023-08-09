# This file contains some neat things with matplotlib, which you'll want to use...
import glob
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib.backend_bases import KeyEvent, MouseEvent


def plot_rects(x, y, axes=None, *args, **kwargs):
    """
    Expecting x, y to be N*4, with the bounding box coordinates. Will plot all rects
    """
    all_x = x[:, [0, 1, 1, 0, 0, 0]].astype(float)
    all_y = y[:, [0, 0, 1, 1, 0, 0]].astype(float)
    all_x[:, -1] += np.nan
    all_y[:, -1] += np.nan
    x_coords = all_x.ravel()
    y_coords = all_y.ravel()
    if axes is None:
        axes = plt.gca()
    axes.plot(x_coords, y_coords, *args, **kwargs)


class SafeConnect:
    """
    Safely perform a MPL connect: Safe because we make sure to disconnect the func on cla() execution.
    obj is anything callable (class with __call__ or a func.)
    event_type is any of: 'button_press_event', 'button_release_event', 'draw_event'
                          'key_press_event', 'key_release_event', 'motion_notify_event'
                          'pick_event', 'resize_event', 'scroll_event', 'figure_enter_event'
                          'figure_leave_event', 'axes_enter_event', 'axes_leave_event'
                          'close_event'
    Usage: instead of plt.connect(...), use safeConnect(...)
    """

    def __init__(self, event_type, obj, axes=None):
        hnd = axes.figure.canvas.mpl_connect(event_type, obj)
        axes = self._axes = plt.gca() if axes is None else axes
        additional = {'handle': hnd, 'object': obj}
        if type(axes.cla) == type(self):
            # Only append another client. This instance will be discarded.
            axes.cla._clients.append(additional)
            return
        # First time: Install self instead of the cla of current axes:
        self.origCla = self._axes.cla
        self._axes.cla = self
        self._clients = [additional]

    def __call__(self):
        """
        Called on axes cla(). Disconnects all clients, restores original cla func, calls it.
        """
        for i in self._clients:
            plt.disconnect(i['handle'])
            hasattr(i['object'], 'detach') and i['object'].detach()
            self._axes.cla = self.origCla
        self._axes.cla()


def callback_example(filtered_data, ihist_handler, bins):
    """
    An example of what to do with the callback from IHist.
    """
    print(f"I got {len(filtered_data)} events (there were total of {len(ihist_handler.data)}). Here are some:")
    print(f"From bins {bins}")
    print(filtered_data.head(5))


class IHist:
    SHOW_T = True
    SHOW_F = False

    def __init__(self, data_pd, values_col='values', labels_col='labels', callback=None):
        """
        Generate an interactive histogram. When you drag the mouse, you select a range,
        and then a function is called on that data. Pressing spacebar toggles between true, false or both.

        :param data_pd: pd.DataFrame with at least columns: self.values_col, self.labels_col.
        :param values_col: Name of column to represent the labels (True / False).
        :param labels_col: Name of column to represent the value in the histogram.
        :param callback: What to call when mouse button is raised.
        """
        self.data = data_pd.reset_index(drop=True)
        self.values_col = values_col
        self.labels_col = labels_col
        self.marker = None
        self.bins = None
        self.show_what = [self.SHOW_T, self.SHOW_F]
        self.callback = callback_example if callback is None else callback
        self.mouse_pressed = False
        self.t_hand = None
        self.f_hand = None
        self.show_normalized = False
        self.figure = None

    def show(self, bins_or_nbins=50, *args, **kwargs):
        _, bins = np.histogram(self.data[self.values_col], bins_or_nbins)

        # Hist each category:
        t_samp = self.data[self.data[self.labels_col] == True]
        f_samp = self.data[self.data[self.labels_col] == False]
        t_count, t_bins = np.histogram(t_samp[self.values_col].values, bins)
        f_count, f_bins = np.histogram(f_samp[self.values_col].values, bins)

        if self.show_normalized:
            t_count = t_count / (1e-16 + t_count.sum())
            f_count = f_count / (1e-16 + f_count.sum())

        def mid_bin(b):
            return 0.5 * (b[:-1] + b[1:])

        self.figure = plt.figure(self.figure.number if self.figure else None)
        plt.clf()
        self.plot_ax = plt.gca()
        self.t_hand = self.plot_ax.plot(mid_bin(t_bins), t_count, 'g')[0]
        self.f_hand = self.plot_ax.plot(mid_bin(f_bins), f_count, 'r')[0]
        self.set_line_widths()

        self.marker = None
        self.bins = bins
        self.idexes = np.digitize(self.data[self.values_col], bins)
        self.plot_ax.handler = self
        SafeConnect('button_press_event', self.on_mouse_down, self.plot_ax)
        SafeConnect('motion_notify_event', self.on_mouse_move, self.plot_ax)
        SafeConnect('button_release_event', self.on_mouse_up, self.plot_ax)
        SafeConnect('key_press_event', self.on_key_press, self.plot_ax)
        self.plot_ax.grid(True)

    def set_line_widths(self):
        if self.t_hand is not None:
            self.t_hand.set_lw(3 if self.SHOW_T in self.show_what else 1.5)
        if self.f_hand is not None:
            self.f_hand.set_lw(3 if self.SHOW_F in self.show_what else 1.5)
        plt.draw()

    def on_mouse_down(self, event: MouseEvent):

        if event.inaxes == None or event.inaxes != self.plot_ax:
            return

        # Delete the marker:
        if self.marker is not None:
            rect_handler = self.marker['handle']
            self.plot_ax.lines = [l for l in self.plot_ax.lines if l != rect_handler]
            self.marker = None

        self.set_line_widths()

        # Only delete marker?
        if event.button != 1:
            self.mouse_pressed = False
            return

        self.mouse_pressed = True
        self.marker = {'start_bin_idx': np.digitize(event.xdata, self.bins),
                       'handle': None
                       }

        # Generate the marker
        self.on_mouse_move(event)

    def on_mouse_move(self, event: MouseEvent):
        """
        Update a patch on the figure as the mouse moves.
        """

        if event.inaxes == None or event.inaxes != self.plot_ax or not self.mouse_pressed:
            return

        self.marker['end_bin_idx'] = np.digitize(event.xdata, self.bins)

        # Show the marker:
        bin_half_w = (self.bins[1] - self.bins[0]) / 2
        spare = bin_half_w / 1.5
        last_bins = len(self.bins) - 1
        x0 = self.bins[np.clip(self.marker['start_bin_idx'], 0, last_bins)] - bin_half_w - spare
        x1 = self.bins[np.clip(self.marker['end_bin_idx'], 0, last_bins)] - bin_half_w + spare
        mid_y = np.array([self.plot_ax.axis()[2:4]]).mean()

        if self.marker['handle'] is None:
            self.marker['handle'] = self.plot_ax.plot([x0, x1], [mid_y, mid_y], color='c', lw=5, alpha=0.3)[0]
        else:
            self.marker['handle'].set_xdata([x0, x1])
            plt.draw()

    def on_mouse_up(self, event: MouseEvent = None):

        self.mouse_pressed = False

        if self.marker is None:
            return

        b0 = min(self.marker['start_bin_idx'], self.marker['end_bin_idx'])
        b1 = max(self.marker['start_bin_idx'], self.marker['end_bin_idx'])
        cond_bins = (b0 <= self.idexes) & (self.idexes <= b1)
        cond_tf = np.in1d(self.data[self.labels_col], self.show_what)

        filtered_data = self.data.loc[cond_bins & cond_tf]

        self.callback(filtered_data, self, [b0, b1])
        _ = event  # Make pylint happy

    def on_key_press(self, event: KeyEvent):
        """
        If spacebar is hit, toggle the what-to-choose-from.
        """

        if event.key == ' ':
            opts = [[True, False], [True], [False]]
            n_opts = len(opts)
            curr_idx = opts.index(self.show_what) if self.show_what in opts else 0
            self.show_what = opts[(curr_idx + 1) % n_opts]
            self.set_line_widths()
            print(f"Now showing {self.show_what}")
        elif event.key == 'enter':
            # Redo last selection
            self.on_mouse_up()
        elif event.key in ['n', 'N']:
            self.show_normalized = not self.show_normalized
            self.show()


def ihist_example():
    """
    Run this code to see IHist object in action.
    """
    n = 250
    is_true = np.random.rand(n) > 0.5
    score = np.random.randn(n) * 3 + is_true * 10
    data = pd.DataFrame({
        'file': ["img_%04d.png" % f for f in range(n)],
        'score': score,
        'label': is_true,
    })
    ih = IHist(data, 'score', 'label')
    ih.show()
    plt.show(block=True)


class GridPresenter:
    IMAGE = 'image'

    def __init__(self, data: dict, callback=None, name='', nrow=4, ncol=6, page=0):
        """
        Show a grid of images, react when the mouse is clicked on any.

        :param data: Dict with {'image': <np.ndarray of dtype int>, ...}. All images same shape
        """
        self.data = data
        self.nrow = nrow
        self.ncol = ncol
        self.per_page = nrow * ncol
        self.page = page
        self.name = name
        n_imgs = len(data[GridPresenter.IMAGE])
        self.last_page = n_imgs // (nrow * ncol)
        self.all_images = np.stack([v for v in self.data[GridPresenter.IMAGE]], axis=0) if n_imgs > 0 else np.array([])
        self.mapping = np.zeros_like(self.all_images, dtype=int) + np.arange(len(self.all_images))[:, np.newaxis, np.newaxis, np.newaxis]
        self.grid = None
        self.mapg = None
        self.callback = callback
        self.plot_ax = None
        self.set_page(0)
        self.verbose = True

    def set_page(self, page=None):
        if page is None:
            page = self.page
        page = page % (1 + self.last_page)
        self.page = page

        def get_grid(imgs):
            import torch
            import torchvision
            tensor = torch.tensor(imgs.transpose([0, 3, 1, 2]))
            return torchvision.utils.make_grid(tensor, nrow=self.ncol).numpy().transpose([1, 2, 0])

        i0 = self.per_page * self.page
        i1 = i0 + self.per_page
        self.grid = get_grid(self.all_images[i0: i1])
        self.mapg = get_grid(self.mapping[i0: i1])[:, :, 0]

    def show(self, plot_ax=None):
        """
        Show itself.
        Upon a click on one of the images, call the callback function.
        It should receive (MPL-event, offset-of-clicked-image, self).
        """

        if plot_ax is not None:
            self.plot_ax = plot_ax
        if self.plot_ax is None:
            plt.figure(self.name)
            self.plot_ax = plt.gca()
        plot_ax = self.plot_ax
        plot_ax.cla()
        plot_ax.imshow(self.grid)
        SafeConnect('button_press_event', self.on_mouse_down, self.plot_ax)
        SafeConnect('key_press_event', self.on_key_press, self.plot_ax)
        plot_ax.title.set_text(f"Name: {self.name}: Page {self.page} of {self.last_page + 1}")
        plt.show()

    def on_key_press(self, event: KeyEvent):
        """
        Spacebar - next page, backspace - previous page.
        """
        if event.key in [' ', 'backspace']:
            self.page += (1 if event.key == ' ' else -1)
        self.set_page()
        self.show()

    def on_mouse_down(self, event: MouseEvent):
        if event.inaxes is None:
            return  # Clicked outside axes
        if self.verbose or self.callback is None:
            x, y = event.xdata, event.ydata
            offset = self.mapg[int(y), int(x)]
            text = [f'{k}: {self.data[k][offset]}' for k in sorted(list(set(self.data.keys()) - {GridPresenter.IMAGE}))]
            print(f"Clicked on image {offset}")
            print('\n'.join(text))

            if self.callback is not None:
                self.callback(event, offset, self)
            elif 'full_path' in self.data:
                the_image = np.array(Image.open(self.data['full_path'][offset]))
                zoom_ax = getattr(self, 'zoom_ax', None)
                if zoom_ax is None:
                    plt.figure(f"{self.name}_zoom")
                    plt.clf()
                    self.zoom_ax = plt.subplot(111)
                self.zoom_ax.cla()
                self.zoom_ax.imshow(the_image)
                if all([k in self.data for k in ['x0', 'x1', 'y0', 'y1']]):
                    rect_x = np.array([[self.data[x][offset] for x in ['x0', 'x1']]])
                    rect_y = np.array([[self.data[y][offset] for y in ['y0', 'y1']]])
                    plot_rects(rect_x, rect_y, axes=zoom_ax, color=self.data['col'][offset])
                plt.show()


def grid_presenter_example(some_folder_with_images):
    """
    Show the superpowers of GridPresentor.

    :param some_folder_with_images: Path to a folder with the left8bit.png files.
    """
    flist = glob.glob(os.path.join(some_folder_with_images, '*_leftImg8bit.png'))
    if len(flist) == 0:
        print(f"Nothing found in {os.path.abspath(flist)}")
        return

    acc = []
    for f in flist:
        img = np.array(Image.open(f))
        is_true = np.random.rand() > 0.5
        score = np.random.rand() * 10 + is_true * 3
        acc.append({GridPresenter.IMAGE: img, 'filename': f, 'is_true': is_true, 'score': score})

    data = {k: [line[k] for line in acc] for k in acc[0].keys()}
    plt.figure('Grid presenter example')
    ax_grid = plt.gca()
    plt.figure('Zoom view')
    ax_zoom = plt.gca()

    def on_click(event, offset, grid_presenter):
        print(f"You have clicked on {event.x}, {event.y}, which is file {grid_presenter.data['filename'][offset]}")
        ax_zoom.cla()
        img = np.array(Image.open(grid_presenter.data['filename'][offset]))
        ax_zoom.imshow(img)

    gp = GridPresenter(data, callback=on_click)
    gp.show(ax_grid)
    plt.show(block=True)


class NNResultExaminer:
    def __init__(self, data, crops_dir='.', full_img_dir='.',
                 values_col='values',
                 labels_col='labels',
                 crop_filename_col='crop_filename',
                 full_filename_col='full_filename',
                 name='', grid_presenter_callback=None):
        """
        Show a histrogram of the results, allow us to explore which results came from what.

        :param crops_dir: Folder where all crops images are.
        :param data: Dict with: 'score', 'is_true', 'is_red', 'filename'. Each points to a list/array of values.
        """
        self.data = pd.DataFrame(data).reset_index(drop=True)
        self.crops_dir = crops_dir
        self.full_img_dir = full_img_dir

        self.values_col = values_col
        self.labels_col = labels_col
        self.crop_filename_col = crop_filename_col
        self.full_filename_col = full_filename_col

        self.name = name
        self.ihist = None
        self.grid_dict = None
        self.gridex = None
        self.grid_fig = None
        self.zoom_fig = None
        self.roc_fig = None
        self.grid_presenter_callback = grid_presenter_callback  # The callback when clicking on the grid-presenter

    def show(self):
        self.ihist = IHist(self.data, self.values_col, self.labels_col, self.on_hist_select)
        self.ihist.show()

    def show_roc(self, bins_or_nbins=100):
        _, bins = np.histogram(self.data[self.values_col], bins_or_nbins)
        t_samp = self.data[self.data[self.labels_col] == True]
        f_samp = self.data[self.data[self.labels_col] == False]
        t_count, t_bins = np.histogram(t_samp[self.values_col].values, bins)
        f_count, f_bins = np.histogram(f_samp[self.values_col].values, bins)

        acc_t = t_count.cumsum() / t_count.sum()
        acc_f = f_count.cumsum() / f_count.sum()

        plt.figure(None if self.roc_fig is None else self.roc_fig.number)
        plt.clf()
        self.roc_fig = plt.gcf()
        plt.plot(acc_f, acc_t, '.-')
        plt.title(f"ROC curve of {self.name}")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.axis([0, 1, 0, 1])
        plt.grid(True)
        plt.plot(0, 1, 'ro', mfc='none', ms=15)
        plt.plot(0, 1, 'r+', mfc='none', ms=15)
        plt.text(0, 1, '\n\nPerfect is here', color='r', verticalalignment='top', horizontalalignment='center')

    def on_hist_select(self, filtered_data: pd.DataFrame, nne: "NNResultExaminer", bins: List):
        # Show the crops of the filtered_data:
        grid_dict = {k: filtered_data[k].values for k in filtered_data.keys()}
        grid_dict[GridPresenter.IMAGE] = [np.array(Image.open(fn)) for fn in filtered_data[self.crop_filename_col]]
        self.grid_dict = grid_dict
        gp = GridPresenter(self.grid_dict, self.grid_presenter_callback, self.name)
        _ = plt.figure(f"Grid: {self.name}, {bins[0]}-{bins[1]}") if self.grid_fig is None else self.grid_fig
        gp.show(plt.subplot(111))
        _ = nne


def nn_examiner_example(scores_h5_filename=r'..\stam_output\scores.h5'):
    """
    Show how to use the NN Examiner, which is really cool, but you have to prepare some data.

    :param scores_h5_filename: Path to an h5 file, which contains the following:
        data: A pd.DataFrame with columns:
            score: Float number with the given score
            is_true: Bool
            crop_path: Path to the crop file
            full_path: Path to the full image file
            col: 'r' or 'g'
            x0, x1, y0, y1 (Optional): Coordinate of the rectangle in the full image
    """

    with pd.HDFStore(scores_h5_filename, mode='r') as fh:
        results = fh['data']
        metadata = fh['metadata']

    results['gt_score'] = results['is_true'] * 1.

    # Ignore the ignores...
    results = results[~results['is_ignore']]

    nne = NNResultExaminer(results, metadata['crop_dir'], metadata['full_dir'],
                           values_col='score',
                           labels_col='is_true',
                           crop_filename_col='crop_path',
                           full_filename_col='full_path',
                           name=metadata['name'])
    nne.show()
    nne.show_roc()
    plt.show(block=True)


if __name__ == '__main__':
    # TODO: ...
    nn_examiner_example(r'C:\Users\dori\Documents\SNC\code\model_0229_on_train\scores.h5')

    # TODO: what is this?
    # ihist_example()
    # grid_presenter_example('..\..\..\data\selected')
