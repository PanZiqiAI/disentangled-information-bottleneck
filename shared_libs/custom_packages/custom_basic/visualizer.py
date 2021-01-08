
import matplotlib
matplotlib.use('Agg')

import os
import cv2
import random
import dominate
import colorsys
import numpy as np
from argparse import Namespace
from collections import OrderedDict
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from torchvision.utils import save_image as torch_save_image
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
from shared_libs.custom_packages.custom_basic.operations import chk_d


########################################################################################################################
# Utils
########################################################################################################################

def random_colors(N, bright=True, shuffle=False):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then convert to RGB.
    """
    # Generator colors
    brightness = 1.0 if bright else 0.7
    hsv = [(i * 1.0 / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # Shuffle
    if shuffle:
        random.shuffle(colors)
    return colors


def gradient_colors(n):
    """
    Gradually changes.
    :param n:
    :return:
    """
    ratio = (np.arange(0, n, dtype=np.float32) / n)[:, np.newaxis]
    colors = np.concatenate([ratio, (1.0 - ratio), np.zeros(shape=ratio.shape)], axis=1)
    return colors


def gray2heat(image_numpy, norm_method='minmax', **kwargs):
    # Normalize
    if norm_method == 'minmax':
        cv2.normalize(image_numpy, image_numpy, 0, 255, cv2.NORM_MINMAX)
        image_numpy = image_numpy.astype(np.uint8)
    elif norm_method == 'bound':
        mini, maxi = kwargs['norm_bound']
        assert maxi > mini
        length = maxi - mini
        # Calculate min
        image_min, image_max = image_numpy.min(), image_numpy.max()
        assert image_min >= mini and image_max <= maxi
        norm_min = int((image_min - mini) / length * 255.0)
        norm_max = int((image_max - mini) / length * 255.0)
        # Normalize
        cv2.normalize(image_numpy, image_numpy, norm_min, norm_max, cv2.NORM_MINMAX)
        image_numpy = image_numpy.astype(np.uint8)
    else:
        raise NotImplementedError
    # Convert
    heat_map = cv2.applyColorMap(image_numpy, cv2.COLORMAP_JET)
    heat_map = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
    return heat_map


class HTML(object):
    """
    This HTML class allows us to save images and write texts into a single HTML file.
    It consists of functions such as <add_header> (add a text header to the HTML file),
    <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
    It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title='Visualization', width=256, refresh=0):
        """
        Initialize the HTML classes
        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html;
            images will be saved at <web_dir>/images/
            title (str)   -- the webpage name
            width (int) -- images width
            refresh (int) -- how often the website refresh itself; if 0, no refreshing
        """
        # Tile & dirs
        self._title = title
        self._web_dir = web_dir
        self._img_dir = os.path.join(self._web_dir, 'images')
        if not os.path.exists(self._web_dir): os.makedirs(self._web_dir)
        if not os.path.exists(self._img_dir): os.makedirs(self._img_dir)
        self._width = width
        # Document
        self._doc = dominate.document(title=title)
        # Refresh
        if refresh > 0:
            with self._doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    @property
    def image_dir(self):
        return self._img_dir

    def add_header(self, text):
        """
        Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self._doc:
            h3(text)

    def add_images(self, ims, txts, links):
        """
        Add images to the HTML file
        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) -- a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        # Create table & add
        new_table = table(border=1, style="table-layout: fixed;")
        self._doc.add(new_table)
        # Set table
        with new_table:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % self._width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        """
        save the current content to the HMTL file
        """
        with open(os.path.join(self._web_dir, 'index.html'), 'wt') as f:
            f.write(self._doc.render())


########################################################################################################################
# API
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Visualizing packages
# ----------------------------------------------------------------------------------------------------------------------

def save_visuals_package(visuals, save_dir, **kwargs):
    # Get save_image method
    save_image = torch_save_image if 'lmd_save_image' not in kwargs.keys() else kwargs['lmd_save_image']
    # Save
    for label, image in visuals.items():
        save_image(image, os.path.join(save_dir, '%s.png' % label))


class IterVisualizer(object):
    """
    Visualize that saves visuals to directories & display them on a HTML.
    """
    def __init__(self, vis_dir, show_html_width, iter_prefix='iter', **kwargs):
        # Configs
        self._iter_prefix = iter_prefix
        # 1. Web page
        if show_html_width > 0:
            self._webpage = HTML(vis_dir, width=show_html_width)
            self._iter_container = []
            self._visual_labels = []
        else:
            if not os.path.exists(vis_dir): os.makedirs(vis_dir)
            self._webpage = Namespace(image_dir=vis_dir)
        # 2. Visualizer
        self._lmd_save_image = kwargs['lmd_save_image'] if 'lmd_save_image' in kwargs.keys() else \
            torch_save_image

    def _get_image_name(self, iter_count, label):
        """
        For example:
             - label = 'xxx':       %s[%d]_xxx
             - label = 'yyy/xxx':   yyy/%s[%d]_xxx
        """
        if '/' in label:
            label = label.split("/")
            prefix, suffix = '/'.join(label[:-1]), label[-1]
            return '%s/%s[%d]_%s' % (prefix, self._iter_prefix, iter_count, suffix)
        else:
            return '%s[%d]_%s' % (self._iter_prefix, iter_count, label)

    def save_images_to_the_disk(self, visuals, iter_count=0, **kwargs):
        # 1. Save to disk
        for label, image in visuals.items():
            # Path
            img_path = os.path.join(self._webpage.image_dir, '%s.png' % self._get_image_name(iter_count, label))
            # Save
            try:
                self._lmd_save_image(image, img_path)
            except FileNotFoundError:
                os.makedirs(os.path.split(img_path)[0])
                self._lmd_save_image(image, img_path)
        # 2. Update for website
        if isinstance(self._webpage, HTML):
            # Update key
            for key in list(visuals.keys()):
                if key not in self._visual_labels:
                    self._visual_labels.append(key)
            # Update iter
            self._iter_container.append(iter_count)
            # Flush website
            if chk_d(kwargs, 'flush_website'):
                self.save_website()

    def save_website(self):
        if not isinstance(self._webpage, HTML): return
        # 1. Save website
        # (1) Add each row
        for n in self._iter_container:
            # (1) Generate display
            ims, txts, links = [], [], []
            for label in self._visual_labels:
                img_path = '%s.png' % self._get_image_name(n, label)
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
            # (2) Add
            self._webpage.add_header('%s[%.3d]' % (self._iter_prefix, n))
            self._webpage.add_images(ims, txts, links)
        # (2) Save to disk
        self._webpage.save()
        # 2. Reset container
        self._iter_container = []


# ----------------------------------------------------------------------------------------------------------------------
# Unsorted
# ----------------------------------------------------------------------------------------------------------------------

def plot_two_scalars_in_one(scalars1, scalars2, x, x_label, title, save_path):
    """
    :param scalars1: {data: ..., y_label: ..., color: ...)
    :param scalars2: {data: ..., y_label: ..., color: ...)
    :param x:
    :param x_label:
    :param save_path:
    :param title:
    :return:
    """
    # 1. Init fig
    _, ax1 = plt.subplots(dpi=200)
    # 2. Plotting two figures
    if x is None: x = np.arange(len(scalars1['data']))
    ax1.set_xlabel(x_label)
    # (1) Part 1
    ax1.set_ylabel(scalars1['y_label'], color=scalars1['color'])
    ax1.plot(x, scalars1['data'], color=scalars1['color'])
    # (2) Part 2
    ax2 = ax1.twinx()
    ax2.set_ylabel(scalars2['y_label'], color=scalars2['color'])
    ax2.plot(x, scalars2['data'], color=scalars2['color'])
    # Other setting
    if title is not None: plt.title(title)
    plt.tight_layout()
    # 3. Save
    plt.savefig(save_path)
    plt.close()


def plot_two_scalars_vert(scalars1, scalars2, save_path, **kwargs):
    """
    :param scalars1: (x), data, (title), (x_label), y_label, (y_lim)
    :param scalars2: (x), data, (title), x_label, y_label, (y_lim)
    :param save_path:
    :return:
    """
    # 1. Init figure
    fig = plt.figure(dpi=200)
    if 'title' in kwargs.keys(): fig.suptitle(kwargs['title'])
    # 2. Two figures
    # (1) Figure 1
    plt.subplot(211)
    if 'title' in scalars1.keys(): plt.title(scalars1['title'])
    if 'x_label' in scalars1.keys(): plt.xlabel(scalars1['x_label'])
    plt.ylabel(scalars1['y_label'])
    plt.plot(scalars1['x'] if 'x' in scalars1.keys() else np.arange(len(scalars1['data'])), scalars1['data'])
    if 'y_lim' in scalars1.keys(): plt.ylim(*scalars1['y_lim'])
    # (2) Figure 2
    plt.subplot(212)
    if 'title' in scalars2.keys(): plt.title(scalars2['title'])
    plt.xlabel(scalars2['x_label'])
    plt.ylabel(scalars2['y_label'])
    plt.plot(scalars2['x'] if 'x' in scalars2.keys() else np.arange(len(scalars2['data'])), scalars2['data'])
    if 'y_lim' in scalars2.keys(): plt.ylim(*scalars2['y_lim'])
    # Show
    plt.tight_layout()
    if 'title' in kwargs.keys():
        if 'top' in kwargs.keys(): plt.subplots_adjust(top=kwargs['top'])
        else: plt.subplots_adjust(top=0.9)
    # Save
    plt.savefig(save_path)
    plt.close()


def plot_elapsed_scalars(scalars, x, x_label, y_label, title, save_path):
    """
    Plotting multiple scalars.
    :param scalars: {label: data}
    :param x:
    :param x_label:
    :param y_label:
    :param title:
    :param save_path:
    :return:
    """
    # 1. Init figure & setup
    plt.figure(dpi=200)
    if title is not None: plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # 2. Plot multi scalars
    if x is None: x = np.arange(len(list(scalars.values())[0]))
    colors = gradient_colors(len(scalars))
    for index, (k, line) in enumerate(scalars.items()):
        assert len(x) == len(line)
        kwargs = {'label': k} if index in [0, len(scalars) - 1] else {}
        plt.plot(x, line, color=colors[index], **kwargs)
    # 3. Save
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# Using t-SNE to visualize high dimensional representations
def visualize_latent_by_tsne(data, label, title, vis_dir, vis_name):
    """
    :param data: Numpy.array. (batch, ...)
    :param label: Numpy.array. (batch, )
    :param title:
    :param vis_dir:
    :param vis_name:
    :return:
    """
    # Randomly generate colors.
    n_classes = len(set(label))
    colors = random_colors(n_classes)
    # 1. Reduce dimensions to (batch, 2)
    data_tsne = TSNE().fit_transform(data)
    # 2. Visualization.
    # (1) Clustering to n_classes sets.
    # 1> Init result.
    data_dict = {}
    # 2> Traverse each data & label
    for x_tsne, y in zip(data_tsne, label):
        if y not in data_dict.keys(): data_dict.update({y: []})
        data_dict[y].append(x_tsne[np.newaxis])
    # (2) Visualize each class.
    for y, data_class in data_dict.items():
        data_class = np.concatenate(data_class, axis=0)
        plt.scatter(data_class[:, 0], data_class[:, 1], c=colors[y])
    # (3) Title & save
    plt.title(title)
    plt.savefig(os.path.join(vis_dir, '%s.png' % vis_name))
    plt.close()


# Plotting multi axes with confidence interval
def plot_multi_axes(figsize, multi_scalars, axes_kwargs, save_dir, save_name, **kwargs):
    """
    :param figsize:
    :param multi_scalars: dict:
        { legend_scalars1: {
            'data': [(x_axes1, y_axes1), (x_axes2, y_axes2), ...],
            'y_bounds': [value_axes1, value_axes2, ...], where 'value' could be N/A or (upper, lower),
            'color': shared color across multi axes of current scalars,
            'linestyle': shared line style across multi axes of current scalars,
            'markersize': linestyle size,
            'irregular': N/A or 'vertical_line', 'horizontal_line', 'scatter' },
          legend_scalars2: {...},
          ... }
    :param axes_kwargs: List of dict. Each dict could have keys: rect, xlabel, ylabel, xlim, ylim.
    :param save_dir:
    :param save_name:
    :param kwargs: Shared kwargs. Could have keys:
        legend: legend_kwargs_dict, e.g., { loc='lower center', ncol=2 }
    :return:
    """
    # 1. Get figure & axes
    fig = plt.figure(figsize=figsize, dpi=500)
    # 2. Plot each axes
    # (1) Init legend
    legend_lines = OrderedDict()
    # (2) Plot
    for axis_index, axis_kwargs in enumerate(axes_kwargs):
        # 1. Get ax
        assert 'rect' in axis_kwargs.keys()
        ax = plt.axes(axis_kwargs['rect'])
        # 2. Plot each line
        for legend, scalars in multi_scalars.items():
            # Check data
            scalars_data_axis = scalars['data'][axis_index]
            if scalars_data_axis is None: continue
            ############################################################################################################
            # Kwargs for plotting current scalar within currect axis
            ############################################################################################################
            plot_kwargs = {}
            if 'markersize' in scalars.keys(): plot_kwargs['markersize'] = scalars['markersize']
            if 'color' in scalars.keys(): plot_kwargs['color'] = scalars['color']
            linestyle = scalars['linestyle'] if 'linestyle' in scalars.keys() else '-'
            ############################################################################################################
            # Plotting
            ############################################################################################################
            # 1. regular
            if 'irregular' not in scalars.keys():
                ########################################################################################################
                # 1. Curve
                ########################################################################################################
                if isinstance(scalars_data_axis, tuple) and isinstance(scalars_data_axis[0], list):
                    # Plot line
                    line, = ax.plot(*scalars_data_axis, linestyle, label=legend, **plot_kwargs)
                    # Plot confidence interval
                    if 'y_bounds' in scalars.keys():
                        upper, lower = scalars['y_bounds'][axis_index]
                        plt.fill_between(scalars_data_axis[0], upper, lower, color=plot_kwargs['color'], alpha=0.1)
                    # Legend
                    legend_lines[legend] = line
                ########################################################################################################
                # 2. Horizontal line
                ########################################################################################################
                elif isinstance(scalars_data_axis, float):
                    # Plot
                    line = ax.axhline(scalars_data_axis, linestyle='--', label=legend, **plot_kwargs)
                    # Legend
                    if legend not in legend_lines.keys(): legend_lines[legend] = line
                else:
                    raise NotImplementedError
            # 2. irregular
            else:
                if scalars['irregular'] == 'vertical_line':
                    # (1) Plot line
                    ax.axvline(scalars_data_axis, linestyle='--', **plot_kwargs)
                    # (2) Show legend
                    if 'text_loc' in scalars.keys():
                        ax.text(
                            *scalars['text_loc'][axis_index], legend,
                            **({} if 'text_kwargs' not in scalars.keys() else scalars['text_kwargs'][axis_index]))
                else:
                    # (1) Plot line
                    ax.axhline(scalars_data_axis, linestyle='--', **plot_kwargs)
                    # (2) Show legend
                    if 'text_loc' in scalars.keys():
                        ax.text(
                            *scalars['text_loc'][axis_index], legend,
                            **({} if 'text_kwargs' not in scalars.keys() else scalars['text_kwargs'][axis_index]))
        # 3. Kwargs for plotting currect axis
        if 'xlabel' in axis_kwargs.keys(): ax.set_xlabel(axis_kwargs['xlabel'])
        if 'ylabel' in axis_kwargs.keys(): ax.set_ylabel(axis_kwargs['ylabel'])
        if 'xlim' in axis_kwargs.keys(): plt.xlim(*axis_kwargs['xlim'])
        if 'ylim' in axis_kwargs.keys(): plt.ylim(*axis_kwargs['ylim'])
        if 'title' in axis_kwargs.keys():
            if isinstance(axis_kwargs['title'], str):
                title, title_kwargs = axis_kwargs['title'], {}
            else:
                title, title_kwargs = axis_kwargs['title']
            ax.set_title(title, **title_kwargs)
    # (3) Set legend
    fig.legend(handles=list(legend_lines.values()), labels=list(legend_lines.keys()),
               **(kwargs['legend'] if 'legend' in kwargs.keys() else {}))
    # 3. Save
    plt.savefig(os.path.join(save_dir, '%s.png' % save_name))
    plt.close()
