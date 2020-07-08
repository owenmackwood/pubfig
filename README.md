# pubfig

A simple module for creating publication-quality figures, and recreating them at the press of a button.

## Why pubfig?

In some respects, this module doesn't do anything that can't already be done with [Matplotlib](https://matplotlib.org/) and [svgutils](https://svgutils.readthedocs.io).
So why does it exist? While `svgutils` is a helpful package, and enables you to programmatically create complete publication-quality figures, it has some rough edges.
These include its apparent inability to correctly handle units in a variety of scenarios, and the fact that when loading SVGs it throws away information essential for compositing the final figure correctly.

So why `pubfig`? It smooths out those rough edges, permitting you exploit the capabilities of `svgutils`, while not having to worry about units, or the peculiarities of how it deals with SVG files. `pubfig` also features the ability to specify an entire publication-ready figure in a single location, and automates away all of the details of compositing with `svgutils`.

Generating elaborate multipanel figures is already possible in Matplotlib (via nested `GridSpec` objects), though it can require substantial effort to get the subplot layout just right. Another frustrating issue is that Matplotlib automates certain aspects of layout and alignment, and can necessitate very careful massaging to get things where you want them. This is precisely where `pubfig` comes in: For aspects of the layout that you want automated (e.g. an N x M grid of subplots), `pubfig` lets Matplotlib handle it. For other aspects where you want precise control over the location of a figure element, `pubfig` allows you to exactly specify it and it will **always** appear in that location, and its appearance will never be affected by other figure elements.

Additional highlights include the ability to [memoize](https://en.wikipedia.org/wiki/Memoization) plots to disk. This makes it possible to skip processing and plotting the underlying data, potentially offering significant speedups when regenerating a figure (something that tends to happen many times when tweaking the final layout). Finally, `pubfig` can also automatically output high-DPI raster images in either PNG (requires Inkscape) or TIFF formats  (requires Inkscape and ImageMagick).

## Requirements

* Python version >= 3.6
* svgutils
* Matplotlib

## Quick start guide

To specify the contents and layout of a figure using `pubfig` one merely defines a class that inherits from `pubfig.FigureSpec` (a figure specification). Your `FigureSpec` class has a few attributes that you are required to provide. At minimum, you must provide:
 
 1) `output_file`: The path and filename for the output SVG file (this can be a `pathlib.Path` or `str`).
 2) `figure_size`: An `ElemSize` object that specifies the `width`, `height`, and `units` of the figure.
 3) `panels`: A `PanelsSpec` where each entry defines an element to be included in the figure. Each element can be either a Matplotlib figure (indicated by using the `PanelFig` class), or an image e.g. a diagram or schematic (by using the `Panel` class).
 
There are other useful attributes of the `FigureSpec`, but for simplicity we'll ignore them for now.

As mentioned, the `panels` attribute of `FigureSpec` must be a `PanelsSpec` with entries that are either a `PanelFig` or `Panel`. The `PanelFig` class is used to specify a Matplotlib `Figure`, and so the constructor requires an `ElemSize` object that defines the `Figure` size (`width`, `height`, `units`). It also accepts a `gridspec_kwargs` dictionary, which enables you to define the number of rows and columns in the Matplotlib `GridSpec` that will be constructed by `pubfig` (along with any other argument accepted by the `Figure.add_gridspec` function). If `gridspec_kwargs` is not provided, then a `GridSpec` of one row and one column is constructed.

The `Panel` class is used to specify an image that should be included in the figure (currently SVG, EPS, PDF, and some raster formats such as PNG are supported). To include an vector image, make a `pubfig.VectorImage` object by passing the file's path to its constructor. The image will be composited into the figure at the correct size (i.e. the absolute dimensions of the image included in the final figure will be the same as the absolute dimensions defined in the vector image file). If you want to rescale the image, pass a `scale` value to the `Panel` constructor.

Here is an example of a figure with three panels, the first is an SVG loaded from the current directory (the `schematic`), the second is a Matplotlib `Figure` with a single plot (`noise_image`), and the third is also a `Figure` (called `time_series`), but this time with two rows and two columns. 

```python
from pubfig import Units, FigureSpec, PanelsSpec, ElemSize, Panel, PanelFig, VectorImage, Location
from pathlib import Path

units = Units.cm

class Figure1_Panels(PanelsSpec):
    schematic: Panel = Panel(
            VectorImage("./images/schematic.svg"),
            Location(.25, .5),
        )
    noise_image: PanelFig = PanelFig(
            ElemSize(2., 2., units),
            Location(4.5, .5),
        )
    time_series: PanelFig = PanelFig(
            ElemSize(6, 4, units),
            Location(.25, 4.75),
            gridspec_kwargs=dict(nrows=2, ncols=2, wspace=1, hspace=1),
        )

class Figure1(FigureSpec):
    figure_size = ElemSize(7.5, 9.0, units)
    output_file = Path.home() / "publication2020c" / "figures" / "figure1.svg"
    panels = Figure1_Panels()
```

Now that the figure has been defined, we need some code to do the actual plotting. The preferred way to do this is to write functions that take an argument of type `PanelFig`, one for each `PanelFig` defined in your `panels`. Inside of those functions, you use the standard Matplotlib functionality to add subplots to the figure and plot data to them. This is accomplished by accessing the `PanelFig.fig` attribute (which is the Matplotlib `Figure` object), and the `PanelFig.gridspec` attribute (which is the `GridSpec` associated with the `Figure`). Note: Since `pubfig` is just constructing standard `GridSpec` objects for you, feel free to use Matplotlib's `GridSpecFromSubplotSpec` to make an arbitrarily complex Matplotlib `Figure` for any single panel.

```python
def plot_noise_image(panel: PanelFig):
    import numpy as np
    """ 
    PanelFig objects have a plt.Figure, with the size requested in the spec.
    They also have a plt.gridspec.GridSpec object, also defined in the spec.
    
    Because we didn't specify any gridspec_kwargs for this 
    panel, pubfig made a 1 x 1 GridSpec for us.
    """
    ax = panel.fig.add_subplot(panel.gridspec[0])
    ax.pcolor(np.random.uniform(size=50*50).reshape(50, 50))
    ax.set(xticks=(), yticks=())


def plot_time_series(panel: PanelFig):
    import numpy as np
    """"
    In this case, the gridspec has shape 2 x 2, as per the `time_series`
    PanelFig gridspec_kwargs.
    """
    gs = panel.gridspec
    plots = gs.get_geometry()
    for k in range(np.product(plots)):
        i, j = np.unravel_index(k, plots)
        ax = panel.fig.add_subplot(gs[i, j])
        ax.plot(np.cos(np.linspace(-k*np.pi, k*np.pi, 10**3)))
        ax.set(ylim=(-1.05, 1.05))
        spines_frames(ax)
```

Finally we can put all the pieces together to plot and composite your figure. The simplest way to accomplish this is to use the `pubfig.compositor` decorator. The decorator takes your `FigureSpec` class, constructs it for you, and passes the instance to the decorated function. You can then access your figure panels, and plot to them.

```python
from pubfig import compositor

@compositor(Figure1, memoize_panels=True, recompute_panels=False)
def create_fig1(figure: Figure1):
    """A function that does the actual plotting of data (note the decorator!)"""
    plot_noise_image(figure.panels.noise_image)
    plot_time_series(figure.panels.time_series)
    # No need to access figure.panels.schematic since it is loaded from disk.
```

## To learn more

There are many more features provided by `pubfig`. You can learn more about them by first reading and running the code in `examples.py`. Next, you can start the IPython interpreter, and try the following:

```
>>> import pubfig as pf
>>> pf?  # Prints the main docstring of the module
>>> pf.FigureSpec?  # The docstring for the FigureSpec class, etc.
>>> pf.Panel?
>>> pf.compositor? 
```

... and so on.

## A note on units

A important part of `pubfig` is how it handles units. In general, it is possible to specify most locations or sizes with their own units. That includes the `location` and `content_offset` attributes of `Panel` and `PanelFig`. If you do not specify units for `location`, then it will be interpreted as having the same units as the `FigureSpec.fig_size` object. If `content_offset` is unitless, it takes the units of the panel's `location`, and if that is not set, then the `FigureSpec.fig_size` units are used. `Text` locations (their `x` and `y` attributes) have the same units as the `Panel.location`, and if no units were specified for that, then they take the units of the `FigureSpec`.

An important special case is the `plt_fig_size` of `PanelFig`, for which its units are used when constructing the Matplotlib `Figure` object, but not for anything else. 
 