"""
A simple example of how to use pubfig.

See the README.md or the documentation in the pubfig module for more information.
"""
from pathlib import Path
import tempfile
from pubfig import FigureSpec, PanelsSpec, ElemSize, Units, ImageType, Location, VectorImage, Panel, PanelFig, Text
from pubfig import compositor, spines_frames

units = Units.cm  # Try changing this to `Units.inch`


class Figure1(FigureSpec):
    class Panels(PanelsSpec):

        schematic: Panel = Panel(
            VectorImage("./images/schematic.pdf"),  # Loaded images use the size specified in the SVG file.
            # RasterImage("./images/schematic.png", ElemSize(31, 30, Units.mm)),  # Can also load raster images
            Location(.25, .5),  # Location of the upper left of the panel (and label if not disabled)
            content_offset=Location(.5, .5),  # How far to shift this SVG relative to the panel location
            text=Text(  # Can also be a tuple of multiple Text objects
                "This is a schematic", .1, -.2, size=10, weight="bold", font="serif"
            ),
        )

        noise_image: PanelFig = PanelFig(
            ElemSize(2., 2., units),  # Physical dimensions of the plt.Figure
            Location(4.5, .5),
            content_offset=Location(.5, .5),
            gridspec_kwargs=dict(left=0, bottom=0, right=1, top=1)  # Passed to fig.add_gridspec()
        )

        time_series: PanelFig = PanelFig(
            ElemSize(6, 4, units),
            Location(.25, 4.75),
            content_offset=Location(.5, 0),
            gridspec_kwargs=dict(nrows=2, ncols=2, wspace=1., hspace=1)
        )

    panels = Panels()
    figure_size = ElemSize(7.5, 9.0, units)
    output_file = Path(tempfile.gettempdir()) / "figure_1.svg"  # SVG extension isn't strictly necessary
    plot_grid_every = 1  # Set this to zero to turn off the grid
    generate_image = ImageType.tiff  # Could also be `png` or `none`
    image_dpi = 300  # Only used if `generate_image` is not `none`


@compositor(Figure1, memoize_panels=True, recompute_panels=False)
def create_fig1(figure: Figure1):
    """A function that does the actual plotting of data (note the decorator!)"""
    plot_noise_image(figure.panels.noise_image)
    plot_time_series(figure.panels.time_series)
    # No need to access figure.panels.schematic since it is always loaded from disk.


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


if __name__ == "__main__":
    create_fig1()
