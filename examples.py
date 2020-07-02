import numpy as np
import tempfile
import matplotlib.pyplot as plt
import svgutils.compose as sc
from pathlib import Path
from typing import NamedTuple
from pubfig import FigureSpec, ElemSize, Units, ImageType, Location, Subplot, SubplotFig, Text
from pubfig import composite, spines_frames


def example():
    units = Units.cm
    class Figure1(FigureSpec):
        size = ElemSize(7.5, 9.0, units)
        svg_path = Path(tempfile.gettempdir()) / "figure_1.svg"
        plot_grid_every = 2.5
        generate_image = ImageType.tiff

        class Subplots(NamedTuple):
            schematic: Subplot \
                = Subplot(sc.SVG(f"{get_schematic_path()}").scale(3/4),
                          Location(0.5, 1),
                          auto_label=True,
                          text=Text("This is a schematic", 0, 4.5, size=10, weight="bold", font="serif"),)

            noise_image: SubplotFig \
                = SubplotFig(ElemSize(3.0, 3.0, units),
                             Location(5.0, 1),
                             gridspec_kwargs=dict(bottom=0, left=0, top=1, right=1))

            time_series: SubplotFig \
                = SubplotFig(ElemSize(6.0, 4.0, units),
                             Location(1, 6),
                             label_location=Location(0, .5),
                             gridspec_kwargs=dict(nrows=2, ncols=2, wspace=1., hspace=1))

        subplots = Subplots()

    figure = Figure1()

    subplot_noise_image(figure.subplots.noise_image)
    subplot_time_series(figure.subplots.time_series)

    composite(figure)


def subplot_noise_image(sp_fig: SubplotFig):
    import numpy as np
    ax = sp_fig.figure.add_subplot(sp_fig.gridspec[0])
    ax.pcolor(np.random.uniform(size=50*50).reshape(50, 50))
    ax.set(xticks=(), yticks=())


def subplot_time_series(sp_fig: SubplotFig):
    import numpy as np
    gs = sp_fig.gridspec
    plots = gs.get_geometry()
    for k in range(np.product(plots)):
        i, j = np.unravel_index(k, plots)
        ax = sp_fig.figure.add_subplot(gs[i, j])
        ax.plot(np.cos(np.linspace(-k*np.pi, k*np.pi, 10**3)))
        ax.set(ylim=(-1.05, 1.05))
        spines_frames(ax)


def get_schematic_path() -> Path:
    fig: plt.Figure = plt.figure(figsize=(2., 2.))
    ax: plt.Axes = fig.add_subplot(1, 1, 1, projection='polar')
    ax.plot(np.arange(20))
    schematic_file = Path(tempfile.gettempdir()) / 'schematic.svg'
    fig.savefig(schematic_file)
    return schematic_file


if __name__ == "__main__":
    example()
