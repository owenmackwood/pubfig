from typing import NamedTuple
from pathlib import Path
import tempfile
from pubfig import FigureSpec, ElemSize, Units, ImageType, Location, SVG, Subplot, SubplotFig, Text
from pubfig import compositor, spines_frames

units = Units.cm


class Figure1(FigureSpec):
    class Subplots(NamedTuple):

        schematic: Subplot = Subplot(
            SVG(Path("./schematic.svg")),
            Location(.25, .5),
            figure_offset=Location(.5, .5),
            text=Text("This is a schematic", 0, -.2, size=10, weight="bold", font="serif")
        )

        noise_image: SubplotFig = SubplotFig(
            ElemSize(3., 3., units),
            Location(4.5, .5)
        )

        time_series: SubplotFig = SubplotFig(
            ElemSize(6, 4, units),
            Location(.25, 4.5),
            figure_offset=Location(.5, 0),
            gridspec_kwargs=dict(nrows=2, ncols=2, wspace=1., hspace=1)
        )

    subplots = Subplots()
    figure_size = ElemSize(7.5, 9, units)
    svg_path = Path(tempfile.gettempdir()) / "figure_1.svg"
    plot_grid_every = 1
    generate_image = ImageType.tiff
    image_dpi = 300


@compositor(Figure1)
def create_fig1(figure: Figure1):
    subplot_noise_image(figure.subplots.noise_image)
    subplot_time_series(figure.subplots.time_series)


def subplot_noise_image(sp_fig: SubplotFig):
    import numpy as np
    ax = sp_fig.fig.add_subplot(sp_fig.gridspec[0])
    ax.pcolor(np.random.uniform(size=50*50).reshape(50, 50))
    ax.set(xticks=(), yticks=())


def subplot_time_series(sp_fig: SubplotFig):
    import numpy as np
    gs = sp_fig.gridspec
    plots = gs.get_geometry()
    for k in range(np.product(plots)):
        i, j = np.unravel_index(k, plots)
        ax = sp_fig.fig.add_subplot(gs[i, j])
        ax.plot(np.cos(np.linspace(-k*np.pi, k*np.pi, 10**3)))
        ax.set(ylim=(-1.05, 1.05))
        spines_frames(ax)


if __name__ == "__main__":
    create_fig1()
