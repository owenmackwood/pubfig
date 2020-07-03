import tempfile
from pathlib import Path
from typing import NamedTuple
from pubfig import FigureSpec, ElemSize, Units, ImageType, Location, SVG, Subplot, SubplotFig, Text
from pubfig import composite, spines_frames


def example():
    units = Units.inch
    class Figure1(FigureSpec):
        size = ElemSize(7.5, 9.0, units)
        svg_path = Path(tempfile.gettempdir()) / "figure_1.svg"
        plot_grid_every = 1
        generate_image = ImageType.tiff
        image_dpi = 300

        class Subplots(NamedTuple):
            schematic: Subplot \
                = Subplot(SVG(get_schematic_path()),
                          Location(.25, .5),
                          auto_label=True,
                          figure_offset=Location(.5, .25),
                          text=Text("This is a schematic", 0, -.2, size=10, weight="bold", font="serif"), )

            noise_image: SubplotFig \
                = SubplotFig(ElemSize(3.0, 3.0, units),
                             Location(4.25, .5),
                             figure_offset=Location(.2, .25),
                             gridspec_kwargs=dict(bottom=0, left=0, top=1, right=1))

            time_series: SubplotFig \
                = SubplotFig(ElemSize(6.0, 4.0, units),
                             Location(.25, 4.5),
                             figure_offset=Location(.5, 0),
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
    schematic_file = Path("./schematic.svg")
    # schematic_file = Path("./empty_mm_mm1.svg")
    # schematic_file = Path("./empty_in_mm254.svg")
    # schematic_file = Path("./empty_mm_in.svg")
    return schematic_file


if __name__ == "__main__":
    example()
