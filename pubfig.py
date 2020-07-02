from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Tuple, Union, Dict, Any, Optional
from enum import Enum
import matplotlib.pyplot as plt
import svgutils.compose as sc
from matplotlib.gridspec import GridSpec

plt_ppi = 72  # Matplotlib defines 1 pt to be 1/72 of an inch
cm_per_inch = 2.54


class Units(Enum):
    inch = 1
    cm = 2
    pt = 3


class ElemSize(NamedTuple):
    width: Union[float, int]
    height: Union[float, int]
    units: Units


class Location(NamedTuple):
    x: Union[float, int]
    y: Union[float, int]
    units: Optional[Units] = None  # None if units from associated ElemSize should be used


class ImageType(Enum):
    none = 1
    tiff = 2
    png = 3


class Text:
    def __init__(self, text: str, x: Union[float, int], y: Union[float, int], **kwargs):
        self.text: str = text
        self.x: Union[float, int] = x
        self.y: Union[float, int] = y
        self.kwargs: Dict[str, Any] = kwargs


class Subplot:
    def __init__(
            self,
            figure: Union[plt.Figure, sc.SVG],
            location: Location,
            text: Optional[Union[Text, Tuple[Text, ...]]] = None,
            auto_label: bool = True,
            label_location: Location = Location(0, 0),
    ):
        self.figure: Union[plt.Figure, sc.SVG] = figure
        self.location: Location = location
        self.text: Optional[Tuple[Text, ...]] = text if isinstance(text, tuple) or text is None else (text,)
        self.auto_label: bool = auto_label
        self.label_location: Location = label_location


class SubplotFig(Subplot):
    def __init__(
            self,
            subplot_size: ElemSize,
            location: Location,
            text: Optional[Union[Text, Tuple[Text, ...]]] = None,
            gridspec_kwargs: Optional[Dict[str, Any]] = None,
            auto_label: bool = True,
            label_location: Location = Location(0, 0),
    ):
        gs_kwargs = gridspec_kwargs or dict()
        gs_kwargs.setdefault("nrows", 1)
        gs_kwargs.setdefault("ncols", 1)

        figure: plt.Figure = plt.figure(figsize=figure_size_in_inches(subplot_size))
        self.gridspec: GridSpec = figure.add_gridspec(**gs_kwargs)

        super().__init__(figure, location, text, auto_label, label_location)


class AutoLabelOptions(NamedTuple):
    first_char: str = "a"
    font: str = "sans"
    size: int = 12
    weight: str = "bold"


class FigureSpec(SimpleNamespace):
    class Subplots(NamedTuple):
        """
        Just a placeholder for the user definition of Subplots,
        so that we can specify that FigureSpec has
        subplots: NamedTuple[Subplot, ...]
        which doesn't work due to a TypeError.
        """
        subplot: Union[Subplot, SubplotFig]

    svg_path: Union[Path, str]
    size: ElemSize
    subplots: Subplots
    plot_grid_every: Union[float, int] = 0
    generate_image: ImageType = ImageType.none
    auto_label_options: AutoLabelOptions = AutoLabelOptions()


def composite(figure: FigureSpec, delete_png=False):
    import tempfile

    tempdir = Path(tempfile.gettempdir())
    panels = []
    if figure.plot_grid_every > 0:
        pts_per_unit = get_pts_per_unit(figure.size.units)
        grid_every = figure.plot_grid_every * pts_per_unit
        panels.append(sc.Grid(grid_every, grid_every, size=0))

    label_n = 0
    auto_label = figure.auto_label_options._asdict()
    first_char = auto_label.pop("first_char")
    for name in figure.subplots._fields:
        subplot = getattr(figure.subplots, name)
        assert isinstance(subplot, Subplot)

        panel_elements = []

        assert isinstance(subplot.figure, (plt.Figure, sc.SVG))

        if isinstance(subplot.figure, plt.Figure):
            fn = tempdir / f"{name}.svg"
            subplot.figure.savefig(fn, transparent=True)
            """
            Rescale required due to matplotlib saving SVG files with correct 'size',
            but with wrong scale. To see this, open the SVG in Inkscape, and change its 
            units to inches, they will be as requested. 
            But inexplicably, the 'user units per pixel' is reported by Inkscape as 0.75.
            This might have something to do with matplotlib defining the SVG size with
            units of 'pt'. Though the viewBox attribute has no units and so presumably should
            also have units of pt, but might actually be defaulting to 'px' (the user unit). 
            I find this all deeply confusing, and it is not at all clear why exactly this is, 
            but might related to the fact that 1 pt == 4/3 px, by some definitions (e.g. CSS).
            Regardless, it causes the subplot-figure to appear at 3/4 of its
            actual size in the composite image.
            https://www.w3.org/TR/SVG11/coords.html#Units
            One thing to try might be to specify figures in pts, based on the default DPI
            used by matplotlib, or alter the default DPI to bring 1 pt == 1 px
            """
            svg = sc.SVG(fn).scale(4/3)
            panel_elements.append(svg)
        else:
            panel_elements.append(subplot.figure)

        if subplot.text is not None:
            panel_elements += [
                sc.Text(
                    t.text,
                    *location_to_str(figure.size.units, Location(t.x, t.y, subplot.location.units)),
                    **t.kwargs
                )
                for t in subplot.text]

        if subplot.auto_label:
            label = sc.Text(
                f"{chr(ord(first_char) + label_n)}",
                *location_to_str(figure.size.units, subplot.label_location),
                **auto_label
            )
            label_n += 1
            panel_elements.append(label)

        panel = sc.Panel(*panel_elements)
        location = location_to_str(figure.size.units, subplot.location)
        panels.append(panel.move(*location))

    def dim2str(width_height: Union[float, int], units: Units):
        return f"{width_height:.2f}in" if units == Units.inch else f"{width_height:.2f}{units.name}"

    sc.Figure(dim2str(figure.size.width, figure.size.units),
              dim2str(figure.size.height, figure.size.units),
              *panels).save(figure.svg_path)

    if figure.generate_image != ImageType.none:
        """ Taken from this shell script:
        #!/bin/sh

        # Convert all arguments (assumed SVG) to a TIFF acceptable to PLOS
        # Requires Inkscape and ImageMagick 6.8 (doesn't work with 6.6.9)

        for i in $@; do
          BN=$(basename $i .svg)
          inkscape --without-gui --export-png="$BN.png" --export-dpi 400 $i
          convert -compress LZW -alpha remove $BN.png $BN.tiff
          mogrify -alpha off $BN.tiff
          rm $BN.png
        done
        """
        basename = f"{figure.svg_path}".rstrip(".svg")
        image_name = f"{basename}.png"
        run(f"inkscape --without-gui --export-png='{image_name}' --export-dpi 400 {figure.svg_path}")
        if figure.generate_image == ImageType.tiff:
            tiff_name = f"{basename}.tiff"
            run(f"convert -compress LZW -alpha remove {image_name} {tiff_name}")
            run(f"mogrify -alpha off {tiff_name}")
            if delete_png:
                run(f"rm {image_name}")
            image_name = tiff_name
        run(f"eog {image_name}")


def get_pts_per_unit(units: Units) -> Union[float, int]:
    if units == Units.inch:
        pts_per_unit = plt_ppi
    elif units == Units.cm:
        pts_per_unit = plt_ppi / cm_per_inch
    else:
        pts_per_unit = 1
    return pts_per_unit


def figure_size_in_inches(fig_size: ElemSize) -> Tuple[float, ...]:
    # Convert the subplot_size to inches, the units used by matplotlib to specify figure size.
    if fig_size.units == Units.pt:
        return tuple(wh / plt_ppi for wh in (fig_size.width, fig_size.height))
    elif fig_size.units == Units.cm:
        return tuple(wh / cm_per_inch for wh in (fig_size.width, fig_size.height))
    else:
        return tuple(wh for wh in (fig_size.width, fig_size.height))


def location_to_str(default_units: Units, loc: Location) -> Tuple[str, ...]:
    units = loc.units or default_units
    pts_per_unit = get_pts_per_unit(units)
    return tuple(f"{xy*pts_per_unit}" for xy in (loc.x, loc.y))


def run(command, check=True, shell=True):
    import subprocess as sp
    try:
        cp = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE, check=check, shell=shell)
        print(cp.stdout.decode())
    except sp.CalledProcessError as e:
        print(f"Returned error (exit status {e.returncode}:\n {e.stderr.decode()}")


def spines_frames(ax, left=True, bottom=True, top=False, right=False, show_frames=False):
    ax.patch.set_visible(show_frames)
    kw = {'top': top, 'right': right, 'left': left, 'bottom': bottom}
    for k, v in kw.items():
        ax.spines[k].set_visible(v)
