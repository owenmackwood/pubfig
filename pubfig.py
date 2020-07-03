from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Tuple, Union, Dict, Any, Optional, Callable, Type, TypeVar
from enum import Enum
import matplotlib.pyplot as plt
import svgutils.compose as sc
from matplotlib.gridspec import GridSpec

plt_ppi = 72  # Matplotlib defines 1 pt to be 1/72 of an inch
cm_per_inch = 2.54
scaling_magic = 4/3

class Units(Enum):
    inch = 1
    cm = 2
    mm = 3
    pt = 4


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


class SVG:
    def __init__(self, file: Path):
        """
        Attempts to compute the correct scale factor for the SVG file.
        Currently only works for some cases, but it remains unclear why it fails when it does.
        :param file: Path to an SVG file to be inserted in the figure.
        """
        from svgutils.transform import fromfile
        self.file = file
        svg = fromfile(f"{file!s}")
        width, w_unit = self.get_width_height(svg, "width")
        height, h_unit = self.get_width_height(svg, "height")
        assert w_unit == h_unit, "Units of SVG drawing dimensions must match!"
        x0, y0, x1, y1 = self.get_view_box(svg)
        h_user, w_user = y1 - y0, x1 - x0
        pts_per_unit = h_user / height
        assert abs(pts_per_unit - w_user / width) < 1e-2, "Vertical scale is different than horizontal in SVG"
        self.scale = get_pts_per_unit(w_unit) / pts_per_unit
        self.svg = sc.SVG()
        self.svg.root = svg.getroot().root

    @staticmethod
    def get_width_height(svg, name) -> Tuple[float, Units]:
        dim: str = svg.root.get(name)
        for unit in ("cm", "in", "mm", "pt", ""):
            if unit in dim:
                val = dim.rstrip(unit)
                return float(val), SVG.to_unit(unit)

    @staticmethod
    def get_view_box(svg):
        vb = svg.root.get("viewBox")
        return (float(xy) for xy in vb.split(" "))

    @staticmethod
    def to_unit(unit: str) -> Units:
        if unit == "in":
            return Units.inch
        elif unit == "cm":
            return Units.cm
        elif unit == "mm":
            return Units.mm
        else:
            return Units.pt


class Subplot:
    def __init__(
            self,
            figure: Union[plt.Figure, SVG],
            location: Location,
            text: Optional[Union[Text, Tuple[Text, ...]]] = None,
            auto_label: bool = True,
            figure_offset: Location = Location(0, 0),
            scale: Optional[float] = None,
    ):
        self.fig: Union[plt.Figure, SVG] = figure
        self.location: Location = location
        self.text: Optional[Tuple[Text, ...]] = text if isinstance(text, tuple) or text is None else (text,)
        self.auto_label: bool = auto_label
        self.figure_offset: Location = figure_offset
        self.scale: float = scale


class SubplotFig(Subplot):
    def __init__(
            self,
            subplot_size: ElemSize,
            location: Location,
            text: Optional[Union[Text, Tuple[Text, ...]]] = None,
            gridspec_kwargs: Optional[Dict[str, Any]] = None,
            auto_label: bool = True,
            figure_offset: Location = Location(0, 0),
    ):
        gs_kwargs = gridspec_kwargs or dict()
        gs_kwargs.setdefault("nrows", 1)
        gs_kwargs.setdefault("ncols", 1)

        figure: plt.Figure = plt.figure(figsize=figure_size_in_inches(subplot_size))
        self.gridspec: GridSpec = figure.add_gridspec(**gs_kwargs)

        super().__init__(figure, location, text, auto_label, figure_offset)


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
    figure_size: ElemSize
    subplots: Subplots
    plot_grid_every: Union[float, int] = 0
    generate_image: ImageType = ImageType.none
    image_dpi: int = 400
    auto_label_options: AutoLabelOptions = AutoLabelOptions()


UserFigureSpec = TypeVar("UserFigureSpec")


def compositor(figure_spec: Type[UserFigureSpec], delete_png: bool=False):
    assert issubclass(figure_spec, FigureSpec), "The compositor needs the figure type: `@compositor(FigureSpec)`"
    def compositor_decorator(fn: Callable[..., Any]):
        def wrapped_fn(*args, **kwargs):
            fig_spec = figure_spec()
            result = fn(fig_spec, *args, **kwargs)
            composite(fig_spec, delete_png)
            return result
        return wrapped_fn
    return compositor_decorator


def composite(figure: FigureSpec, delete_png=False):
    import tempfile

    tempdir = Path(tempfile.gettempdir())
    panels = []
    if figure.plot_grid_every > 0:
        pts_per_unit = get_pts_per_unit(figure.figure_size.units)
        grid_every = figure.plot_grid_every * pts_per_unit
        panels.append(sc.Grid(grid_every, grid_every, size=0))

    label_n = 0
    auto_label = figure.auto_label_options._asdict()
    first_char = auto_label.pop("first_char")
    for name in figure.subplots._fields:
        subplot = getattr(figure.subplots, name)
        assert isinstance(subplot, Subplot)

        panel_elements = []

        assert isinstance(subplot.fig, (plt.Figure, SVG))
        figure_offset = location_to_str(figure.figure_size.units, subplot.figure_offset)
        if isinstance(subplot.fig, plt.Figure):
            fn = tempdir / f"{name}.svg"
            subplot.fig.savefig(fn, transparent=True)
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
            svg = sc.SVG(fn)
            if subplot.scale is None:
                svg.scale(scaling_magic)  # Default scaling as per block comment above
            else:  # Custom scaling per user request
                svg.scale(subplot.scale)
            panel_elements.append(svg.move(*figure_offset))
        else:
            scale = subplot.scale or subplot.fig.scale
            print(f"Scaling SVG {subplot.fig.file} by {scale}")
            panel_elements.append(subplot.fig.svg.scale(scale).move(*figure_offset))

        if subplot.text is not None:
            panel_elements += [
                sc.Text(
                    t.text,
                    *location_to_str(figure.figure_size.units, Location(t.x, t.y, subplot.location.units)),
                    **t.kwargs
                ).move(*figure_offset)
                for t in subplot.text]

        if subplot.auto_label:
            label = sc.Text(
                f"{chr(ord(first_char) + label_n)}",
                *location_to_str(figure.figure_size.units, Location(0, 0)),
                **auto_label
            )
            label_n += 1
            panel_elements.append(label)

        panel = sc.Panel(*panel_elements)
        location = location_to_str(figure.figure_size.units, subplot.location)
        panels.append(panel.move(*location))

    def dim2str(width_height: Union[float, int], units: Units):
        return f"{width_height:.2f}in" if units == Units.inch else f"{width_height:.2f}{units.name}"

    sc.Figure(dim2str(figure.figure_size.width, figure.figure_size.units),
              dim2str(figure.figure_size.height, figure.figure_size.units),
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
        run(f"inkscape --without-gui --export-png='{image_name}' --export-dpi {figure.image_dpi} {figure.svg_path}")
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
    elif units == Units.mm:
        pts_per_unit = plt_ppi / cm_per_inch / 10
    else:
        pts_per_unit = 1
    return scaling_magic * pts_per_unit


def figure_size_in_inches(fig_size: ElemSize) -> Tuple[float, ...]:
    # Convert the subplot_size to inches, the units used by matplotlib to specify figure size.
    if fig_size.units == Units.pt:
        return tuple(wh / plt_ppi for wh in (fig_size.width, fig_size.height))
    elif fig_size.units == Units.cm:
        return tuple(wh / cm_per_inch for wh in (fig_size.width, fig_size.height))
    elif fig_size.units == Units.mm:
        return tuple(wh / cm_per_inch * 10 for wh in (fig_size.width, fig_size.height))
    else:
        return tuple(wh for wh in (fig_size.width, fig_size.height))


def location_to_str(default_units: Units, loc: Location) -> Tuple[str, ...]:
    units = loc.units or default_units
    pts_per_unit = get_pts_per_unit(units)
    return tuple(f"{xy*pts_per_unit}" for xy in (loc.x, loc.y))


def check_svg_scale(svg: sc.SVG):
    """
    :param svg:
    :param target_scale:
    :return:
    """
    import re

    xform: str = svg.root.get("transform")
    print(f"Units:{svg.root.get('units')} WH: {svg.root.get('width')} {svg.root.get('height')} XFORM: {xform}")
    g = re.match(r".+?scale\(([-+]?\d*\.\d+|\d+)\)", xform or "")
    return float(g.group(1) if g else 1)


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
