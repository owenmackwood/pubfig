"""
pubfig
=====

A simple module for creating publication-quality figures, and recreating them at the press of a button.

See README.md for a quick-start guide.

Important classes
---------------------
FigureSpec
    This is the baseclass for specifying figures.
Panel
    Used to specify a saved image that should be placed
    somewhere in the composited figure.
PanelFig
    Used to place a Matplotlib figure, of a specified
    size, somewhere in the composited figure.

Important functions
---------------------
compositor
    A decorator for functions that can generate a figure from a FigureSpec.
    This is recommended way to produce figures.
composite
    A function that takes a FigureSpec object and composites the figure.
    This function is not needed if the compositor decorator is used.

Utilities
---------------------
spines_frames
    A very simple function that hides axis spines for a Matplotlib plot
"""
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Tuple, Union, Dict, Any, Optional, Callable, Type, Generator
from typing_extensions import Protocol
from enum import Enum
import matplotlib.pyplot as plt
import svgutils.compose as sc
from svgutils.transform import SVGFigure
from matplotlib.gridspec import GridSpec

Length = Union[float, int]
PanelsSpec = NamedTuple


class Units(Enum):
    """
    An enumeration used to specify the units of a figure or its elements.
    Also provides methods to convert between units, though the user should
    not need to use them.
    """
    inch = 1  # Can't use `in` because it's a reserved word
    cm = 2
    mm = 3
    pt = 4
    pc = 5
    px = 6

    def __str__(self) -> str:
        """Change `inch` to `in` for compatibility with SVG/CSS"""
        return "in" if self == Units.inch else self.name

    @staticmethod
    def to_units(unit: str) -> "Units":
        """Adapt SVG/CSS unit of `in` to Units.inch"""
        return Units.inch if unit == "in" else getattr(Units, unit)

    def to_pts(self, length: Length) -> Length:
        if self == Units.inch:
            pts = length * _c.pt_per_in
        elif self == Units.cm:
            pts = length * _c.pt_per_in / _c.cm_per_in
        elif self == Units.mm:
            pts = length * _c.pt_per_in / (_c.cm_per_in * 10)
        elif self == Units.pc:
            pts = length * _c.pt_per_pc
        elif self == Units.px:
            pts = length / _c.px_per_pt
        else:  # Units.pt
            pts = length
        return pts

    def to_inches(self, length: Length) -> Length:
        return self.to_pts(length) / _c.pt_per_in

    def to_px(self, length: Length) -> Length:
        return _c.px_per_pt * self.to_pts(length)


class ElemSize(NamedTuple):
    """Used to specify the size of the figure or its elements in physical units."""
    width: Length
    height: Length
    units: Units


class Location(NamedTuple):
    """Used to specify location of panels or text within the figure."""
    x: Length
    y: Length
    units: Optional[Units] = None  # None if units from associated ElemSize should be used


class ImageType(Enum):
    """
    Used as part of a FigureSpec to indicate which type of raster image to output, if any.
    Regardless of which is chosen, an SVG file will be saved to disk.

    `none` : Only the SVG will be produced.
    `tiff` : A TIFF image will be output. If the `delete_png` argument passed to the compositor
        decorator (or composite function) is False, then the intermediate PNG will be left on disc.
    `png` : A PNG image will be output.
    """
    none = 1
    tiff = 2
    png = 3


class Text:
    """
    A class to define text objects, associated with a panel.
    The `x` and `y` arguments are relative to the panel location,
    and use the same units as the ElemSize that defines the panel.

    The following SVG ``<text>`` tag attributes can also be set:

    size : int
        The "font-size" attribute (default value: 8)
    font : str
        The "font-family" attribute (default value: "Verdana")
    weight : str : {"normal", "bold", "bolder", "lighter", "<number>"}
        The "font-weight" attribute. Can also be a string containing a
         number in the range 100-900, e.g. "900" (default value: "normal")
    letterspacing : int
        The "letter-spacing" attribute (default value: 0)
    anchor : str
        The "text-anchor" attribute (default value: "start")
    color : str
        The "fill" attribute (default value: "black")
    """
    def __init__(
            self,
            text: str,
            x: Length, y: Length,
            size: int = 8,
            font: str = "Verdana",
            weight: str = "normal",
            letterspacing: int = 0,
            anchor: str = "start",
            color: str = "black"
    ):
        self.text: str = text
        self.x: Length = x
        self.y: Length = y
        self.kwargs: Dict[str, Any] = dict(
            size=size, font=font, weight=weight, letterspacing=letterspacing, anchor=anchor, color=color
        )


class VectorImage:
    """
    A class that attempts to compute the correct scale factor for SVG/EPS/PDF files,
    since svgutils throws away the necessary information when its SVG class
    is used to load the file directly.

    This will fail if the drawing width/height are without units, or the
    viewBox has units.
    """

    def __init__(self, file: Union[Path, str]):
        from svgutils.transform import fromfile

        self.file: Path = Path(file)

        if self.file.suffix in (".eps", ".pdf"):
            self._convert_to_svg()

        svg = fromfile(f"{self.file!s}")

        doc_width, w_unit = self.get_width_height(svg, "width")
        doc_height, h_unit = self.get_width_height(svg, "height")
        assert w_unit == h_unit, "Units of SVG drawing dimensions must match!"

        min_x, min_y, vb_width, vb_height = self.get_view_box(svg)
        assert abs(min_x) < 1e-3 and abs(min_y) < 1e-3, \
            "The min-x/y of the SVG viewBox is non-zero, comment out this line to try loading it anyway"

        user_units_scale = vb_height / doc_height
        assert abs(user_units_scale - vb_width / doc_width) < 1e-2, "Vertical scale is different than horizontal in SVG"

        xform_scale = self.get_svg_scale(svg)

        self.scale = xform_scale * w_unit.to_px(1) / user_units_scale
        self.svg = sc.SVG()
        self.svg.root = svg.getroot().root

    def _convert_to_svg(self):
        import tempfile

        eps_path = self.file
        file_name = Path(tempfile.gettempdir()) / eps_path.name[:-4]
        temp_svg = file_name.with_suffix(".svg")

        print(f"Converting {eps_path} to SVG for compositing.")
        _run(f"inkscape --without-gui --export-plain-svg='{temp_svg}' {eps_path}")

        self.file = temp_svg

    @staticmethod
    def get_width_height(svg: SVGFigure, width_or_height) -> Tuple[float, Units]:
        dim: str = svg.root.get(width_or_height)
        for unit in (str(n) for n in Units):
            if unit in dim:
                val = dim.rstrip(unit)
                return float(val), Units.to_units(unit)
        return float(dim), Units.px

    @staticmethod
    def get_view_box(svg: SVGFigure) -> Tuple[float, ...]:
        vb = svg.root.get("viewBox")
        return tuple(float(xy) for xy in vb.split(" "))

    def get_svg_scale(self, svg: SVGFigure) -> float:
        import re

        scale = 1.
        xform: str = svg.root.get("transform")
        if xform is not None:
            print(f"{self.file} root transform: {xform}")
            g = re.match(r".+?scale\(([-+]?\d*\.\d+|\d+)\)", xform)
            if g:
                scale = float(g.group(1))
                if abs(scale - 1.) > 1e-3:
                    print(f"Found global scaling factor in {self.file.absolute()} of {scale}")

        return scale


class RasterImage:
    def __init__(self, file: Union[Path, str], img_size: ElemSize):
        self.file: Path = Path(file)
        self.img_size: ElemSize = img_size


class Panel:
    """
    Figures are comprised of Panel objects. Each panel contains:

    fig : plt.Figure or VectorImage or RasterImage
        A reference to the content to be displayed in the panel. This can be
        either a Matplotlib Figure, or an image loaded from disk.

    location : Location
        The location in the figure where to place the upper-left corner of the panel.

    text : Text or Tuple[Text, ...], optional
        A set of text objects to include in the panel. The position of the text
        objects is relative to the upper-left corner of the panel.

    auto_label : bool
        Whether to label this panel automatically. Panels are labelled according
        to the order in which they are defined in the FigureSpec. The type of
        label is determined by the auto_label_options attribute of the FigureSpec.
        The label will always be located at `location`.

    content_offset : Location
        Where to position the content referred to by the `fig` attribute, relative
        to the upper-left corner of the panel. If no units are set in this Location
        object, then units are taken from `location`. If `location` has no units
        then the FigureSpec.fig_size units are used.

    scale : float, optional
        How much to scale the content. This is generally unnecessary, though
        there may be special circumstances where it is appropriate, e.g. when
        the content is an image with inappropriate physical dimensions.
    """
    def __init__(
            self,
            figure: Union[plt.Figure, VectorImage, RasterImage],
            location: Location,
            text: Optional[Union[Text, Tuple[Text, ...]]] = None,
            auto_label: bool = True,
            content_offset: Location = Location(0, 0),
            scale: Optional[float] = None,
    ):
        self.fig: Union[plt.Figure, VectorImage, RasterImage] = figure
        self.location: Location = location
        self.text: Optional[Tuple[Text, ...]] = text if isinstance(text, tuple) or text is None else (text,)
        self.auto_label: bool = auto_label
        self.content_offset: Location = content_offset
        self.scale: float = scale


class PanelFig(Panel):
    """
    Panel objects for which a Matplotlib Figure should be automatically constructed.
    Once the plt.Figure is constructed, a GridSpec is added to it, using the
    provided arguments. If `nrows` or `ncols` are not provided, they are set to 1.
    For an explanation of the remaining PanelFig attributes, see the class Panel.

    plt_fig_size : ElemSize
        The size of the Matplotlib Figure, i.e. the `figsize` for the constructor.
        The units of this ElemSize are used for its construction, but never again.

    gridspec_kwargs : Dict[str, Any], optional
        Any valid argument for Figure.add_gridspec(), including `nrows` and `ncols`.
    """
    def __init__(
            self,
            plt_fig_size: ElemSize,
            location: Location,
            text: Optional[Union[Text, Tuple[Text, ...]]] = None,
            auto_label: bool = True,
            content_offset: Location = Location(0, 0),
            gridspec_kwargs: Optional[Dict[str, Any]] = None,
    ):
        gs_kwargs = gridspec_kwargs or dict()
        gs_kwargs.setdefault("nrows", 1)
        gs_kwargs.setdefault("ncols", 1)

        figure: plt.Figure = plt.figure(
            figsize=tuple(plt_fig_size.units.to_inches(wh) for wh in (plt_fig_size.width, plt_fig_size.height))
        )
        self.gridspec: GridSpec = figure.add_gridspec(**gs_kwargs)

        super().__init__(figure, location, text, auto_label, content_offset)


def _generate_labels(first_char: str) -> str:
    """A generator that yields a series of labels"""
    label_n = 0
    if first_char.lower() == 'i':
        lower = first_char == 'i'
        while True:
            yield _int_to_roman(label_n + 1, lower)
            label_n += 1
    else:
        while True:
            yield f"{chr(ord(first_char) + label_n)}"
            label_n += 1


class AutoLabelOptions(NamedTuple):
    """
    Options to configure how automatic labels appear, and the
    particular sequence that is generated.

    first_char : Text
        The first character in the sequence of panel labels.
        Suggested values are {'a', 'A', 'i', 'I'}. If set to 'i'
        or 'I', Roman numeral labels will be generated. The `x` and `y`
        attributes of the Text object are ignored. Otherwise, see the
        Text class for details of the configurable text attributes.
    label_generator : Generator yielding strings
        This can be used to produce arbitrary label sequences,
        and should be callable with a single argument (the `first_char`).
    """
    first_char: Text = Text("a", 0, 0, size=12, weight="bold")
    label_generator: Generator[str, None, None] = _generate_labels


class FigureSpec(SimpleNamespace):
    """
    Class for specifying the contents and layout of a figure.

    output_file : Path or str
        The location and filename for the SVG file to be composited.
    figure_size: ElemSize
        Physical dimension of the figure.
    panels: Panels
        A named tuple of Panel objects, each of which specifies
        an individually positioned piece of figure content.
    plot_grid_every: Length
        If non-zero a grid will be generated every `plot_grid_every`.
        Is take to be in the same units as the `figure_size`.
    generate_image: ImageType
        Whether to render a raster image (PNG or TIFF) from the SVG.
    image_dpi: int
        If `generate_image` is not `none`, this in conjunction with
        `fig_size` will determine the resolution of the raster image.
    auto_label_options: AutoLabelOptions
        Determines the appearance of automated panel labels. Whether to
        automatically label panels is set on individual Panel objects.
    """
    class Panels(PanelsSpec):
        """
        Just a placeholder for the user definition of Panels,
        so that we can specify that FigureSpec has type
        panels: NamedTuple[Panel, ...]
        which doesn't work due to a TypeError.
        """
        panel: Union[Panel, PanelFig]

    output_file: Union[Path, str]
    figure_size: ElemSize
    panels: Panels
    plot_grid_every: Length = 0
    generate_image: ImageType = ImageType.none
    image_dpi: int = 400
    auto_label_options: AutoLabelOptions = AutoLabelOptions()


class PlottingFunction(Protocol):
    """
    Used for the type hints in pubfig.compositor, so that it knows decorated
    plotting functions take at least one argument of type FigureSpec.
    """
    def __call__(self, fig_spec: FigureSpec, *args, **kwargs) -> Any: ...


PlottingFunctionDecorator = Callable[[PlottingFunction], Callable[..., Any]]


def compositor(
        figure_spec: Type[FigureSpec],
        memoize_panels: bool = False,
        recompute_panels: bool = True,
        delete_png: bool = True,
) -> PlottingFunctionDecorator:
    """
    Returns a decorator for functions that operate on user-defined FigureSpec objects.

    Suppose the user-defined FigureSpec is called UserFigureSpec. The returned decorator
    will automatically construct an instance of the UserFigureSpec class and pass it as
    the first argument to the decorated function. When that function returns, the
    UserFigureSpec instance is passed to the `composite` function where the final
    SVG figure is generated.

    Usage:
    ```
    @compositor(UserFigureSpec)
    def plot_user_figure(figure: UserFigureSpec):
        ...  # Plotting code here
    ```

    Parameters
    ----------
    figure_spec : Type[FigureSpec]
        A reference to the user defined subclass of FigureSpec.
    memoize_panels : bool
        If true, the contents of each PanelFig are saved to disk, and in the future
        will be loaded from there instead of calling the decorated plotting
        function (until `recompute_panels` is true, the memoized data is deleted,
        or this argument is set to false).
    recompute_panels : bool
        Overrides the `memoize_panels` argument, causing the decorated plotting function
        to be called. If `memoize_panels` is true, the memoized plot data is overwritten
        with the new plot data.
    delete_png : bool
        A boolean indicating whether to delete the generated PNG file. Can be useful when
        the desired output is a TIFF (the PNG is a necessary intermediate file).

    Returns
    -------
    Another function is returned, that decorates the user function, having captured
    the type of FigureSpec to be instantiated and then composited.
    """
    assert issubclass(figure_spec, FigureSpec), \
        "The compositor needs the user-defined figure type: `@compositor(FigureSpec)`"

    def compositor_decorator(fn: PlottingFunction) -> Callable[..., Any]:

        def wrapped_fn(*args, **kwargs) -> Any:
            fig_spec = figure_spec()
            compute_panels = not memoize_panels or recompute_panels or _memoized_panels_missing(fig_spec)
            if compute_panels:
                print("Computing new panel contents.")
                result = fn(fig_spec, *args, **kwargs)
            else:
                print("Loading all panel contents from disk.")
                result = None
            composite(fig_spec, memoize_panels, compute_panels, delete_png)
            return result

        return wrapped_fn

    return compositor_decorator


def composite(
        fig_spec: FigureSpec,
        memoize_panels: bool = False,
        recompute_panels: bool = True,
        delete_png: bool = True,
) -> None:
    """
    Function that composites a figure from a FigureSpec.

    Parameters
    ----------
    fig_spec : FigureSpec
    memoize_panels : bool
    recompute_panels : bool
    delete_png :  bool
        See the pubfig.compositor decorator for a description of the parameters.

    Returns
    -------
        None
    """
    import tempfile

    svg_path = fig_spec.output_file
    if isinstance(svg_path, str):
        svg_path = Path(svg_path)

    assert not svg_path.is_dir(), "The output file name you provided is a directory"

    if svg_path.suffix != ".svg":
        svg_path = svg_path.with_suffix(".svg")

    svg_path = svg_path.expanduser()

    if not svg_path.parent.exists():
        svg_path.parent.mkdir(parents=True, exist_ok=True)

    if memoize_panels:
        panels_path = svg_path.parent / ".panels"
        if not panels_path.exists():
            panels_path.mkdir()
    else:
        panels_path = Path(tempfile.gettempdir())

    panels = []
    if fig_spec.plot_grid_every > 0:
        panels.append(_generate_grid(fig_spec.figure_size, fig_spec.plot_grid_every, font_size=8))

    auto_label = fig_spec.auto_label_options

    label_generator = auto_label.label_generator(auto_label.first_char.text)

    for name in fig_spec.panels._fields:
        panel = getattr(fig_spec.panels, name)
        assert isinstance(panel, Panel)

        panel_elements = []

        assert isinstance(panel.fig, (plt.Figure, VectorImage, RasterImage))
        content_offset = _location_to_str(
            panel.location.units or fig_spec.figure_size.units, panel.content_offset
        )
        if isinstance(panel, PanelFig):
            svg = _get_panel_content(panels_path, panel, name, memoize_panels, recompute_panels)
            panel_elements.append(svg.move(*content_offset))
        elif isinstance(panel.fig, VectorImage):
            scale = panel.scale or panel.fig.scale
            print(f"Scaling vector image {panel.fig.file.absolute()} by {scale:.3f}")
            panel_elements.append(panel.fig.svg.scale(scale).move(*content_offset))
        elif isinstance(panel.fig, RasterImage):
            img_size = panel.fig.img_size
            scale = panel.scale or 1.
            img = sc.Image(
                img_size.units.to_px(img_size.width),
                img_size.units.to_px(img_size.height),
                f"{panel.fig.file}",
            )
            panel_elements.append(img.scale(scale).move(*content_offset))
        else:
            raise TypeError(f"Unknown type of panel content {type(panel.fig)}")

        if panel.text is not None:
            panel_elements += [
                sc.Text(
                    t.text,
                    *_location_to_str(fig_spec.figure_size.units, Location(t.x, t.y, panel.location.units)),
                    **t.kwargs
                ).move(*content_offset)
                for t in panel.text]

        if panel.auto_label:
            label = sc.Text(
                next(label_generator),
                *_location_to_str(fig_spec.figure_size.units, Location(0, 0)),
                **auto_label.first_char.kwargs
            )
            panel_elements.append(label)

        location = _location_to_str(fig_spec.figure_size.units, panel.location)
        panels.append(sc.Panel(*panel_elements).move(*location))

    fs = fig_spec.figure_size
    sc.Figure(
        f"{fs.units.to_px(fs.width):.2f}px",
        f"{fs.units.to_px(fs.height):.2f}px",
        *panels
    ).save(svg_path)

    if fig_spec.generate_image != ImageType.none:
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
        basename = f"{svg_path}"[:-4]
        image_name = f"{basename}.png"
        _run(f"inkscape --without-gui --export-png='{image_name}' --export-dpi {fig_spec.image_dpi} {svg_path}")
        if fig_spec.generate_image == ImageType.tiff:
            tiff_name = f"{basename}.tiff"
            _run(f"convert -compress LZW -alpha remove {image_name} {tiff_name}")
            _run(f"mogrify -alpha off {tiff_name}")
            if delete_png:
                _run(f"rm {image_name}")
            image_name = tiff_name
        _run(f"eog {image_name}")


def _get_panel_content(
        panels_path: Path, panel: PanelFig, panel_name: str, memoize_panels: bool, recompute_panels: bool
) -> sc.SVG:
    """
    Obtains the panel content, either from the plt.Figure or from
    the disk, as appropriate.

    Parameters
    ----------
    panels_path : Path
    panel : PanelFig
    panel_name : str
    memoize_panels : bool
    recompute_panels : bool

    Returns
    -------
        sc.SVG
    """
    fn = panels_path / f"{panel_name}.svg"

    if recompute_panels:
        print(f"Saving {fn}")
        panel.fig.savefig(fn, transparent=True)

    svg = sc.SVG(f"{fn!s}")

    if not memoize_panels:
        print(f"Removing {fn}")
        fn.unlink()
        assert not fn.exists()

    if panel.scale is None:
        """
        Default scaling to recover correct size in the final image.
        This is necessary because regardless of the units to define the output svgutils.Figure,
        svgutils seems to treat all SVG elements as being defined in `px`
        """
        svg.scale(_c.px_per_pt)
    else:
        # Custom scaling per user request
        svg.scale(panel.scale)
    return svg


def _memoized_panels_missing(fig_spec: FigureSpec) -> bool:
    """
    Check if any panel Figure files are missing on disk.

    Parameters
    ----------
    fig_spec : FigureSpec

    Returns
    -------
        bool
    """
    panels_path = fig_spec.output_file.parent / ".panels"
    return not all((panels_path / name).with_suffix(".svg").exists()
                   for name in fig_spec.panels._fields if isinstance(getattr(fig_spec.panels, name), PanelFig)
                   )


def _location_to_str(default_units: Units, loc: Location) -> Tuple[str, ...]:
    """Converts locations to `px` for svgutils compatibility."""
    units = loc.units or default_units
    return tuple(f"{units.to_px(xy)}" for xy in (loc.x, loc.y))


def _generate_grid(
        size: ElemSize, dxy: Length, width: float = 0.5, font_size: int = 8
) -> sc.Element:
    """
    Adapted from svgutils.compose.Grid
    
    Fills a rectangle with horizontal and vertical grid lines, spaced every `dxy` in both directions.

    Parameters
    ----------
    size : ElemSize
        The size of the rectangle to fill with grid lines
    dxy : Length
        The spacing of the grid lines, in units of size.unit
    width : float
        Line with of the grid lines
    font_size : int
        The size of the numbers marking each grid line. Set to zero for no labels.

    Returns
    -------
    svgutils.compose.Element
        The grid lines and labels
    """
    from svgutils.transform import LineElement, TextElement, GroupElement
    x, y = dxy, dxy
    lines = []
    txt = []
    units = size.units
    d_px = font_size
    width_px, height_px = units.to_px(size.width), units.to_px(size.height)
    while x <= size.width:
        x_px = units.to_px(x)
        lines.append(LineElement([(x_px, 0), (x_px, height_px)], width=width))
        txt.append(TextElement(x_px+d_px/2, d_px, str(x), size=font_size))
        x += dxy
    while y <= size.height:
        y_px = units.to_px(y)
        lines.append(LineElement([(0, y_px), (width_px, y_px)], width=width))
        txt.append(TextElement(d_px/2, y_px+d_px, str(y), size=font_size))
        y += dxy

    return sc.Element(GroupElement(txt+lines).root)


def _run(command: str, check: bool = True, shell: bool = True) -> None:
    """
    Helper function to run shell commands.

    Parameters
    ----------
    command : str
        The command to be run
    check : bool
    shell : bool
        See the documentation of subprocess.run for a description

    Returns
    -------
        None
    """
    import subprocess as sp
    try:
        cp = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE, check=check, shell=shell)
        print(cp.stdout.decode())
    except sp.CalledProcessError as e:
        print(f"Returned error (exit status {e.returncode}:\n {e.stderr.decode()}")


def _int_to_roman(value: int, lower=False):
    """
    Convert an integer to a Roman numeral.
    Modified from: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s24.html
    """
    if not 0 < value < 4000:
        raise ValueError("Argument must be between 1 and 3999")
    ints = (1000, 900,  500, 400, 100,  90, 50,  40, 10,  9,   5,  4,   1)
    nums = ('M',  'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I')
    result = []
    for i in range(len(ints)):
        count = int(value / ints[i])
        result.append(nums[i] * count)
        value -= ints[i] * count
    result = ''.join(result)
    return result.lower() if lower else result


class _Conversions(NamedTuple):
    """
    https://www.w3.org/TR/2011/REC-CSS2-20110607/syndata.html#length-units

    The SVG/CSS standards define the following units:

        in: inches — 1in is equal to 2.54cm.
        cm: centimeters
        mm: millimeters
        pt: points — the points used by CSS are equal to 1/72nd of 1in.
        pc: picas — 1pc is equal to 12pt.
        px: pixel units — 1px is equal to 0.75pt.

    When plotting to an SVG file, Matplotlib defines the user-units to be `pt`.
    On the other hand it seems that svgutils needs things specified in `px`.

    Useful references:
    https://www.w3.org/TR/SVG2/coords.html
    https://wiki.inkscape.org/wiki/index.php?title=Units_In_Inkscape
    """
    pt_per_in = 72
    px_per_in = 96
    cm_per_in = 2.54
    px_per_pt = 4 / 3
    pt_per_pc = 12
    pc_per_in = pt_per_in // pt_per_pc


_c = _Conversions()


def spines_frames(
        ax: plt.Axes, left=True, bottom=True, top=False, right=False, show_frames=True
) -> None:
    """
    Helper function to hide axis spines.

    Parameters
    ----------
    ax : plt.Axes
        The axis on which to show or hide spines
    left : bool
    bottom : bool
    top : bool
    right : bool
        Whether to show the corresponding spine
    show_frames : bool
        Whether to show the background patch in the axis
    Returns
    -------
        None
    """
    ax.patch.set_visible(show_frames)
    kw = {'top': top, 'right': right, 'left': left, 'bottom': bottom}
    for k, v in kw.items():
        ax.spines[k].set_visible(v)
