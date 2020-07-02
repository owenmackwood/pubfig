# pubfig
A simple module for making publication quality figures with Matplotlib and svgutils

## Why pubfig?

In some respects, this module doesn't do anything that can't already be done with [Matplotlib](https://matplotlib.org/) and [svgutils](https://svgutils.readthedocs.io).
So what's the point? While svgutils is an excellent package, and enables you to programmatically create complete publication-ready figures, it has some rough edges.
These include it's apparent inability to correctly apply units to certain transformations, so when specifying the location of a subplot, one needs to work in points. 
Furthermore, Matplotlib has its own peculiarities. In particular, figure sizes are specified in inches, it has a fixed point density of 72 points per inch, and when saving figures as SVG it sets their scale to 0.75, making their size incorrect when using svgutils to composite a final figure.

So why pubfig? It smooths out those rough edges, permitting you exploit the capabilities of svgutils, while not having to worry about units, or the peculiarities of Matplotlib.
It also features the ability to specify an entire publication-ready figure in a single location, and automates away all of the details of compositing with svgutils.
One final nicety is that it can also automatically output high-DPI raster images in either PNG (requires Inkscape) or TIFF formats  (requires Inkscape and ImageMagick).
