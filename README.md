# Map Generator

A procedural 2D map generator using Perlin noise and Voronoi diagrams

## Installation

`pip install -r requirements.txt`

### Windows users:
- Get the windows shapely .whl here:  https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
- Download and pip install the one for your version of python
- You should now be good to install the rest of the requirements above

## Usage
```
usage: main.py [-s : str] [--size : int] [-f : int] [-r : int]
               [-g : flt]

arguments:
  -h, --help .......... show this help message and exit
  -s, --save .......... Path to save output image to
  --size .............. Size of voronoi diagram. Impacts generation speed
                        greatly
  -f, --frequency ..... Frequency of perlin noise, lower is more frequent
  -r, --relaxation .... Iterations of the lloyd's relaxation algorithm
  -g, --gradient ...... Strength of the gaussian gradient. Higher makes the
                        ocean larger
```

## Examples

#### Using default parameters:
`python main.py -s ./images/voro.png`

[![Map](/images/voro.png)](https://github.com/AliShazly/map-generator/blob/master/images/voro.png)

## Limitations

- Very slow map generation. Can take up to a minute to generate maps with larger sizes
- No good way to include features that don't include modifying polygons due to the way the map is represented
- Elevation does not produce interesting results, since I based it off of the distance from the nearest coast.
- Lower map sizes don't handle elevation very well.
