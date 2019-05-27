# Map Generator

A procedural 2D map generator using Perlin noise and Voronoi diagrams

## Installation

> pip install -r requirements.txt

### Windows users:
- Get the windows shapely .whl here → https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely
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

> python main.py --s ./images/voro.png --size 75 -f 20 -r 2 -g 3.15

[![Map](/images/voro.png)](https://github.com/AliShazly/map-generator/blob/master/images/voro.png)

## Limitations