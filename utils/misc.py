import numpy as np
import csv
from config import DefaultConfig
import os
from pathlib import Path

bb = DefaultConfig.patch_size    #128 for nucleus

#Returns a clickMap(m,n), and boundingBoxes
#cx, cy: lists of x and y coordinates
#m, n: height and width of the full image
def get_clickmap_boundingbox(cx, cy, m, n):
    clickMap = np.zeros((m, n), dtype=np.uint8)

    # Removing points out of image dimension (these points may have been clicked unwanted)
    x_del_indices = set([i for i in range(len(cx)) if cx[i]>=n or cx[i]<0])
    y_del_indices = set([i for i in range(len(cy)) if cy[i]>=m or cy[i]<0])
    del_indices = list(x_del_indices.union(y_del_indices))
    cx = np.delete(cx, del_indices)
    cy = np.delete(cy, del_indices)

    clickMap[cy, cx] = 1
    boundingBoxes = []
    for i in range(len(cx)):
        xStart = cx[i] - bb // 2
        yStart = cy[i] - bb // 2
        if xStart < 0:
            xStart = 0
        if yStart < 0:
            yStart = 0
        xEnd = xStart + bb - 1
        yEnd = yStart + bb - 1
        if xEnd > n - 1:
            xEnd = n - 1
            xStart = xEnd - bb + 1
        if yEnd > m - 1:
            yEnd = m - 1
            yStart = yEnd - bb + 1
        boundingBoxes.append([xStart, yStart, xEnd, yEnd])
    return clickMap, boundingBoxes


#Returns a list of x coordinates and a list of y coordinates from the given CSV file
def get_coords_from_csv(filename):
    #Open file
    file = open(filename)
    csvReader = csv.reader(file)

    clicks_x = []
    clicks_y = []

    for row in csvReader:
        #If the line does not have exactly two values:
        if len(row) != 2:
            raise ValueError(f"The CSV file: '{filename}' does not have valid entries.")

        #Add x and y
        v1 = row[0]
        v2= row[1]
        x = int(v1)
        y = int(v2)
        clicks_x.append(x)
        clicks_y.append(y)

    file.close()

    #If the file is empty:
    if len(clicks_x) == 0:
        raise ValueError(f"The CSV file '{filename}' is empty")

    return clicks_x, clicks_y


#Generate an output name
def get_output_filename(fn, outDir):
    if outDir is None:
        split = os.path.splitext(fn)
        return f'{split[0]}_NuClick.png'
    else:
        path = Path(fn)
        filename = path.stem
        return f'{outDir}/{filename}_NuClick.png'


#Generate a list of tuples (path to image, path to points) from args
def get_images_points(args):
    images = []
    points = []
    images_points = []

    #If image names are provided:
    if (args.imgdir is None):
        images = args.image
    #Else if a directory is provided:
    else:
        for fn in os.listdir(args.imgdir):
            if os.path.splitext(fn)[1] in {'.png', '.jpg', '.bmp', '.tiff'}:
                images.append(f'{args.imgdir}/{fn}')
        images.sort()
    
    #If csv files are provided:
    if (args.pntdir is None):
        points = args.points
    #Else if a directory is provided:
    else:
        for fn in os.listdir(args.pntdir):
            if os.path.splitext(fn)[1] == '.csv':
                points.append(f'{args.pntdir}/{fn}')
        points.sort()

    
    if (len(images) != len(points)):
        raise ValueError("Images and points do not match")

    for i in range(len(images)):
        images_points.append((images[i], points[i]))

    return images_points

    