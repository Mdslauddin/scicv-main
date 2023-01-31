
"""
# Perform Morphological Operations
- imerode	Erode image
- imdilate	Dilate image
imopen	Morphologically open image
imclose	Morphologically close image
- imtophat	Top-hat filtering
- imbothat	Bottom-hat filtering
- imclearborder	Suppress light structures connected to image border
- imfill	Fill image regions and holes
- bwhitmiss	Binary hit-miss operation
- bwmorph	Morphological operations on binary images
- bwmorph3	Morphological operations on binary volume
- bwperim	Find perimeter of objects in binary image
- bwskel	Reduce all objects to lines in 2-D binary image or 3-D binary volume
- bwulterode	Ultimate erosion

# Perform Morphological Reconstruction
imreconstruct	Morphological reconstruction
imregionalmax	Regional maxima
imregionalmin	Regional minima
imextendedmax	Extended-maxima transform
imextendedmin	Extended-minima transform
imhmax	H-maxima transform
imhmin	H-minima transform
- imimposemin	Impose minima


Create Structuring Elements and Connectivity Arrays
strel	Morphological structuring element
offsetstrel	Morphological offset structuring element
conndef	Create connectivity array
iptcheckconn	Check validity of connectivity argument


Create and Use Lookup Tables
applylut	Neighborhood operations on binary images using lookup tables
bwlookup	Nonlinear filtering using lookup tables
makelut	Create lookup table for use with bwlookup


Pack Binary Images
bwpack	Pack binary image
bwunpack	Unpack binary image
"""
__all__ = ['imerod','imdilate','imopen','imclose',]