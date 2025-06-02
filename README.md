# ElectroLoc

## Summary

ElectroLoc is a Python module that allows localizing and label electrode contacts in CT images. This module allows the following:
- Running the algorithm on a CT image to localize and labels its contacts.
- Saving the output to a CSV file or to a binary NiFTY mask.
- Plotting several components of the output separately (original image, electrode mask, predicted contacts, ...)
- Using a ground truth to evaluate the performance of the algorithm
- and more.

## Installation

1. Make sure to have [Python](https://www.python.org/downloads/) and [pip](https://pip.pypa.io/en/stable/installation/) available on your computer:
```sh
# Both of these commands should return the versions of your Python and pip
python --version
python -m pip -- version
```
If one these commands an error, then first install [Python](https://www.python.org/downloads/) on your computer. Normally, `pip` should be installed automatically during the installation of Python, and the commands above should work. Otherwise, this can be remedied by directly installing pip [here](https://pip.pypa.io/en/stable/installation/). Once Python and pip have both been successfully installed, the rest of this installation guide can be followed.

2. Download the source code as a `.zip` file and extract it to the desired directory, or clone this repository:
```sh
git clone https://github.com/Quent-DL/ElectroLoc.git
```

3. Navigate to the root directory of this repository:
```sh
cd /path/to/root/directory

# On Linux and Mac:
ls         # Should display ElectroLoc, .gitignore, LICENCE, README.md, and requirements.txt

# On Windows:
dir        # Should display ElectroLoc, .gitignore, LICENCE, README.md, and requirements.txt
```
4. Install the required libraries:
```sh
python -m pip install -r requirements.txt
```
Wait until all packages are downloaded. This may take a while.

5. The package is now ready to use ! You can try it by executing the following command from the root of the repository:
```sh
python -m ElectroLoc /path/to/CT/image.nii.gz  /path/to/brain/mask.nii.gz  /path/to/electrode/information.csv
```
ElectroLoc accepts several parameters for running the algorithm, saving its output or plotting results. More information about these parameters can be obtained with
```
python -m ElectroLoc --help
```

## Inputs and Outputs

### Inputs

The algorithm requires three mandatory inputs: a CT volume (format `.nii.gz`), a brain mask (`.nii.gz`), and information about the electrodes (`.csv`). The latter must strictly follow the following format:

------------------------------------
|nb_contacts|vox_x|vox_y|vox_z|
|-----------|-----|-----|-----|
|10|310|262|34|
|12|315|205|47|
| ... |
-------------------

Make sure that there are NO SPACES in the header (first line), even after commas.

The inputs are:
- `nb_contacts`: the number of contacts along that electrode;
- `vox_x,vox_y,vox_z`: the coordinates of the electrode's entry point in the CT image, expressed in voxel coordinates.
The order of the electrodes within the list does not matter.

### Outputs

#### Information file

With the parameter `-o`, the output of the algorithm can be saved into a CSV file with the following formats:

|vox_x|vox_y|vox_z|world_x|world_y|world_z|electrode_id|c_id|
|-----|-----|-----|-------|-------|-------|------------|----|
|294.048|262.64|92.612|-17.835|1.696|138.313|0|0|
|301.557|262.424|92.485|-21.355|1.595|138.234|0|1|
| ... |
-------------------------------------------------------------

The fields produced are:
- `vox_x,vox_y,vox_z`: the coordinates of the predicted contacts in the CT image;
- `world_x,world_y,world_z`: the coordinates of the predicted contacts in the physical space of the CT's image (using the affine matrix);
- `electrode_id`: the electrode identifier of the contact, so that all contacts in an electrode can easily be found;
- `c_id`: the contact identifier, *i.e.* the position of the contact along the electrode. The deepest contacts gets a `c_id` of 0, then the second deepest gets `c_id` 1, and so on.


#### Contact mask

The parameter `-m` allows saving the result of the predicted contacts as a binary NIfTI file. At the time of writing this README, this feature outputs non-symmetrical blocks of voxels around the contacts. This could be fixed in future works.


## Content of the project

The code is located in the package `ElectroLoc`, which is composed of the following files:
```sh
ElectroLoc/
|
|   # Backbone of the executable algorithm 
├── pipeline_wrapper.py       # Handles actions surrounding the algorithm such as retrieving the parameters, running the pipeline, saving, plotting, ...
├── pipeline.py               # The actual core of the algorithm, that launches the preprocessing, centroid extraction, linear modeling, and post-processing modules
|
|   # Modules used by the algorithm
├── centroids_extraction.py   # Extracts pointwise centroids from the CT's image along the electrodes
├── linear_modeling.py        # Fits linear models to the centroids
├── postprocessing.py         # Replaces linear models by curved ones, and generates equidistant contacts
|
|   # Complementary files
├── validation.py             # Launches batch validation to assess the performances of the model (*)
└── misc/                     # Contains various utilitary classes and functions used throughout the algorithm
    └── ...                   
```

*(\*) The input systme of the algorithm was updated recently (from hard-coded input file paths to terminal-based arguments). The file `validation.py` was not updated to this new system yet, and is currently unavailable for use.*


## License

MIT License

Copyright (c) 2025 Quentin De Laet

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
