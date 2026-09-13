# TODO: Riley Polynomial Verification

I want to create some independent references for Riley's distortion models using PxInt2D/GridInt2D to create images of undistorted and distorted grids. I want you to setup a series of python scripts in ./riley-verif/ for this. 

The first step to do this will be to create an undistorted setup script with GridInt2D and Riley that produces exactly the same image of an eggbox grid. We should verify this with direct subtraction of the floating point images to within a given tolerance. 

You will need to create as set of python scripts using a def main(), if __name__ == "__main__": setup:
case0_gridint2d_ref_nodistort.py - Runs the render with gridint2d saving as raw .npy floats and a .tiff for visualisation. 
case0_riley_ref_nodistort.py - Runs the render with riley saving as above.
case0_verify_exact_nodistort.py - Loads the output renders from both cases and produces two matplotlib images: 1) using .npy, row of three images 1.1=grindint2d, 1.2=riley, 1.3=difference with max difference in the fig title, 2) as 1) but using 8 bit tiffs.
common_params.py - CONSTANTS we import into all cases like camera information (pixels etc), bit depth (set to 8), grid pitch, threads to use etc

We should use the following setup for now:
400x250 pixels,
Riley=pixel box ssaa with 32x32 samples (check RAM if over 20GB let me know)
GridInt2D= gauss integration with 4x4 samples per px 
Eggbox grid shader with 5 pixels/period - we can use world or uv coords 

Render output and analysis should go to sub-dirs in ./out/ 
- ./out/case0_gridint2d/
- ./out/case0_riley/
- ./out/case_verify_exact/

There is a uv venv in the .venv directory you can use to set this up. Follow our ./styelguides/PYTHONSTYLEGUIDE.md.

Riley and the associated demos can be found here: ~/riley-raster/ - start with the README to extract the python demos. There is also this directory for my recent paper doing exactly the same thing comparing Riley to GridInt2d for an eggbox expressed on an FE mesh.: ~/rcc-2d/ - start with the README.md to understand how this works.

