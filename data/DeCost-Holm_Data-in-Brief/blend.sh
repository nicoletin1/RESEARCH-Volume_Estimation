#!/bin/bash
# example shell script demonstrating powder image rendering
# Brian DeCost -- Carnegie Mellon University -- 2016

# I use slurm array jobs to render these images,
# and use SLURM_ARRAY_TASK_ID as a RENDER_ID
# GNU parallel is another nice solution
RENDER_ID=1

# note: running more than two (or so) blender jobs on a single socket
# will severely degrade performance 

# set powder render parameters: 
# distribution family, parameters, number of samples
DIST=lognormal
LOC=0.1
SHAPE=0.5
N=800

# create a directory for this distribution
CLASS=${DIST}-loc${LOC}-shape${SHAPE}
mkdir -p ${CLASS}

mkdir -p ${CLASS}/particles
PARTICLESPATH=${CLASS}/particles/particles${RENDER_ID}.json

mkdir -p ${CLASS}/renders
RENDERPATH=${CLASS}/renders/particles${RENDER_ID}.png

# sample ground truth particle sizes from the given distribution
python generate_sample.py -n ${N} --textfile ${PARTICLESPATH} --distribution ${DIST} --loc ${LOC} --shape ${SHAPE}

# render powder image using blender
# blender script takes 'arguments' as environment variables (limitation of blender's python interface)
PARTICLESPATH=${PARTICLESPATH} RENDERPATH=${RENDERPATH} blender -b --python blender_powder.py
