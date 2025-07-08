#!/usr/bin/env python
""" generate random particles for rendering with blender script blend_powder.py """
# Brian DeCost -- Carnegie Mellon University -- 2016
# Modified by Nicole Tin -- Velexi Research -- 2025

import json
import typer
import numpy as np
from scipy.stats import weibull_min
from datetime import datetime
from typing import Literal

def fit_weibull_dist(loc=1, shape=0.1, nsamples=1e5):
    """ fit a weibull distribution to a given lognormal distribution """
    loc = np.log(loc)
    samples = np.random.lognormal(mean=loc, sigma=shape, size=nsamples)
    weibull_params = weibull_min.fit(samples)
    return weibull_min(*weibull_params)

def generate_sample(
    samplesize: int = typer.Option(
        10, 
        "-n", "--samplesize", 
        help="number of particles"
    ),
    textfile: str = typer.Option(
        "particles.json", 
        "-o", "--textfile", 
        help="json output path"
    ),
    distribution: str = typer.Option(
        "lognormal", 
        "-d", "--distribution", 
        help="distribution family to sample from",
        show_choices=True,
        case_sensitive=False,
        # restrict to choices:
        # Typer will automatically show choices if you use Choice as the type
    ),
    loc: float = typer.Option(
        0.1, 
        "-m", "--loc", 
        help="distribution scale parameter"
    ),
    shape: float = typer.Option(
        0.5, 
        "-s", "--shape", 
        help="distribution shape parameter"
    ),
):
    allowed = {"lognormal", "normal", "weibull_fit"}
    if distribution not in allowed:
        raise typer.BadParameter(f"Choose from: {', '.join(allowed)}")
    return _generate_sample(samplesize, textfile, distribution, loc, shape)

def _generate_sample(
    samplesize,
    textfile,
    distribution,
    loc,
    shape,
):
    """Sample particle sizes from a distribution and make a json file."""

    # sample particle sizes from a specified generating distribution
    if distribution == 'lognormal':
        loc_log = np.log(loc)
        size = np.random.lognormal(mean=loc_log, sigma=shape, size=samplesize)
    elif distribution == 'normal':
        size = np.random.normal(loc=loc, scale=shape, size=samplesize)
    elif distribution == 'weibull_fit':
        nsamples = int(1e5)
        dist = fit_weibull_dist(loc=loc, shape=shape, nsamples=nsamples)
        size = dist.rvs(size=samplesize)
    else:
        typer.echo('error: choose between normal, lognormal, and weibull_fit distributions')
        raise typer.Exit(code=1)

    # particle positions from uniform distribution
    xx = np.random.uniform(low=0, high=1, size=samplesize)
    yy = np.random.uniform(low=0, high=1, size=samplesize)
    zz = np.random.uniform(low=0, high=1, size=samplesize)

    # serialize everything to json for the blender script
    particles = []
    for s, x, y, z in zip(size, xx, yy, zz):
        particles.append({'size': float(s), 'x': float(x), 'y': float(y), 'z': float(z)})

    data = {
        'distribution': distribution,
        'loc': loc,
        'shape': shape,
        'timestamp': datetime.utcnow().isoformat(),
        'particles': particles
    }

    with open(textfile, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    typer.run(generate_sample)
