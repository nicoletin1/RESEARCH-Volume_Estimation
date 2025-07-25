#!/usr/bin/env python
"""
Generate random 3D shapes (sphere, cube, cylinder, torus, distorted sphere, pebble, superquadric) for rendering or analysis.
Output is a JSON file with shape parameters and positions, similar to the spherical particle generator.
"""
import json
import numpy as np
import typer
from datetime import datetime
from typing import Literal
from scipy.stats import norm, lognorm, weibull_min, skewnorm

def generate_shape_sample(
    samplesize: int = typer.Option(
        10, "-n", "--samplesize", help="number of shapes"),
    textfile: str = typer.Option(
        "shapes.json", "-o", "--textfile", help="json output path"),
    shape: str = typer.Option(
        "sphere", "-t", "--shape", help="shape type", show_choices=True, case_sensitive=False),
    distribution: str = typer.Option(
        "random", "-d", "--distribution", help="distribution type (random, normal, lognormal, weibull, left-skew, multimodal)", show_choices=True, case_sensitive=False),
    mean: float = typer.Option(
        None, "-m", "--mean", help="mean value for shape size (optional, overrides default)")
):
    """
    CLI for generating random 3D shapes with specified distribution and mean.
    Args:
        samplesize (int): Number of shapes to generate.
        textfile (str): Output JSON file path.
        shape (str): Shape type to generate.
        distribution (str): Distribution type for shape size.
        mean (float): Mean value for shape size (optional).
    """
    allowed_shapes = {"sphere", "cube", "cylinder", "torus", "distorted_sphere", "pebble", "superquadric"}
    allowed_dists = {"random", "normal", "lognormal", "weibull", "left-skew", "multimodal"}
    if shape not in allowed_shapes:
        raise typer.BadParameter(f"Choose shape from: {', '.join(allowed_shapes)}")
    if distribution not in allowed_dists:
        raise typer.BadParameter(f"Choose distribution from: {', '.join(allowed_dists)}")
    return _generate_shape_sample(samplesize, textfile, shape, distribution, mean)

def _sample_size(distribution, mean, default_mean=1.0):
    """
    Sample a size parameter according to the specified distribution and mean.
    Args:
        distribution (str): Distribution type.
        mean (float): Mean value for size.
        default_mean (float): Default mean if not provided.
    Returns:
        float: Sampled size value.
    """
    if mean is None:
        mean = default_mean
    if distribution == "random":
        return np.random.uniform(0.5 * mean, 1.5 * mean)
    elif distribution == "normal":
        return np.abs(norm.rvs(loc=mean, scale=0.3 * mean))
    elif distribution == "lognormal":
        sigma = 0.3
        return lognorm(s=sigma, scale=np.exp(np.log(mean))).rvs()
    elif distribution == "weibull":
        c = 1.5
        return weibull_min(c, scale=mean).rvs()
    elif distribution == "left-skew":
        # Negative skew, more small values
        return np.abs(skewnorm.rvs(a=-5, loc=mean, scale=0.3 * mean))
    elif distribution == "multimodal":
        # Mixture of two normals
        if np.random.rand() < 0.5:
            return np.abs(norm.rvs(loc=mean, scale=0.2 * mean))
        else:
            return np.abs(norm.rvs(loc=2 * mean, scale=0.2 * mean))
    else:
        return np.random.uniform(0.5 * mean, 1.5 * mean)

def _generate_shape_sample(samplesize, textfile, shape, distribution, mean):
    """
    Generate a list of shapes with random positions and size parameters sampled from the specified distribution.
    Args:
        samplesize (int): Number of shapes to generate.
        textfile (str): Output JSON file path.
        shape (str): Shape type to generate.
        distribution (str): Distribution type for shape size.
        mean (float): Mean value for shape size (optional).
    """
    shapes = []
    for _ in range(samplesize):
        pos = 20 * (np.random.rand(3) - 0.5)  # Random position in [-10,10]
        if shape == "sphere":
            radius = _sample_size(distribution, mean, default_mean=1.0)
            shapes.append({"type": "sphere", "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "radius": float(radius)})
        elif shape == "cube":
            size = _sample_size(distribution, mean, default_mean=1.0)
            shapes.append({"type": "cube", "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "size": float(size)})
        elif shape == "cylinder":
            radius = _sample_size(distribution, mean, default_mean=0.7)
            height = _sample_size(distribution, mean, default_mean=1.0)
            shapes.append({"type": "cylinder", "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "radius": float(radius), "height": float(height)})
        elif shape == "torus":
            R = _sample_size(distribution, mean, default_mean=1.0)
            r = _sample_size(distribution, mean, default_mean=0.5)
            shapes.append({"type": "torus", "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "R": float(R), "r": float(r)})
        elif shape == "distorted_sphere":
            base_radius = _sample_size(distribution, mean, default_mean=1.0)
            distortion = np.random.uniform(0.1, 0.4)
            shapes.append({"type": "distorted_sphere", "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "base_radius": float(base_radius), "distortion": float(distortion)})
        elif shape == "pebble":
            a = _sample_size(distribution, mean, default_mean=1.2)
            b = _sample_size(distribution, mean, default_mean=0.8)
            c = _sample_size(distribution, mean, default_mean=0.6)
            shapes.append({"type": "pebble", "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "a": float(a), "b": float(b), "c": float(c)})
        elif shape == "superquadric":
            a1 = _sample_size(distribution, mean, default_mean=1.0)
            a2 = _sample_size(distribution, mean, default_mean=1.0)
            a3 = _sample_size(distribution, mean, default_mean=1.0)
            eps1 = np.random.uniform(0.3, 1.3)
            eps2 = np.random.uniform(0.3, 1.3)
            shapes.append({"type": "superquadric", "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "a1": float(a1), "a2": float(a2), "a3": float(a3), "eps1": float(eps1), "eps2": float(eps2)})
    data = {
        "shape_type": shape,
        "distribution": distribution,
        "mean": mean,
        "timestamp": datetime.utcnow().isoformat(),
        "shapes": shapes
    }
    with open(textfile, "w") as f:
        json.dump(data, f)

if __name__ == "__main__":
    # Typer CLI wrapper
    typer.run(generate_shape_sample)
