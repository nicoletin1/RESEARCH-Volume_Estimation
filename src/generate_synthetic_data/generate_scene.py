#!/usr/bin/env python
"""
Generate scenes of 3D shapes using matplotlib with white background, grayscale objects, and overhead lighting.
For each distribution/mean combination, creates a folder with particles/ (jsons) and renders/ (pngs) with matching filenames.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import typer
from datetime import datetime
from generate_shape_sample import _generate_shape_sample

def render_shape(ax, shape):
    """
    Render a single shape on the given matplotlib 3D axis in grayscale with enhanced shading.
    """
    ls = LightSource(azdeg=45, altdeg=60)  # Overhead but slightly angled for visible shading
    # Sphere
    if shape["type"] == "sphere":
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        xs = shape["radius"] * np.cos(u) * np.sin(v) + shape["x"]
        ys = shape["radius"] * np.sin(u) * np.sin(v) + shape["y"]
        zs = shape["radius"] * np.cos(v) + shape["z"]
        rgb = ls.shade(zs, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(xs, ys, zs, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
    # Cube
    elif shape["type"] == "cube":
        size = shape["size"]
        x0, y0, z0 = shape["x"], shape["y"], shape["z"]
        cube_verts = np.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
        ]) * (size/2) + np.array([x0, y0, z0])
        cube_faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
        ax.add_collection3d(Poly3DCollection([[cube_verts[vert] for vert in face] for face in cube_faces], facecolors='0.7', alpha=0.8))
    # Cylinder
    elif shape["type"] == "cylinder":
        theta = np.linspace(0, 2*np.pi, 30)
        z = np.array([0, shape["height"]])
        theta, z = np.meshgrid(theta, z)
        xcyl = shape["radius"] * np.cos(theta) + shape["x"]
        ycyl = shape["radius"] * np.sin(theta) + shape["y"]
        zcyl = z + shape["z"]
        rgb = ls.shade(zcyl, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(xcyl, ycyl, zcyl, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
    # Torus
    elif shape["type"] == "torus":
        theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, 30), np.linspace(0, 2*np.pi, 30))
        R, r = shape["R"], shape["r"]
        xtor = (R + r * np.cos(phi)) * np.cos(theta) + shape["x"]
        ytor = (R + r * np.cos(phi)) * np.sin(theta) + shape["y"]
        ztor = r * np.sin(phi) + shape["z"]
        rgb = ls.shade(ztor, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(xtor, ytor, ztor, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
    # Distorted Sphere
    elif shape["type"] == "distorted_sphere":
        n = 40
        theta_d, phi_d = np.meshgrid(np.linspace(0, 2*np.pi, n), np.linspace(0, np.pi, n))
        rd = shape["base_radius"] + shape["distortion"] * np.sin(3*theta_d) * np.sin(2*phi_d)
        xd = rd * np.sin(phi_d) * np.cos(theta_d) + shape["x"]
        yd = rd * np.sin(phi_d) * np.sin(theta_d) + shape["y"]
        zd = rd * np.cos(phi_d) + shape["z"]
        rgb = ls.shade(zd, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(xd, yd, zd, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
    # Pebble
    elif shape["type"] == "pebble":
        a, b, c = shape["a"], shape["b"], shape["c"]
        theta_p, phi_p = np.meshgrid(np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 40))
        xp = a * (1 + 0.1 * np.sin(3*theta_p)) * np.sin(phi_p) * np.cos(theta_p) + shape["x"]
        yp = b * (1 + 0.1 * np.cos(2*phi_p)) * np.sin(phi_p) * np.sin(theta_p) + shape["y"]
        zp = c * np.cos(phi_p) + shape["z"]
        rgb = ls.shade(zp, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(xp, yp, zp, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
    # Superquadric
    elif shape["type"] == "superquadric":
        n = 40
        u, v = np.meshgrid(np.linspace(-np.pi, np.pi, n), np.linspace(-np.pi/2, np.pi/2, n))
        def cos_e(angle, expn):
            return np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** expn
        def sin_e(angle, expn):
            return np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** expn
        X = shape["a1"] * cos_e(v, shape["eps1"]) * cos_e(u, shape["eps2"]) + shape["x"]
        Y = shape["a2"] * cos_e(v, shape["eps1"]) * sin_e(u, shape["eps2"]) + shape["y"]
        Z = shape["a3"] * sin_e(v, shape["eps1"]) + shape["z"]
        rgb = ls.shade(Z, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(X, Y, Z, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)

def generate_scene(
    distribution: str = typer.Option("random", "-d", "--distribution", help="distribution type"),
    mean: float = typer.Option(1.0, "-m", "--mean", help="mean value for shape size"),
    shape: str = typer.Option("sphere", "-t", "--shape", help="shape type"),
    samplesize: int = typer.Option(10, "-n", "--samplesize", help="number of shapes per scene"),
    outdir: str = typer.Option("./data/synthetic-data", "-o", "--outdir", help="output directory")
):
    """
    Generate a scene with specified shape, distribution, and mean. Save JSON and PNG with matching filenames in particles/ and renders/.
    """
    # Directory setup
    folder_name = f"{shape}-{distribution}-mean{mean}"
    base_dir = os.path.join(outdir, folder_name)
    particles_dir = os.path.join(base_dir, "particles")
    renders_dir = os.path.join(base_dir, "renders")
    os.makedirs(particles_dir, exist_ok=True)
    os.makedirs(renders_dir, exist_ok=True)
    # Generate shapes
    json_filename = f"scene_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    png_filename = json_filename.replace('.json', '.png')
    json_path = os.path.join(particles_dir, json_filename)
    png_path = os.path.join(renders_dir, png_filename)
    _generate_shape_sample(samplesize, json_path, shape, distribution, mean)
    # Load shapes
    with open(json_path, 'r') as f:
        data = json.load(f)
    shapes = data["shapes"]
    # Render scene
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=90, azim=0)  # Top-down view
    for shape in shapes:
        render_shape(ax, shape)
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.05, dpi=200)
    plt.close(fig)
    print(f"Saved: {json_path}\nSaved: {png_path}")

if __name__ == "__main__":
    typer.run(generate_scene)
