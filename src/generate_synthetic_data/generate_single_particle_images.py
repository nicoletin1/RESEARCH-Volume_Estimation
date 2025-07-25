#!/usr/bin/env python
"""
Generate a 128x128 grayscale image for each particle, with a corresponding JSON file containing size and shape parameters.
Each image contains a single particle centered in the frame, rendered with matplotlib (top-down, white background, grayscale, shaded).
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import typer
from datetime import datetime
from generate_shape_sample import _sample_size


def render_single_particle(shape_type, params, out_png, out_json, rotation_deg=0, tilt_x=0, tilt_y=0):
    """
    Render a single particle (shape) centered in a 128x128 image, rotated in-plane by rotation_deg, and tilted by tilt_x and tilt_y, then save its parameters to JSON.
    """
    fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=90 + tilt_x, azim=tilt_y)
    ls = LightSource(azdeg=45, altdeg=60)
    # Center at (0,0,0)
    x, y, z = 0, 0, 0
    # Rotation matrix for in-plane rotation (Z)
    theta = np.deg2rad(rotation_deg)
    Rmat_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0, 0, 1]])
    # Rotation matrix for tilt (X and Y)
    tx = np.deg2rad(tilt_x)
    ty = np.deg2rad(tilt_y)
    Rmat_x = np.array([[1, 0, 0],
                       [0, np.cos(tx), -np.sin(tx)],
                       [0, np.sin(tx), np.cos(tx)]])
    Rmat_y = np.array([[np.cos(ty), 0, np.sin(ty)],
                       [0, 1, 0],
                       [-np.sin(ty), 0, np.cos(ty)]])
    Rmat = Rmat_z @ Rmat_y @ Rmat_x
    def rotate(X, Y, Z):
        pts = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        pts_rot = pts @ Rmat.T
        return pts_rot[:,0].reshape(X.shape), pts_rot[:,1].reshape(Y.shape), pts_rot[:,2].reshape(Z.shape)
    # Render shape
    if shape_type == "sphere":
        # Spheres are symmetric, so only one rotation is needed
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        xs = params["radius"] * np.cos(u) * np.sin(v) + x
        ys = params["radius"] * np.sin(u) * np.sin(v) + y
        zs = params["radius"] * np.cos(v) + z
        xs, ys, zs = rotate(xs, ys, zs)
        rgb = ls.shade(zs, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(xs, ys, zs, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
    elif shape_type == "cube":
        size = params["size"]
        cube_verts = np.array([
            [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
            [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
        ]) * (size/2)
        cube_verts_rot = cube_verts @ Rmat.T
        cube_faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
        ax.add_collection3d(Poly3DCollection([[cube_verts_rot[vert] for vert in face] for face in cube_faces], facecolors='0.7', alpha=0.8))
    elif shape_type == "cylinder":
        # Cylinder axis orientation: random in XY plane, then apply in-plane rotation
        axis_angle = np.deg2rad(rotation_deg)
        axis = np.array([np.cos(axis_angle), np.sin(axis_angle), 0])
        theta_c = np.linspace(0, 2*np.pi, 30)
        zc = np.array([-params["height"] / 2, params["height"] / 2])
        theta_c, zc = np.meshgrid(theta_c, zc)
        xcyl = params["radius"] * np.cos(theta_c)
        ycyl = params["radius"] * np.sin(theta_c)
        zcyl = zc
        pts = np.stack([xcyl.flatten(), ycyl.flatten(), zcyl.flatten()], axis=1)
        def rotation_matrix_from_vectors(vec1, vec2):
            a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            if s == 0:
                return np.eye(3)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
        R_align = rotation_matrix_from_vectors(np.array([0,0,1]), axis)
        pts_rot = pts @ R_align.T @ Rmat.T
        xcyl = pts_rot[:,0].reshape(theta_c.shape)
        ycyl = pts_rot[:,1].reshape(theta_c.shape)
        zcyl = pts_rot[:,2].reshape(theta_c.shape)
        rgb = ls.shade(zcyl, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(xcyl, ycyl, zcyl, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
    elif shape_type == "torus":
        theta_t, phi_t = np.meshgrid(np.linspace(0, 2*np.pi, 30), np.linspace(0, 2*np.pi, 30))
        R, r = params["R"], params["r"]
        xtor = (R + r * np.cos(phi_t)) * np.cos(theta_t)
        ytor = (R + r * np.cos(phi_t)) * np.sin(theta_t)
        ztor = r * np.sin(phi_t)
        xtor, ytor, ztor = rotate(xtor, ytor, ztor)
        rgb = ls.shade(ztor, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(xtor, ytor, ztor, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
    elif shape_type == "distorted_sphere":
        n = 40
        theta_d, phi_d = np.meshgrid(np.linspace(0, 2*np.pi, n), np.linspace(0, np.pi, n))
        rd = params["base_radius"] + params["distortion"] * np.sin(3*theta_d) * np.sin(2*phi_d)
        xd = rd * np.sin(phi_d) * np.cos(theta_d)
        yd = rd * np.sin(phi_d) * np.sin(theta_d)
        zd = rd * np.cos(phi_d)
        xd, yd, zd = rotate(xd, yd, zd)
        rgb = ls.shade(zd, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(xd, yd, zd, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
    elif shape_type == "pebble":
        a, b, c = params["a"], params["b"], params["c"]
        theta_p, phi_p = np.meshgrid(np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 40))
        xp = a * (1 + 0.1 * np.sin(3*theta_p)) * np.sin(phi_p) * np.cos(theta_p)
        yp = b * (1 + 0.1 * np.cos(2*phi_p)) * np.sin(phi_p) * np.sin(theta_p)
        zp = c * np.cos(phi_p)
        xp, yp, zp = rotate(xp, yp, zp)
        rgb = ls.shade(zp, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(xp, yp, zp, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
    elif shape_type == "superquadric":
        n = 40
        u, v = np.meshgrid(np.linspace(-np.pi, np.pi, n), np.linspace(-np.pi/2, np.pi/2, n))
        def cos_e(angle, expn):
            return np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** expn
        def sin_e(angle, expn):
            return np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** expn
        X = params["a1"] * cos_e(v, params["eps1"]) * cos_e(u, params["eps2"])
        Y = params["a2"] * cos_e(v, params["eps1"]) * sin_e(u, params["eps2"])
        Z = params["a3"] * sin_e(v, params["eps1"])
        X, Y, Z = rotate(X, Y, Z)
        rgb = ls.shade(Z, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
        ax.plot_surface(X, Y, Z, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
        # Set axis limits to fit the needle
        ax.set_xlim(-radius*2, radius*2)
        ax.set_ylim(-radius*2, radius*2)
        ax.set_zlim(-height/2, height/2)
    else:
        # Needle: pointed (double cone) or blunt (thin cylinder)
        if shape_type == "needle":
            radius = params["radius"]
            height = params["height"]
            needle_type = params.get("needle_type", "blunt")
            if needle_type == "pointed":
                # Pointed needle: double cone
                n = 30
                theta = np.linspace(0, 2*np.pi, n)
                z_cone = np.linspace(0, height/2, n)
                r_cone = radius * (1 - z_cone/(height/2))
                Z1, Theta1 = np.meshgrid(z_cone, theta)
                R1, _ = np.meshgrid(r_cone, theta)
                X1 = R1 * np.cos(Theta1)
                Y1 = R1 * np.sin(Theta1)
                Z1 = Z1
                # Lower cone
                z_cone2 = np.linspace(0, -height/2, n)
                r_cone2 = radius * (1 + z_cone2/(height/2))
                Z2, Theta2 = np.meshgrid(z_cone2, theta)
                R2, _ = np.meshgrid(r_cone2, theta)
                X2 = R2 * np.cos(Theta2)
                Y2 = R2 * np.sin(Theta2)
                Z2 = Z2
                # Combine
                X = np.concatenate([X1, X2], axis=0)
                Y = np.concatenate([Y1, Y2], axis=0)
                Z = np.concatenate([Z1, Z2], axis=0)
                X, Y, Z = rotate(X, Y, Z)
                rgb = ls.shade(Z, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
                ax.plot_surface(X, Y, Z, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
                ax.set_xlim(-radius*2, radius*2)
                ax.set_ylim(-radius*2, radius*2)
                ax.set_zlim(-height/2, height/2)
            else:
                # Blunt needle: thin cylinder
                theta_c = np.linspace(0, 2*np.pi, 30)
                zc = np.array([-height / 2, height / 2])
                theta_c, zc = np.meshgrid(theta_c, zc)
                xcyl = radius * np.cos(theta_c)
                ycyl = radius * np.sin(theta_c)
                zcyl = zc
                pts = np.stack([xcyl.flatten(), ycyl.flatten(), zcyl.flatten()], axis=1)
                def rotation_matrix_from_vectors(vec1, vec2):
                    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
                    v = np.cross(a, b)
                    c = np.dot(a, b)
                    s = np.linalg.norm(v)
                    if s == 0:
                        return np.eye(3)
                    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
                axis_angle = np.deg2rad(rotation_deg)
                axis = np.array([np.cos(axis_angle), np.sin(axis_angle), 0])
                R_align = rotation_matrix_from_vectors(np.array([0,0,1]), axis)
                pts_rot = pts @ R_align.T @ Rmat.T
                xcyl = pts_rot[:,0].reshape(theta_c.shape)
                ycyl = pts_rot[:,1].reshape(theta_c.shape)
                zcyl = pts_rot[:,2].reshape(theta_c.shape)
                rgb = ls.shade(zcyl, cmap=plt.cm.Greys, vert_exag=0.3, blend_mode='soft')
                ax.plot_surface(xcyl, ycyl, zcyl, facecolors=rgb, alpha=0.8, linewidth=0, antialiased=True, shade=True)
                ax.set_xlim(-radius*2, radius*2)
                ax.set_ylim(-radius*2, radius*2)
                ax.set_zlim(-height/2, height/2)
        else:
            # Default axis limits for other shapes
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-2, 2)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=128)
    plt.close(fig)
    # Save parameters
    params_with_rot = params.copy()
    params_with_rot["rotation_deg"] = rotation_deg
    params_with_rot["tilt_x"] = tilt_x
    params_with_rot["tilt_y"] = tilt_y
    with open(out_json, 'w') as f:
        json.dump(params_with_rot, f)

def generate_single_particle_images(
    distribution: str = typer.Option("random", "-d", "--distribution", help="distribution type"),
    mean: float = typer.Option(1.0, "-m", "--mean", help="mean value for shape size"),
    shape: str = typer.Option("sphere", "-t", "--shape", help="shape type (sphere, mildly_distorted_sphere, rod, blunt_needle, pointed_needle, brick)", show_choices=True, case_sensitive=False),
    samplesize: int = typer.Option(10, "-n", "--samplesize", help="number of images to generate"),
    outdir: str = typer.Option("./data/single_particles", "-o", "--outdir", help="output directory")
):
    """
    Generate a set of 128x128 images, each with a single particle and a JSON file of its parameters.
    Supported shapes: sphere, mildly_distorted_sphere, rod, blunt_needle, pointed_needle, brick.
    Output directory structure: {outdir}/{shape}-{distribution}-mean{mean}/images/ and params/
    """
    folder_name = f"{shape}-{distribution}-mean{mean}"
    base_dir = os.path.join(outdir, folder_name)
    images_dir = os.path.join(base_dir, "images")
    params_dir = os.path.join(base_dir, "params")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    for i in range(samplesize):
        # Map shape names to parameters
        if shape == "sphere":
            params = {"type": "sphere", "radius": float(_sample_size(distribution, mean, default_mean=1.0))}
            rot_deg = np.random.uniform(0, 360)
            tilt_x = np.random.uniform(-30, 30)
            tilt_y = np.random.uniform(-30, 30)
            img_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.png"
            json_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.json"
            out_png = os.path.join(images_dir, img_name)
            out_json = os.path.join(params_dir, json_name)
            render_single_particle(shape, params, out_png, out_json, rotation_deg=rot_deg, tilt_x=tilt_x, tilt_y=tilt_y)
            print(f"Saved: {out_png}\nSaved: {out_json}")
        elif shape == "mildly_distorted_sphere":
            params = {"type": "distorted_sphere", "base_radius": float(_sample_size(distribution, mean, default_mean=1.0)), "distortion": float(np.random.uniform(0.05, 0.15))}
            rot_deg = np.random.uniform(0, 360)
            tilt_x = np.random.uniform(-30, 30)
            tilt_y = np.random.uniform(-30, 30)
            img_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.png"
            json_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.json"
            out_png = os.path.join(images_dir, img_name)
            out_json = os.path.join(params_dir, json_name)
            render_single_particle("distorted_sphere", params, out_png, out_json, rotation_deg=rot_deg, tilt_x=tilt_x, tilt_y=tilt_y)
            print(f"Saved: {out_png}\nSaved: {out_json}")
        elif shape == "rod":
            # Cylinder with high aspect ratio
            radius = float(_sample_size(distribution, mean, default_mean=0.2))
            height = float(_sample_size(distribution, mean, default_mean=2.0))
            params = {"type": "cylinder", "radius": radius, "height": height}
            rot_deg = np.random.uniform(0, 360)
            tilt_x = np.random.uniform(-30, 30)
            tilt_y = np.random.uniform(-30, 30)
            img_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.png"
            json_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.json"
            out_png = os.path.join(images_dir, img_name)
            out_json = os.path.join(params_dir, json_name)
            render_single_particle("cylinder", params, out_png, out_json, rotation_deg=rot_deg, tilt_x=tilt_x, tilt_y=tilt_y)
            print(f"Saved: {out_png}\nSaved: {out_json}")
        elif shape == "blunt_needle":
            # Blunt needle: thin cylinder, radius << height
            height = float(_sample_size(distribution, mean, default_mean=2.0))
            radius = height * 0.05  # radius is 5% of height
            params = {"type": "needle", "radius": radius, "height": height, "needle_type": "blunt"}
            rot_deg = np.random.uniform(0, 360)
            tilt_x = np.random.uniform(-30, 30)
            tilt_y = np.random.uniform(-30, 30)
            img_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.png"
            json_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.json"
            out_png = os.path.join(images_dir, img_name)
            out_json = os.path.join(params_dir, json_name)
            render_single_particle("needle", params, out_png, out_json, rotation_deg=rot_deg, tilt_x=tilt_x, tilt_y=tilt_y)
            print(f"Saved: {out_png}\nSaved: {out_json}")
        elif shape == "pointed_needle":
            # Pointed needle: double cone, radius << height
            height = float(_sample_size(distribution, mean, default_mean=2.0))
            radius = height * 0.05  # radius is 5% of height
            params = {"type": "needle", "radius": radius, "height": height, "needle_type": "pointed"}
            rot_deg = np.random.uniform(0, 360)
            tilt_x = np.random.uniform(-30, 30)
            tilt_y = np.random.uniform(-30, 30)
            img_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.png"
            json_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.json"
            out_png = os.path.join(images_dir, img_name)
            out_json = os.path.join(params_dir, json_name)
            render_single_particle("needle", params, out_png, out_json, rotation_deg=rot_deg, tilt_x=tilt_x, tilt_y=tilt_y)
            print(f"Saved: {out_png}\nSaved: {out_json}")
        elif shape == "brick":
            # Cube with random aspect ratios (rectangular cuboid)
            size_x = float(_sample_size(distribution, mean, default_mean=1.0))
            size_y = float(_sample_size(distribution, mean, default_mean=0.5))
            size_z = float(_sample_size(distribution, mean, default_mean=0.2))
            params = {"type": "cube", "size_x": size_x, "size_y": size_y, "size_z": size_z}
            rot_deg = np.random.uniform(0, 360)
            tilt_x = np.random.uniform(-30, 30)
            tilt_y = np.random.uniform(-30, 30)
            img_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.png"
            json_name = f"particle_{i:04d}_rot{int(rot_deg):03d}.json"
            out_png = os.path.join(images_dir, img_name)
            out_json = os.path.join(params_dir, json_name)
            render_single_particle("brick", params, out_png, out_json, rotation_deg=rot_deg, tilt_x=tilt_x, tilt_y=tilt_y)
            print(f"Saved: {out_png}\nSaved: {out_json}")
        else:
            continue

if __name__ == "__main__":
    typer.run(generate_single_particle_images)
