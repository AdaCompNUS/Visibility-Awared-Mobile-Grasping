"""Shared Blender (bpy) helpers: Cycles setup, lights, camera, floor, output."""
import math
import os

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def setup_cycles(
    samples=128,
    res=(1920, 1080),
    fps=30,
    denoise=True,
    transparent=False,
    time_limit=0.0,
    motion_blur=False,
    exposure=0.0,
    look="AgX - Medium High Contrast",
):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.refresh_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.time_limit = time_limit
    sc.cycles.use_denoising = denoise
    sc.cycles.denoiser = "OPTIX"
    sc.cycles.max_bounces = 6
    sc.cycles.transparent_max_bounces = 8
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.fps = fps
    sc.render.film_transparent = transparent
    sc.render.use_motion_blur = motion_blur
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    sc.render.image_settings.compression = 50
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = look
    except Exception:
        pass
    sc.view_settings.exposure = exposure
    return sc


def world_background(rgb=(0.01, 0.011, 0.014), strength=1.0):
    sc = bpy.context.scene
    w = sc.world or bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*rgb, 1.0)
    bg.inputs[1].default_value = strength
    return w


def world_gradient(top=(0.02, 0.03, 0.05), bottom=(0.0, 0.0, 0.0), strength=1.0):
    sc = bpy.context.scene
    w = sc.world or bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs[1].default_value = strength
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*bottom, 1)
    ramp.color_ramp.elements[1].color = (*top, 1)
    tex = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = "MULTIPLY_ADD"
    m.inputs[1].default_value = 0.5
    m.inputs[2].default_value = 0.5
    nt.links.new(tex.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], m.inputs[0])
    nt.links.new(m.outputs[0], ramp.inputs[0])
    nt.links.new(ramp.outputs["Color"], bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    return w


def track_to(obj, target):
    c = obj.constraints.new("TRACK_TO")
    c.target = target
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    return c


def add_target(name, loc, collection=None):
    e = bpy.data.objects.new(name, None)
    e.location = loc
    (collection or bpy.context.scene.collection).objects.link(e)
    return e


def add_area_light(
    name,
    loc,
    target,
    size=2.0,
    energy=500.0,
    color=(1, 1, 1),
    collection=None,
    shape="SQUARE",
    spread=180,
):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.size = size
    ld.color = color
    ld.shape = shape
    ld.spread = math.radians(spread)
    lo = bpy.data.objects.new(name, ld)
    lo.location = loc
    (collection or bpy.context.scene.collection).objects.link(lo)
    track_to(lo, target)
    return lo


def add_sun(name, rotation, energy=3.0, color=(1, 1, 1), angle=2.0, collection=None):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = energy
    ld.color = color
    ld.angle = math.radians(angle)
    lo = bpy.data.objects.new(name, ld)
    lo.rotation_euler = rotation
    (collection or bpy.context.scene.collection).objects.link(lo)
    return lo


def add_camera(
    name, loc, target, lens=35.0, collection=None, sensor=36.0, dof_dist=None, fstop=2.8
):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = sensor
    cd.clip_end = 200
    if dof_dist is not None:
        cd.dof.use_dof = True
        cd.dof.focus_distance = dof_dist
        cd.dof.aperture_fstop = fstop
    co = bpy.data.objects.new(name, cd)
    co.location = loc
    (collection or bpy.context.scene.collection).objects.link(co)
    if target is not None:
        track_to(co, target)
    bpy.context.scene.camera = co
    return co


def add_floor(
    size=40.0,
    rgba=(0.02, 0.02, 0.024, 1),
    roughness=0.25,
    metallic=0.0,
    z=0.0,
    name="floor",
    collection=None,
):
    me = bpy.data.meshes.new(name)
    s = size / 2
    me.from_pydata([(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = rgba
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Metallic"].default_value = metallic
    me.materials.append(mat)
    return ob


def render_still(path, frame=None):
    sc = bpy.context.scene
    if frame is not None:
        sc.frame_set(frame)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def render_animation(dirpath, start, end, step=1, prefix="frame_"):
    sc = bpy.context.scene
    os.makedirs(dirpath, exist_ok=True)
    sc.frame_start, sc.frame_end, sc.frame_step = start, end, step
    sc.render.filepath = os.path.join(dirpath, prefix)
    bpy.ops.render.render(animation=True)
    return dirpath


def ease_in_out(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def iter_fcurves(action):
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                for cb in strip.channelbags:
                    for fc in cb.fcurves:
                        yield fc
    else:
        for fc in action.fcurves:
            yield fc


def smooth_fcurves(obj, interpolation="BEZIER"):
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    for fc in iter_fcurves(ad.action):
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation
            kp.handle_left_type = kp.handle_right_type = "AUTO_CLAMPED"


def linear_fcurves(obj):
    smooth_fcurves(obj, "LINEAR")
