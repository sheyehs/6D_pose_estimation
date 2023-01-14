import bpy
import mathutils

import os
import time
import math
import random

N_ITERATIONS = 10
LEN_STR_N_ITERATIONS = len(str(N_ITERATIONS))

PLANE_SIZE = 2

N_CUBES_MIN = 25
N_CUBES_MAX = 50

CUBE_SIZE = 0.1
CUBE_LOCATION_RANGE = (-0.5, 0.5)
CUBE_ROTATION_RANGE = (0, 90)

start = time.time()

# set camera FOV
camera = bpy.context.scene.camera
camera.data.lens_unit = 'FOV'
camera.data.angle = math.pi / 2

for i in range(N_ITERATIONS):
    # clear the scene
    bpy.ops.object.select_by_type(extend=False, type='MESH')
    bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')
    
    for material in bpy.data.materials:
        material.user_clear()
        bpy.data.materials.remove(material)
        
    # randomize camera
    if 1: 
        camera = bpy.context.scene.camera  
        location = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(1, 2)] 
        point = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0]
        direction = mathutils.Vector(point) - mathutils.Vector(location)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        camera.location = location
        camera.rotation_euler =  rot_quat.to_euler()
    
    # randomize light
    if 1:
        light = bpy.data.objects['Light']
        location = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(2, 3)]
        point = [random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0]
        direction = mathutils.Vector(point) - mathutils.Vector(location)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        light.location = location
        light.rotation_euler =  rot_quat.to_euler()
        light = bpy.data.lights['Light']
        light.energy = random.randint(250, 750)
    
    # generate the plane
    bpy.ops.mesh.primitive_plane_add(size=PLANE_SIZE)
    plane = bpy.data.objects['Plane']
    mat = bpy.data.materials.new("plane_material")
    mat.use_nodes = True 
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (random.random(), random.random(), random.random(), 1)
    plane.active_material = mat
    bpy.ops.rigidbody.object_add()
    plane.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.friction = 1
    
    # generate a box
    location = [random.uniform(-0.5, 0.5) for _ in range(2)] + [0.1]
    rotation = [0.0, 0.0] + [random.uniform(0, math.pi / 4)]
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation, scale=[0.5, 0.5, 0.1])
    box = bpy.data.objects['Cube']
    box.name = 'box'
    box.color = (1, 0, 0, 1)
    bpy.ops.rigidbody.object_add() 
    plane.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.friction = 1
    
    # generate cubes
    n_cubes = random.randint(N_CUBES_MIN, N_CUBES_MAX)
    for _ in range(n_cubes):
        location = [random.uniform(*CUBE_LOCATION_RANGE) for _ in range(2)] + [random.uniform(0.5, 1)]
        rotation = [math.radians(random.uniform(*CUBE_ROTATION_RANGE)) for _ in range(3)]
        cube = bpy.ops.mesh.primitive_cube_add(size=CUBE_SIZE, location=location, rotation=rotation)
        bpy.ops.object.modifier_add(type='COLLISION')
        bpy.ops.rigidbody.object_add()

        bpy.context.object.rigid_body.friction = 1

        
    # finish the animation to let the objects falling down
    bpy.ops.nla.bake(frame_start=1, frame_end=250, bake_types={'OBJECT'})
    bpy.context.scene.frame_set(250)


    # save the scene
    filepath = r'C:\Users\sheye\Repos\6D_pose_estimation\synthetic_images'
    filename = f'img{i+1:0{LEN_STR_N_ITERATIONS}d}'
    bpy.context.scene.render.filepath =  os.path.join(filepath, filename)
    bpy.ops.render.render(write_still=True)


print(f"time used: {time.time() - start}")
