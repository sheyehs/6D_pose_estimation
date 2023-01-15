import bpy
import os
import numpy as np
from matplotlib import pyplot as plt

bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True

scene = bpy.context.scene
scene.use_nodes = True
tree = scene.node_tree
links = tree.links

# clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# create input render layer node
rl = tree.nodes.new('CompositorNodeRLayers')    

# create map range node in middle
mr = tree.nodes.new(type="CompositorNodeMapRange")
# map depth from [0.5, 5] to [0, 5] and clamp out of [0.5, 0]
mr.use_clamp = True
mr.inputs[1].default_value = 0.5      
mr.inputs[2].default_value = 5.0 
mr.inputs[3].default_value = 0.0
mr.inputs[4].default_value = 5.0

# link render node to map node
links.new(rl.outputs['Depth'], mr.inputs[0])

# create output node
v = tree.nodes.new('CompositorNodeViewer')   
v.use_alpha = False

# link map node to view node
links.new(mr.outputs[0], v.inputs[0]) # link Z to output


filepath = r'C:\Users\sheye\Repos\6D_pose_estimation\synthetic_images'
filename = "depth.png" 
filepath = os.path.join(filepath, filename)

# file output node
# todo: can change file name?
#fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
#fileOutput.base_path = r'C:\Users\sheye\Repos\6D_pose_estimation\synthetic_images'
#links.new(mr.outputs[0], fileOutput.inputs[0])

# must update depth after render
bpy.ops.render.render()

# get viewer pixels
pixels = bpy.data.images['Viewer Node'].pixels
arr = np.array(pixels)
arr = arr.reshape((1080,1920,4))  # camera resolution and RGBA
arr = np.flip(arr, axis=0)
plt.imshow(arr)

plt.savefig(filepath)
