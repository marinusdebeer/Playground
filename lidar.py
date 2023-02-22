import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2
# Connect to the Carla server and get the world object
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

# --------------
# Add a new LIDAR sensor to my ego
# --------------
lidar_cam = None
lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('channels',str(32))
lidar_bp.set_attribute('points_per_second',str(90000))
lidar_bp.set_attribute('rotation_frequency',str(40))
lidar_bp.set_attribute('range',str(20))
lidar_location = carla.Location(0,0,2)
lidar_rotation = carla.Rotation(0,0,0)
lidar_transform = carla.Transform(lidar_location,lidar_rotation)

blueprint_library = world.get_blueprint_library()
bp = blueprint_library.filter('model3')[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(bp, spawn_point)
lidar_sen = world.spawn_actor(lidar_bp,lidar_transform,attach_to=vehicle)
lidar_sen.listen(lambda point_cloud: point_cloud.save_to_disk('tutorial/new_lidar_output/%.6d.ply' % point_cloud.frame))
time.sleep(50)
