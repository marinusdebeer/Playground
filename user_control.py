import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import pygame

try:
    sys.path.append(glob.glob('C:/Users/Marinus/OneDrive/Desktop/Development/CARLA_0.9.14/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
# Initialize Pygame window
pygame.init()
display = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Initialize CARLA client and world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
im_width = 640
im_height = 480
# Load map and spawn vehicle
world_map = world.get_map()
spawn_points = world_map.get_spawn_points()
vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
vehicle_transform = spawn_points[0]
vehicle = world.spawn_actor(vehicle_bp, vehicle_transform)

# Create camera sensor
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', f'{im_width}')
camera_bp.set_attribute('image_size_y', f'{im_height}')
camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=-30))

camera_bp.set_attribute('fov', '60')
# camera_bp.set_attribute('pitch', '-30')
# camera_bp.set_attribute('yaw', '180')

camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Set camera properties



def process_img(image):
    i = np.array(image.raw_data)
    #np.save("iout.npy", i)
    i2 = i.reshape((im_height, im_width, 4))
    i3 = i2[:, :, :3]
    if True:
        cv2.imshow("",i3)
        cv2.waitKey(1)
    # pygame.surfarray.blit_array(display, i3)
    # pygame.display.flip()



# Set camera callback function
camera_sensor.listen(process_img)

# Define key mappings
key_mappings = {
    pygame.K_w: carla.VehicleControl(throttle=1.0),
    pygame.K_s: carla.VehicleControl(brake=1.0),
    pygame.K_a: carla.VehicleControl(steer=-1.0),
    pygame.K_d: carla.VehicleControl(steer=1.0),
    pygame.K_q: carla.VehicleControl(hand_brake=True),
    pygame.K_ESCAPE: sys.exit
}

# Main loop
while True:
    # Handle keyboard input
    control = carla.VehicleControl()
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key in key_mappings:
                control = key_mappings[event.key]
        elif event.type == pygame.KEYUP:
            if event.key in key_mappings:
                control = carla.VehicleControl()

    # Update vehicle control
    vehicle.apply_control(control)

    # Limit Pygame frame rate
    clock.tick(60)
