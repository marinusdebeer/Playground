import numpy as np
import glob
import os
import sys
import random
import time
try:
    sys.path.append(glob.glob('C:/Users/Marinus/OneDrive/Desktop/Development/CARLA_0.9.14/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(3.0)


actors = client.get_world().get_actors()

# Filter the list to include only vehicles and sensors
vehicles = [actor for actor in actors if 'vehicle' in actor.type_id]
sensors = [actor for actor in actors if 'sensor' in actor.type_id]

# Destroy all vehicles and sensors
for vehicle in vehicles:
    vehicle.destroy()
for sensor in sensors:
    sensor.destroy()