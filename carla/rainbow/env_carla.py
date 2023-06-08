import math
import cv2
import random
import time
import numpy as np
import glob
import os
import sys


try:
    sys.path.append(glob.glob('C:/Users/Marinus/OneDrive/Desktop/Development/CARLA_0.9.14/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Error")
    pass
import carla




x_bounds = (-117, -25)
y_bounds = (18, 141)
vx_bounds = (-40, 40)
vy_bounds = (-40, 40)

n_x = 100
n_y = 130
n_vx = 40
n_vy = 40
rand = 0

VIDEO_LOCATION="F:/Coding/recordings/"
SECONDS_PER_EPISODE = 30


MAP_X_BOUNDS = (-120, 120)
MAP_Y_BOUNDS = (-80, 150)


def discretize_state(state):
    x_idx = int((state[0] - x_bounds[0]) / (x_bounds[1] - x_bounds[0]) * n_x)
    y_idx = int((state[1] - y_bounds[0]) / (y_bounds[1] - y_bounds[0]) * n_y)
    vx_idx = int((state[2] - vx_bounds[0]) / (vx_bounds[1] - vx_bounds[0]) * n_vx)
    vy_idx = int((state[3] - vy_bounds[0]) / (vy_bounds[1] - vy_bounds[0]) * n_vy)
    # print((x_idx, y_idx, vx_idx, vy_idx))
    return (x_idx, y_idx, vx_idx, vy_idx)

class CarEnv:
    STEER_AMT = 1.0
    front_camera = None
    goal_position = carla.Location(x=-109, y=44, z=0)
    vehicle_prev_loc = carla.Location(x=-25, y=135, z=0.1)
    im_width = 640
    im_height = 480

    def __init__(self, render=False):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        # self.client.load_world('Town01') # Change 'Town01' to your desired map
        self.world = self.client.get_world()
        self.world.unload_map_layer(carla.MapLayer.Buildings)
        settings = self.world.get_settings()
        # settings.no_rendering_mode = True
        settings.fixed_delta_seconds = 0.05
        # settings.quality_level = 'high'
        self.render = render
        # settings.fixed_delta_seconds = 0.1
        self.collision_hist = []
        self.actor_list = []

        # weather = carla.WeatherParameters()

        # # Set weather parameters for snow
        # weather.cloudiness = 80.0
        # weather.precipitation = 10.0
        # weather.precipitation_deposits = 10.0
        # weather.wind_intensity = 10.0
        # weather.sun_azimuth_angle = -180.0
        # weather.sun_altitude_angle = -90.0

        # # Apply the new weather settings
        # self.world.set_weather(weather)

        
        self.world.apply_settings(settings)
        self.blueprint_library = self.world.get_blueprint_library()

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # self.camera_transform = carla.Transform(carla.Location(x=-6.5, z=2.4))
        self.camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7), carla.Rotation(pitch=-30))
        self.camera_bp.set_attribute('image_size_x', f'{self.im_width}')
        self.camera_bp.set_attribute('image_size_y', f'{self.im_height}')
        self.camera_bp.set_attribute('fov', '110')
        self.video_file = None
        self.video_codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_fps = 30
        self.video_size = (self.im_width, self.im_height)
        self.video_writer = None
        
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.starting_position = None
        self.spawn_point = None
    def destroy_actors(self):
        for actor in self.actor_list:
            actor.destroy()
        # print("All actors destroyed")
        self.actor_list = []
    def approaching_goal(self, prev, curr):
        dist = self.distance_to_goal(prev) - self.distance_to_goal(curr)
        # print(dist)
        if dist > 0.01:
            return 1
        elif dist < 0:
            return -0.5
        else:
            return -0.2
        # print(f"prev: {prev} curr: {curr} dist: {dist}")

    def video(self, episode):
        self.video_file = f"{VIDEO_LOCATION}episode_{episode}.mp4"
        self.video_writer = cv2.VideoWriter(self.video_file, self.video_codec, self.video_fps, self.video_size)

    # Define a function that computes the distance to the goal
    def distance_to_goal(self, location):
        return location.distance(self.goal_position)
    
    def capture_and_save_image(self, camera, episode):
        if self.front_camera is not None:
            # Convert the image to BGR format
            img = cv2.cvtColor(self.front_camera, cv2.COLOR_RGB2BGR)
            # Save the image as a file
            cv2.imwrite('episode_{}.png'.format(episode), img)
        # image = camera.convert(carla.ColorConverter.Raw)
        # image.save_to_disk('episode_{}.png'.format(episode))

    def process_img(self, image):
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.video_writer is not None:
            # Convert the image to BGR format
            img = cv2.cvtColor(i3, cv2.COLOR_RGB2BGR)
            # Write the frame to the video file
            self.video_writer.write(img)
        if self.render:
            cv2.imshow("",i3)
            cv2.waitKey(1)
        self.front_camera = i3
    
    def reset(self):
        
        self.destroy_actors()
        self.collision_hist = []
        # self.transform = random.choice(self.world.get_map().get_spawn_points())
        # x_bounds = (-117, -25)
        # y_bounds = (18, 141)
        rand_spawn_point_x = random.randint(-28, -26)
        rand_spawn_point_y = random.randint(130, 140)
        yaw = random.randint(160,220)
        # print(rand_spawn_point_x, rand_spawn_point_y, yaw)
        self.spawn_point = carla.Transform(carla.Location(x=rand_spawn_point_x, y=rand_spawn_point_y, z=0.1), carla.Rotation(yaw=yaw))
        # self.spawn_point = self.world.get_map().get_spawn_points()[1]
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
        self.actor_list.append(self.vehicle)
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # time.sleep(2)

        #attach camera for recordings
        self.camera = self.world.spawn_actor(self.camera_bp, self.camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: self.process_img(data))

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # while self.front_camera is None:
        #     time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # time.sleep(0.2)
        self.vehicle_prev_loc = self.vehicle.get_location()
        loc = self.vehicle_prev_loc
        vel = self.vehicle.get_velocity()
        # return discretize_state([loc.x, loc.y, vel.x, vel.y])
        return (loc.x, loc.y, vel.x, vel.y)
        # return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def step(self, action):
        # time.sleep(0.01)
        # left
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=-1*self.STEER_AMT, brake=0.0))
        # straight
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0, brake=0.0))
        # right
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=1*self.STEER_AMT, brake=0.0))
        # brake
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0, brake=1.0))
        loc = self.vehicle.get_location()
        # print(loc.x, loc.y)
        vel = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2))
        done = False
        reward = self.approaching_goal(self.vehicle_prev_loc, loc)
        # print(reward)
        if len(self.collision_hist) != 0:
            done = True
            reward = -1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
        # if loc.x <= MAP_X_BOUNDS[0] or loc.x >= MAP_X_BOUNDS[1] or loc.y <= MAP_Y_BOUNDS[0] or loc.y >= MAP_Y_BOUNDS[1]:
        #     done = True
        #     reward = -1000
        if loc.x <= -117 or loc.x >= -25 or loc.y <= 18 or loc.y >= 141:
            reward = -1000
        # x_bounds = (-117, -25)
        # y_bounds = (18, 141)
        if loc.x <= -117:
            loc.x = -116
            done = True
        if loc.x >= -25:
            loc.x = -26
            done = True
        if loc.y <= 18:
            loc.y = 19
            done = True
        if loc.y >= 141:
            loc.y = 140
            done = True
        # goal_position = carla.Location(x=-109, y=44, z=0)
        if loc.x >= -117 and loc.x <= -101 and loc.y >= 42 and loc.y <= 44:
            reward = 5
            done = True
            print("GOAL REACHED!!!")
        self.vehicle_prev_loc = carla.Location(x=loc.x, y=loc.y, z=0.1)
        # observation = discretize_state([loc.x, loc.y, vel.x, vel.y])
        return (loc.x, loc.y, vel.x, vel.y), reward, done, None
