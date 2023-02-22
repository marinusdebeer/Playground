import random
from collections import deque
import numpy as np
# import cv2
import time
import tensorflow as tf
from keras import backend as backend
from keras.models import load_model
from carla_dqn_position_velocity import CarEnv, MEMORY_FRACTION


# MODEL_PATH = 'models/pos_vel_-1040_-1243_-781_ep_30_trail_12.model'
# MODEL_PATH = 'models/pos_vel_-93_-163_66_ep_100_trail_12.model'
MODEL_PATH = 'models/pos_vel_392_-190_1291_ep_100_trail_14.model'

model = None
def get_qs(state):
    # # Convert state dictionary to numpy array
    # state_array = np.array([state['lx'], state['ly'], state['vx'], state['vy']])

    # # Reshape the state array and normalize the values
    # state_array = state_array.reshape(-1, 4) / 255

    # Use the model to predict the Q-values for the current state
    q_values = model.predict(state)[0]

    return q_values
if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = MEMORY_FRACTION
    sess = tf.compat.v1.Session(config=config)
    backend.set_session(sess)
    model = load_model(MODEL_PATH)
    env = CarEnv()
    fps_counter = deque(maxlen=60)
    # model.predict(np.ones((1, env.im_height, env.im_width, 3)))
    # get_qs({'lx':-11.1, 'ly':138.5, 'vx':0, 'vy':0})
    while True:
        print('Restarting episode')
        current_state = env.reset()
        env.collision_hist = []
        done = False
        while True:
            step_start = time.time()
            # cv2.imshow(f'Agent - preview', current_state)
            # cv2.waitKey(1)
            qs = get_qs(current_state)
            action = np.argmax(qs)
            new_state, reward, done, _ = env.step(action)
            current_state = new_state
            if done:
                break
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
            print(f'Agent: {len(fps_counter)/sum(fps_counter):>4.1f} FPS | Action: [{qs[0]:>5.2f}, {qs[1]:>5.2f}, {qs[2]:>5.2f}] {action}')
        for actor in env.actor_list:
            actor.destroy()