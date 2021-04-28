import setup_path 
import airsim
import cv2
import time
import math
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# cnnecting to the client
client = airsim.VehicleClient()
client.confirmConnection()

car_client = airsim.CarClient()
car_client.confirmConnection()

# creating directories
seq = "99"
seq_dir = f'Virtus/sequences/{seq}'
pos_dir = f'Virtus/poses'
left_dir = f'Virtus/sequences/{seq}/image_2'
right_dir = f'Virtus/sequences/{seq}/image_3'

os.makedirs(seq_dir)
os.makedirs(left_dir)
os.makedirs(right_dir)

# creating files
times = open(os.path.join(seq_dir,"times.txt"),"w+")
poses = open(os.path.join(pos_dir,f"{seq}.txt"),"w+")
speed = open(os.path.join(seq_dir,"speed.txt"),"w+")

# user definitions
counter = 0
fps = 10
frames = 6000

# resetting vehicle's pose
client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)

# rotation matrix to transform to camera coordinate system
r = R.from_euler('zxy', [-90, -90, 0], degrees=True)

# translation vector
t = np.zeros((3,1))

all_times = np.zeros((1,frames))
all_speeds = np.zeros((1,frames))
all_poses = np.zeros((3,4,frames))
all_left = []
all_right = []

toc = time.time()

# main looping
while counter < frames:

    tic = time.time()

    # getting time
    all_times[0,counter] = tic - toc

    # capturing images
    responses = client.simGetImages([
        airsim.ImageRequest("myLeftCam", airsim.ImageType.Scene)
    ])
    # car pose
    pose = client.simGetVehiclePose()
    # writing speed file
    car_state = car_client.getCarState()
    #client.simPause(False)

    # getting speed
    all_speeds[0,counter] = car_state.speed

    # getting pose
    t[0] = pose.position.x_val
    t[1] = pose.position.y_val
    t[2] = pose.position.z_val
    new_t = r.as_matrix()@t

    angles = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    r_m = R.from_euler('zxy', angles)
    
    M = np.hstack((r_m.as_matrix(),new_t))
    all_poses[:,:,counter] = M
    
    number_str = str(counter)
    zero_filled_number = number_str.zfill(6)
    all_left.append({zero_filled_number:responses[0].image_data_uint8})

    counter += 1
    print(f'Frames captured: {counter}')

    rest = time.time() - tic
    if rest < 1/fps:
        time.sleep(1/fps - rest)

# writing files
for i in range(frames):
    temp = np.array([all_times[0,i]],dtype=float)
    temp = np.format_float_scientific(temp,unique=False,precision=6)
    times.write(f"{temp}\n")
times.close()

for i in range(frames):
    temp = np.array([all_speeds[0,i]],dtype=float)
    temp = np.format_float_scientific(temp,unique=False,precision=6)
    speed.write(f"{temp}\n")
speed.close()

for i in range(frames):
    string = ' '.join([np.format_float_scientific(x,unique=False,precision=6) for x in np.ravel(all_poses[:,:,i])])
    poses.write(f"{string}\n")
poses.close()

for image in all_left:
    for key in image:
        filename = os.path.join(left_dir, f"{key}")
        airsim.write_file(os.path.normpath(filename + '.png'), image[key])
        I = cv2.imread(filename+'.png')
        ret = cv2.imwrite(filename+'.png',I)
