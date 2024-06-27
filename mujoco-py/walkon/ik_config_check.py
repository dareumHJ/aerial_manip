import numpy as np
import mujoco
import mujoco.viewer as viewer
import mediapy as media
import pandas as pd
import math

xml = "WALKON5_full_revolute.xml"
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 5

# Put a position of the joints to get a test point
pi = np.pi

# qpos = np.array([body x, y, z, qw, qx, qy, qz, L ABD, L ROT, L EXT, LK FLX, LA EVR, LA PLA, R ADD, R INNER ROT, R EXT, RK FLX, RA INV, RA PLA ])
initial_qpos = np.array([0, 0, 1.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
final_qpos = np.array([0, 0, 1.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0])

traj = pd.read_csv("motion_control/gait_swing_left_ABS.csv", skiprows=2)
traj = traj.astype(float)
# qpos = np.array([traj.iloc[i,18], traj.iloc[i,19], traj.iloc[i,20], traj.iloc[i,21], traj.iloc[i,22], traj.iloc[i,23], traj.iloc[i,24], traj.iloc[i,2], -traj.iloc[i,3], traj.iloc[i,1], traj.iloc[i,4], traj.iloc[i,6], traj.iloc[i,5], -traj.iloc[i,11], traj.iloc[i,12], -traj.iloc[i,10], traj.iloc[i,13], traj.iloc[i,15], -traj.iloc[i,14]])

# Set the initial joint position
data.qpos = initial_qpos.copy()

# Capture the initial and target positions
mujoco.mj_forward(model, data)
init_point = data.body('BASE').xpos.copy()

data.qpos = final_qpos.copy()
mujoco.mj_forward(model, data)
target_point = data.body('BASE').xpos.copy()

# Interpolate between initial and final positions
num_steps = 100  # Number of steps in the trajectory
qpos_trajectory = np.linspace(initial_qpos, final_qpos, num_steps)

# Initialize an empty list to store the rendered images
trajectory_images = []


with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True

    for i in range(1, traj.shape[0] + 1, 20):
        # qpos = np.array([0, 0, 1.2, 0, 0, 0, 0, traj.iloc[i,2], -traj.iloc[i,3], -traj.iloc[i,1], traj.iloc[i,4], traj.iloc[i,6], traj.iloc[i,5], -traj.iloc[i,11], traj.iloc[i,12], -traj.iloc[i,10], traj.iloc[i,13], traj.iloc[i,15], -traj.iloc[i,14]]) * math.pi / 180
        # qpos = np.array([0, 0, 1.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        qpos = np.array([traj.iloc[i,18], traj.iloc[i,19], traj.iloc[i,20], traj.iloc[i,21], traj.iloc[i,22], traj.iloc[i,23], traj.iloc[i,24], traj.iloc[i,2], -traj.iloc[i,3], -traj.iloc[i,1], traj.iloc[i,4], traj.iloc[i,6], traj.iloc[i,5], -traj.iloc[i,11], traj.iloc[i,12], -traj.iloc[i,10], traj.iloc[i,13], traj.iloc[i,15], -traj.iloc[i,14]])
        qpos[7:] = qpos[7:] * math.pi / 180
        # qpos = qpos_trajectory[i,:]
        data.qpos = qpos
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        image = renderer.render()
        trajectory_images.append(image)

        # Optional: Update viewer in real-time (slows down the process)
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

            model.vis.scale.contactwidth = 0.05
            model.vis.scale.contactheight = 0.015
            model.vis.scale.forcewidth = 0.025
            model.vis.map.force = 0.15

        viewer.sync()
        print(qpos)

# Save or display the trajectory images as a video or sequence
media.show_images(trajectory_images, title="Trajectory", fps=60)
