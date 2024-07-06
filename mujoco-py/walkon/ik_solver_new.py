import numpy as np
import mujoco
import mujoco.viewer as viewer
import mediapy as media
import math

# Choose a model
xml = "WALKON5_left_leg.xml"
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

#Video Setup
DURATION = 20 #(seconds)
FRAMERATE = 100 #(Hz)
frames = []

step_size = 0.005
tol = 0.01
alpha = 0.5
damping = 0.001

def compute_com(model, data):
    total_mass = np.sum(model.body_mass)

    mass = (model.body_mass)
    xpos = (data.xipos)

    com = np.sum(xpos.T * mass, axis=1) / total_mass
    return com



mujoco.mj_resetData(model, data)

# 70 ~ 90
des_qpos1 = 90*np.pi/180
des_qpos2 = 90*np.pi/180

# des_qpos1 = 0*np.pi/180
# des_qpos2 = 0*np.pi/180

# find err value
joint1_pos = data.body(model.body('L_FOOT').id).xipos.copy()
joint2_pos = data.body(model.body('L_ANKLE').id).xipos.copy()
com = compute_com(model, data)
unit_vec1 = [1, 0, 0]
unit_vec2 = [0, 1, 0]
unit_vec2 = unit_vec2 / np.linalg.norm(unit_vec2)

Ks = 30.0
Bs = 3.0
Ks2 = 30.0
Bs2 = 3.0

plant_idx = (data.qpos.size) - 1
inver_idx = (data.qpos.size) - 2

# dq = np.zeros((1, 3))

with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:
        # mujoco.mj_differentiatePos(model, dq, step_size, qpos0, data.qpos)

        joint1_pos = data.body(model.body('L_FOOT').id).xipos.copy()
        joint2_pos = data.body(model.body('L_ANKLE').id).xipos.copy()
        com = compute_com(model, data)
        det_vec1 = com - joint1_pos
        det_vec2 = com - joint2_pos

        cur_pos1 = math.acos(np.dot(det_vec1, unit_vec1) / np.linalg.norm(det_vec1))
        cur_pos2 = math.acos(np.dot(det_vec2, unit_vec2) / np.linalg.norm(det_vec2))
        err1 = -(des_qpos1 - cur_pos1)
        # err1 = des_qpos1 - (data.qpos.copy())[plant_idx]
        err2 = -(des_qpos2 - cur_pos2)
        # err2 = des_qpos2 - (data.qpos.copy())[inver_idx]

        errdiff1 = 0.0 - (data.qvel.copy())[plant_idx - 1]
        errdiff2 = 0.0 - (data.qvel.copy())[inver_idx - 1]

        #Set control signal
        ctrlval1 = (Ks * err1 + Bs * errdiff1)
        ctrlval2 = (Ks2 * err2 + Bs2 * errdiff2)
        data.ctrl = [ctrlval2, ctrlval1]

        if (data.time > 3 and data.time < 4):
            data.xfrc_applied[1] = [-30, -30, 0, 0, 0, 0]
            
        else:
            data.xfrc_applied[1] = [0, 0, 0, 0, 0, 0]
        # Step the simulation.
        mujoco.mj_step(model, data)

        print("---------joint pos and com-----------")
        print(cur_pos1)
        print(des_qpos1)
        print(ctrlval1)
        print("-------------------------------------")
        print(cur_pos2)
        print(des_qpos2)
        print(ctrlval2)

        if len(frames) < data.time * FRAMERATE:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
        viewer.sync()

media.show_video(frames, fps=FRAMERATE)