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
DURATION = 10 #(seconds)
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
des_qpos1 = 100*np.pi/180
# des_qpos2 = 90*np.pi/180

# des_qpos1 = 0*np.pi/180
des_qpos2 = 0*np.pi/180
des_qpos2 = 0*np.pi/180

# find err value
joint1_pos = data.body(model.body('L_FOOT').id).xipos.copy()
joint2_pos = data.body(model.body('L_ANKLE').id).xipos.copy()
com = compute_com(model, data)
unit_vec1 = [0, -1, 0]
unit_vec2 = [1, 0, 1]
unit_vec2 = unit_vec2 / np.linalg.norm(unit_vec2)

Ks = 300.0
Bs = 50.0
Ks2 = 1000.0
Bs2 = 0.0

# dq = np.zeros((1, 3))

with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:
        # mujoco.mj_differentiatePos(model, dq, step_size, qpos0, data.qpos)

        joint1_pos = data.body(model.body('L_FOOT').id).xipos.copy()
        joint2_pos = data.body(model.body('L_ANKLE').id).xipos.copy()
        com = compute_com(model, data)
        det_vec1 = com - joint1_pos
        det_vec2 = com - joint2_pos
        err1 = des_qpos1 - math.acos(np.dot(det_vec1, unit_vec1) / np.linalg.norm(det_vec1))
        # err1 = des_qpos1 - (data.qpos.copy())[11]
        err2 = des_qpos2 - math.acos(np.dot(det_vec2, unit_vec2) / np.linalg.norm(det_vec2))
        # err2 = des_qpos2 - (data.qpos.copy())[10]
        # err = (des_qpos - (data.qpos.copy())[12])
        errdiff1 = 0.0 - (data.qvel.copy())[10]
        errdiff2 = 0.0 - (data.qvel.copy())[9]

        #Set control signal
        ctrlval1 = (Ks * err1 + Bs * errdiff1)
        ctrlval2 = (Ks2 * err2 + Bs2 * errdiff2)
        ctrlval2 = 0
        data.ctrl = [0, 0, 0, ctrlval2, ctrlval1]

        if (data.time > 3):
            data.xfrc_applied = [0, 0, 0, 0, 0, 0]
        else:
            data.xfrc_applied = [0, 0, 0, 0, 0, 0]
        #Step the simulation.
        mujoco.mj_step(model, data)

        print("---------joint pos and com-----------")
        print((data.qpos.copy())[10])
        print(err2)
        print(ctrlval2)

        if len(frames) < data.time * FRAMERATE:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
        viewer.sync()

media.show_video(frames, fps=FRAMERATE)