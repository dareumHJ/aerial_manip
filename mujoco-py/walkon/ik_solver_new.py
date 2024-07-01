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
FRAMERATE = 60 #(Hz)
frames = []

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

step_size = 0.1
tol = 0.01
alpha = 0.5
damping = 0.001

def compute_com(model, data):
    total_mass = np.sum(model.body_mass)

    mass = model.body_mass
    xpos = data.xipos

    com = np.sum(xpos.T * mass, axis=1) / total_mass
    return com

#Get error.
end_effector_id = model.body('L_ANKLE').id #"End-effector we wish to control.
foot_id = model.body('L_FOOT').id
current_pose = data.body(end_effector_id).xpos #Current pose

mujoco.mj_resetData(model, data)
qpos0 = data.qpos.copy()
nv = model.nv

mujoco.mj_forward(model, data)
jac_com = np.zeros((3, nv))
mujoco.mj_jacSubtreeCom(model, data, jac_com, end_effector_id)

jac_foot = np.zeros((3, nv))
mujoco.mj_jacBodyCom(model, data, jac_foot, None, foot_id)

jac_diff = jac_com - jac_foot
Qbal = jac_diff.T @ jac_diff

des_qpos = 0*np.pi/180

# find err value
joint_pos = data.body(foot_id).xipos.copy()
com = compute_com(model, data)
det_vec = com - joint_pos
unit_vec = [0, 0, 1]
err = des_qpos-math.acos(np.dot(det_vec, unit_vec) / det_vec.size)

Ks = 200.0
Bs = 200.0

# dq = np.zeros((1, 3))

prev_err = 0.0
errdiff = (err - prev_err) / step_size

with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:
        # mujoco.mj_differentiatePos(model, dq, step_size, qpos0, data.qpos)

        joint_pos = data.body(foot_id).xpos.copy()
        com = compute_com(model, data)
        det_vec = com - joint_pos
        err = des_qpos - math.acos(np.dot(det_vec, unit_vec) / np.linalg.norm(det_vec))
        # err = (des_qpos - (data.qpos.copy())[12])
        errdiff = 0.0 - (data.qvel)[10]

        #Set control signal
        ctrlval = Ks * err + Bs * errdiff
        # ctrlval = 20
        data.ctrl = [0, 0, 0, 0, ctrlval]
        #Step the simulation.
        mujoco.mj_step(model, data)

        prev_err = err

        # print("Center of Mass:", com)
        print(ctrlval)

        if len(frames) < data.time * FRAMERATE:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
        viewer.sync()

media.show_video(frames, fps=FRAMERATE)