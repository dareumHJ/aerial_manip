import numpy as np
import mujoco
import mujoco.viewer as viewer
import mediapy as media
import math

# Choose a model
xml = "test_leg2.xml"
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
des_qpos1 = 80*np.pi/180
des_qpos2 = 90*np.pi/180

# des_qpos1 = 0*np.pi/180
# des_qpos2 = 0*np.pi/180

unit_vec1 = np.array([1, 0, 0])
unit_vec2 = np.array([0, 1, 0])
unit_vec3 = np.array([0, 0, 1])
norm_vec1 = np.array([1, 0, 1])
norm_vec1 = norm_vec1 / np.linalg.norm(norm_vec1)

Ks = 50.0
Bs = 4.0
Ks2 = 50.0
Bs2 = 4.0

plant_idx = (data.qpos.size) - 1
inver_idx = (data.qpos.size) - 2

# dq = np.zeros((1, 3))

with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:
        # mujoco.mj_differentiatePos(model, dq, step_size, qpos0, data.qpos)

        # ROTATION MATRIX (Global -> Local)
        mat1 = (data.body(model.body('L_FOOT').id).xmat.copy()).reshape(3, 3)
        mat2 = (data.body(model.body('L_ANKLE').id).xmat.copy()).reshape(3, 3)


        joint1_pos = data.body(model.body('L_FOOT').id).xpos.copy() + mat1 @ np.array([-0.15552, 0.17526, -0.9975])
        joint2_pos = data.body(model.body('L_ANKLE').id).xpos.copy() + mat2 @ np.array([-0.06603, 0.10476, -0.90801])
        com = compute_com(model, data)
        det_vec1 = com - joint1_pos
        det_vec2 = com - joint2_pos


        proj_vec1 = np.linalg.inv(mat1) @ det_vec1
        proj_vec1 = proj_vec1 - (np.dot(proj_vec1, unit_vec2)) * unit_vec2

        proj_vec2 = np.linalg.inv(mat2) @ det_vec2
        proj_vec2 = proj_vec2 - (np.dot(proj_vec2, norm_vec1)) * norm_vec1
        
        cur_pos1 = math.acos(np.dot((proj_vec1), unit_vec1) / np.linalg.norm(proj_vec1))
        cur_pos2 = math.acos(np.dot((proj_vec2), unit_vec2) / np.linalg.norm(proj_vec2))
        err1 = -(des_qpos1 - cur_pos1)
        # err1 = des_qpos1 - (data.qpos.copy())[plant_idx]
        err2 = -(des_qpos2 - cur_pos2)
        # err2 = des_qpos2 - (data.qpos.copy())[inver_idx]

        errdiff1 = 0.0 - (data.qvel.copy())[plant_idx - 1]
        errdiff2 = 0.0 - (data.qvel.copy())[inver_idx - 1]

        #Set control signal
        ctrlval1 = (Ks * err1 + Bs * errdiff1)
        ctrlval2 = (Ks2 * err2 + Bs2 * errdiff2)
        data.ctrl = [0, ctrlval2, ctrlval1]

        if (data.time > 2 and data.time < 5):
            data.xfrc_applied[2] = [0, 0, 0, 0, 0, 0]
            
        else:
            data.xfrc_applied[2] = [0, 0, 0, 0, 0, 0]
        # Step the simulation.
        mujoco.mj_step(model, data)

        print("---------joint pos and com-----------")
        # print(cur_pos1)
        # print(des_qpos1)
        # print(ctrlval1)
        # print((data.qpos.copy())[plant_idx])
        print(cur_pos2 * 180 / np.pi)
        print(err2)
        print(ctrlval2)
        print("-------------------------------------")

        
        viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0.03, 0],
            pos=(mat2 @ proj_vec2),
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 1])
        )
        viewer.user_scn.ngeom += 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0.03, 0],
            pos = [0, 0, 0],
            # pos=com,
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 1])
        )
        viewer.user_scn.ngeom += 1

        if len(frames) < data.time * FRAMERATE:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
        viewer.sync()

        with viewer.lock():
            model.vis.scale.com = 0.1
            # model.vis.scale.frames = 0.1

media.show_video(frames, fps=FRAMERATE)