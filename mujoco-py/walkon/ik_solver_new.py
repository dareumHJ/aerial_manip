import numpy as np
import mujoco
import mujoco.viewer as viewer
import mediapy as media

# Choose a model
xml = "WALKON5_left_leg.xml"
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

#Video Setup
DURATION = 4 #(seconds)
FRAMERATE = 60 #(Hz)
frames = []

jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

step_size = 0.1
tol = 0.01
alpha = 0.5
damping = 0.001

#Get error.
end_effector_id = model.body('L_ANKLE').id #"End-effector we wish to control.
foot_id = model.body('L_FOOT').id
current_pose = data.body(end_effector_id).xpos #Current pose

goal = [-0.2, -0.2, 0.3] #Desire position

error = np.subtract(goal, current_pose) #Init Error

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
err = (des_qpos - (data.qpos.copy())[12])

Kp = 100.0
Kd = 20.0

prev_err = 0.0
errdiff = (err - prev_err) / step_size

with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:
        # mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)

        err = (des_qpos - (data.qpos.copy())[12])
        errdiff = (err - prev_err) / step_size

        #Set control signal
        ctrlval = Kp * err + Kd * errdiff
        data.ctrl = [ctrlval]
        #Step the simulation.
        mujoco.mj_step(model, data)

        prev_err = err

        print(err)
        if len(frames) < data.time * FRAMERATE:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
        viewer.sync()

media.show_video(frames, fps=FRAMERATE)