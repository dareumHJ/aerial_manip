# Do Inverse Kinematics once then Control...

import numpy as np
import mujoco_py
import mujoco
import mujoco.viewer
import mediapy as media
import matplotlib.pyplot as plt

xml = "ur3e.xml"
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

#Video Setup
DURATION = 20 #(seconds)
FRAMERATE = 60 #(Hz)
frames = []

#Reset state and time.
mujoco.mj_resetData(model, data)

# Init position.
pi = np.pi
data.qpos = [0, pi/2, -pi/2] #ENABLE if you want test circle

# Init parameters
M = np.zeros((model.nv, model.nv))
jacp = np.zeros((3, model.nv)) #translation jacobian
jacr = np.zeros((3, model.nv)) #rotational jacobian
dt = 0.005
tol = 0.001
damping = 0.0
Kp = 100.0
Kd = 10.0

end_effector_id = model.body('E_E').id
current_pose = data.body(end_effector_id).xpos

#Goal Position set
goal = [-0.13, 0.0, 0.264] #Desire position
goal2 = [-0.08, 0.05, 0.28]

#Error initialization
error = np.subtract(goal, current_pose) #Init Error
prev_error = 0

#Function Definitions
def check_joint_limits(q):
    for i in range(len(q)):
        q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))

def inv_kinematics(model, data, jacp, jacr, goal, end_effector_id, error):
    #Calculate jacobian
    mujoco.mj_jac(model, data, jacp, jacr, goal, end_effector_id)
    n = jacp.shape[1]
    I = np.identity(n)
    product = jacp.T @ jacp + damping * I
    if np.isclose(np.linalg.det(product), 0):
        j_inv = np.linalg.pinv(product) @ jacp.T
    else:
        j_inv = np.linalg.inv(product) @ jacp.T
    delta_q = j_inv @ error
    return delta_q

# Get goal qpos using iterative inverse kinematics
while (np.linalg.norm(error) > tol):
    dq = inv_kinematics(model, data, jacp, jacr, goal2, end_effector_id, error)
    q = data.qpos.copy()
    q += dq * 0.05
    data.qpos = q
    mujoco.mj_forward(model, data)
    error = goal2 - data.body(end_effector_id).xpos
print(data.body(end_effector_id).xpos)
print(np.linalg.norm(error))
q_goal = data.qpos.copy()

# reset
mujoco.mj_resetData(model, data)
data.qpos = [0, pi/2, -pi/2]

# new jacobian
_jacp = np.zeros((3, model.nv))
_jacr = np.zeros((3, model.nv))

# new initialize
_error = 0
_prev_error = 0

#Simulate
with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:
        mujoco.mj_fullM(model, M, data.qM)
        cg = data.qfrc_bias

        # calculate error and error_d
        error = np.subtract(q_goal, data.qpos)
        error_d = (error - prev_error)/dt
        
        # new error
        # _error = np.subtract(goal, data.body(end_effector_id).xpos)
        # _error_d = (_error - _prev_error)/dt
        #Check limits
        # check_joint_limits(data.qpos)
        
        ctrlval = M @ (Kp * error + Kd * error_d)

        # mujoco.mj_jacBody(model, data, _jacp, _jacr, end_efFector_id)
        # ctrlval = _jacp.T @ (-Kp * _error - Kd * _error_d)
        #Set control signal
        data.ctrl = cg + ctrlval

        print(np.linalg.norm(error))

        #Step the simulation.
        mujoco.mj_step(model, data)
        #Update previous error
        prev_error = error
        # _prev_error - _error

        #Render and save frames.
        if len(frames) < data.time * FRAMERATE:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

            model.vis.scale.contactwidth = 0.05
            model.vis.scale.contactheight = 0.015
            model.vis.scale.forcewidth = 0.025
            model.vis.map.force = 0.15

        viewer.sync()
        
#Display video.
media.show_video(frames, fps=FRAMERATE)