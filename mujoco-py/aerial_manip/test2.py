import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media

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

#Init position.
pi = np.pi
data.qpos = [0, pi/2, -pi/2] #ENABLE if you want test circle

#Init parameters
M = np.zeros((model.nv, model.nv))
gain = 20
tol = 0.01
# alpha = 0.5
damping = 0.01

#Get error.
end_effector_id = model.body('E_E').id
current_pose = data.body(end_effector_id).xpos

goal = [0.07, -0.05, 0.28] #Desire position
goal2 = [-0.07, 0.05, 0.28]

error = np.subtract(goal, current_pose) #Init Error

prev_error = 0

def check_joint_limits(q):
    for i in range(len(q)):
        q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))

def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    x = r * np.cos(2 * np.pi * f * t) + h
    y = 0.04
    z = r * np.sin(2 * np.pi * f * t) + k
    return np.array([x, y, z])


#Simulate
with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:
        
        # goal = circle(data.time, 0.05, 0, 0.26, 3)
        if (np.linalg.norm(error) >= tol):

            error_d = error - prev_error

            #Calculate jacobian
            mujoco.mj_jac(model, data, jacp, jacr, goal, end_effector_id)
            #Calculate delta of joint q
            n = jacp.shape[1]
            I = np.identity(n)
            product = jacp.T @ jacp + damping * I

            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ jacp.T
            else:
                j_inv = np.linalg.inv(product) @ jacp.T

            delta_q = j_inv @ error

            error_d = error - prev_error

            #Compute next step
            q = data.qpos.copy()
            q += gain * delta_q + 3.0 * error_d / 0.005
            
            #Check limits
            # check_joint_limits(data.qpos)
            
            #Set control signal
            data.ctrl = q
            
            #Step the simulation.
            mujoco.mj_step(model, data)

            if (data.time < 5):
                error = np.subtract(goal, data.body(end_effector_id).xpos)
            else:
                error = np.subtract(goal2, data.body(end_effector_id).xpos)

            prev_error = error
        else:
            mujoco.mj_step(model, data)

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