import numpy as np
import mujoco
import mujoco.viewer
import mediapy as media

xml = "WALKON5_left_leg.xml"
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

#Video Setup
DURATION = 4 #(seconds)
FRAMERATE = 60 #(Hz)
frames = []

#Reset state and time.
mujoco.mj_resetData(model, data)

#Init position.
# pi = np.pi
# data.qpos = [3*pi/2, -pi/2, pi/2, 3*pi/2, 3*pi/2, 0] #ENABLE if you want test circle

#Init parameters
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))

step_size = 0.5
tol = 0.01
alpha = 0.5
damping = 0.001

#Get error.
end_effector_id = model.body('L_FOOT').id #"End-effector we wish to control.
current_pose = data.body(end_effector_id).xpos #Current pose

goal = [0.1, 0.1, 0.0] #Desire position
# goal = current_pose

error = np.subtract(goal, current_pose) #Init Error

def check_joint_limits(q):
    """Check if the joints is under or above its limits"""
    for i in range(len(q)):
        q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))

#Simulate
with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:
        
        # goal = circle(data.time, 0.1, 0.5, 0.0, 0.5) #ENABLE to test circle.
        if (np.linalg.norm(error) >= tol):
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
            #Compute next step

            q = data.qpos.copy()
            q += step_size * delta_q

            #Check limits
            check_joint_limits(data.qpos)
            
            #Set control signal
            data.ctrl = q 
            #Step the simulation.
            mujoco.mj_step(model, data)

            error = np.subtract(goal, data.body(end_effector_id).xpos)
        #Render and save frames.
        if len(frames) < data.time * FRAMERATE:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

            # tweak scales of contact visualization elements
            model.vis.scale.contactwidth = 0.05
            model.vis.scale.contactheight = 0.015
            model.vis.scale.forcewidth = 0.025
            model.vis.map.force = 0.15

        viewer.sync()
        
#Display video.
media.show_video(frames, fps=FRAMERATE)