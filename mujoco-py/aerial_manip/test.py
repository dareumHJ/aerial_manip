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
# pi = np.pi
# data.qpos = [3*pi/2, -pi/2, pi/2, 3*pi/2, 3*pi/2, 0] #ENABLE if you want test circle

#Init parameters
# jacp = np.zeros((3, model.nv)) #translation jacobian
# jacr = np.zeros((3, model.nv)) #rotational jacobian
# print(model.nv)
# step_size = 0.5
# tol = 0.01
# alpha = 0.5
# damping = 0.001

#Get error.
# end_effector_id = model.body('wrist_3_link').id #"End-effector we wish to control.
# current_pose = data.body(end_effector_id).xpos #Current pose

# goal = [0.49, 0.13, 0.59] #Desire position

# error = np.subtract(goal, current_pose) #Init Error

def check_joint_limits(q):
    """Check if the joints is under or above its limits"""
    for i in range(len(q)):
        q[i] = max(model.jnt_range[i][0], min(q[i], model.jnt_range[i][1]))

def circle(t: float, r: float, h: float, k: float, f: float) -> np.ndarray:
    """Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    as a function of time t and frequency f."""
    x = r * np.cos(2 * np.pi * f * t) + h
    y = r * np.sin(2 * np.pi * f * t) + k
    z = 0.5
    return np.array([x, y, z])

#Simulate
with mujoco.viewer.launch_passive(model, data) as viewer:
    while data.time < DURATION:
        
        # goal = circle(data.time, 0.1, 0.5, 0.0, 0.5) #ENABLE to test circle.
        
        # if (np.linalg.norm(error) >= tol):
        #     #Calculate jacobian
        #     mujoco.mj_jac(model, data, jacp, jacr, goal, end_effector_id)
        #     #Calculate delta of joint q
        #     n = jacp.shape[1]
        #     I = np.identity(n)
        #     product = jacp.T @ jacp + damping * I

        #     if np.isclose(np.linalg.det(product), 0):
        #         j_inv = np.linalg.pinv(product) @ jacp.T
        #     else:
        #         j_inv = np.linalg.inv(product) @ jacp.T

        #     delta_q = j_inv @ error

        #     #Compute next step
        #     q = data.qpos.copy()
        #     q += step_size * delta_q
            
        #     #Check limits
        #     check_joint_limits(data.qpos)
            
        #     #Set control signal
        #     data.ctrl = q 
            #Step the simulation.
        data.ctrl = [3]
        mujoco.mj_step(model, data)

            # error = np.subtract(goal, data.body(end_effector_id).xpos)
        
        #Render and save frames.
        if len(frames) < data.time * FRAMERATE:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

        viewer.sync()
        
#Display video.
media.show_video(frames, fps=FRAMERATE)