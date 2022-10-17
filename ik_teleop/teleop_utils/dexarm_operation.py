import os
import cv2
import numpy as np
import yaml

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from hydra.utils import get_original_cwd

from ik_teleop.ik_core.allegro_controller import AllegroIKController
from ik_teleop.teleop_utils.calibrate import BoundCalibrator

from ik_teleop.utils.transformations import perform_persperctive_transformation

from copy import deepcopy as copy

import time
import torch
from polymetis import RobotInterface
import torchcontrol as toco
from typing import Dict
import numpy as np 

HAND_COORD_TOPIC = '/transformed_mediapipe_joint_coords'
CURR_JOINT_STATE_TOPIC = '/allegroHand/joint_states'

TRANS_HAND_TIPS = {
    'thumb': 6,
    'index': 7,
    'middle': 8,
    'ring': 9,
    'pinky': 10
}

# Policy class taken from examples/4_custom_updatable_controller.py
class MyPDPolicy(toco.PolicyModule):
    """
    Custom policy that performs PD control around a desired joint position
    """

    def __init__(self, joint_pos_current, kq, kqd, **kwargs):
        """
        Args:
            joint_pos_current (torch.Tensor):   Joint positions at initialization
            kq, kqd (torch.Tensor):             PD gains (1d array)
        """
        super().__init__(**kwargs)

        self.q_desired = torch.nn.Parameter(joint_pos_current)

        # Initialize modules
        self.feedback = toco.modules.JointSpacePD(kq, kqd)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        q_current = state_dict["joint_positions"]
        qd_current = state_dict["joint_velocities"]

        # Execute PD control
        output = self.feedback(
            q_current, qd_current, self.q_desired, torch.zeros_like(qd_current)
        )

        return {"joint_torques": output}

class DexArmOp(object):
    def __init__(self, allegro_bound_path, cfg = None):
        try:
            rospy.init_node('dexarm_operation')
        except:
            pass

        # ROS subscribers
        self.hand_coords = None
        self.current_joint_state = None
        rospy.Subscriber(HAND_COORD_TOPIC, Float64MultiArray, self._callback_hand_coords, queue_size = 1)
        rospy.Subscriber(CURR_JOINT_STATE_TOPIC, JointState, self._callback_curr_joint_state, queue_size = 1)

        # Initializing calibrator and performing calibration sequence
        self.calibrator = BoundCalibrator(storage_dir = get_original_cwd())

        print("***************************************************************")
        print("     Starting calibration process ")
        print("***************************************************************")
        self.calibrated_bounds = self.calibrator.check_and_load_calibration_file()

        with open(allegro_bound_path, 'r') as file:
            self.allegro_bounds = yaml.safe_load(file)['allegro_bounds']

        # Initializing IK solver
        self.ik_solver = AllegroIKController(cfg = cfg)
        self.cfg = cfg 

        print(self.ik_solver)

        # Initialize robot interface    
        self.robot = RobotInterface(ip_address="172.16.0.1", enforce_version = False)
        # Moving robot to home position
        self.robot.go_home()

        # Create policy instance
        q_initial = self.robot.get_joint_positions()
        default_kq = torch.Tensor(self.robot.metadata.default_Kq)
        default_kqd = torch.Tensor(self.robot.metadata.default_Kqd)
        # Send policy
        print("\nRunning PD policy...")

        policy = MyPDPolicy(
            joint_pos_current=q_initial,
            kq = default_kq,
            kqd = default_kqd,
        )

        self.time_to_go = 1.0
        self.hz = 50  # update frequency
        self.robot.send_torch_policy(policy, blocking=False)

        self.moving_average_queues = {
                'thumb': [],
                'index': [],
                'middle': [],
                'ring': []
            }

    def _callback_hand_coords(self, hand_coords):
        self.hand_coords = np.array(list(hand_coords.data)).reshape(11, 2)

    def _callback_curr_joint_state(self, curr_joint_state):
        self.current_joint_state = list(curr_joint_state.position)

    def get_finger_tip_data(self):
        finger_tip_coords = {}

        for key in TRANS_HAND_TIPS.keys():
            finger_tip_coords[key] = self.hand_coords[TRANS_HAND_TIPS[key]]

        return finger_tip_coords


    def go_to_goal(self, start, goal):
        total = int(self.time_to_go * self.hz)
        for t in range(total):
            interp_state = ((total-t)/(total-1)) * start + ((t-1)/(total-1)) * goal
            self.robot.update_current_policy({"q_desired": interp_state})
            time.sleep(1 / self.hz)

    def move(self):
        print("\n******************************************************************************")
        print("     Controller initiated. ")
        print("******************************************************************************\n")
        print("Start controlling the robot hand using the tele-op.\n")

        # if you need to visualize
        import ikpy.utils.plot as plot_utils
        import matplotlib.pyplot as plt
        
        fig, ax = plot_utils.init_3d_figure()
        
        position = {}
        while True:
            finger_types = ["index", "middle", "ring"]

            self.current_joint_state = self.robot.get_joint_positions()
            if self.hand_coords is not None and self.current_joint_state is not None:
                finger_tip_coords = self.get_finger_tip_data()
                desired_joint_angles = copy(self.current_joint_state)

                # Movement for index finger
                position["index"], desired_joint_angles = self.ik_solver.bounded_linear_finger_motion(
                    "index", 
                    self.allegro_bounds['height'], # X value 
                    self.allegro_bounds['index']['y'], # Y value
                    finger_tip_coords['index'][1], # Z value
                    self.calibrated_bounds[0], 
                    self.allegro_bounds['index']['z_bounds'], 
                    self.moving_average_queues['index'], 
                    desired_joint_angles
                )

                # # Movement for the Middle finger
                position["middle"], desired_joint_angles = self.ik_solver.bounded_linear_finger_motion(
                    "middle", 
                    self.allegro_bounds['height'], # X value 
                    self.allegro_bounds['middle']['y'], # Y value
                    finger_tip_coords['middle'][1], # Z value
                    self.calibrated_bounds[1], 
                    self.allegro_bounds['middle']['z_bounds'], 
                    self.moving_average_queues['middle'], 
                    desired_joint_angles
                )

                # Movement for the Ring finger using the ring and pinky fingers
                position["ring"], desired_joint_angles = self.ik_solver.bounded_linear_fingers_motion(
                    "ring", 
                    self.allegro_bounds['height'], 
                    finger_tip_coords['pinky'][1], 
                    finger_tip_coords['ring'][1], 
                    self.calibrated_bounds[3], # Pinky bound for Allegro Y axis movement
                    self.calibrated_bounds[2], # Ring bound for Allegro Z axis movement
                    self.allegro_bounds['ring']['y_bounds'], 
                    self.allegro_bounds['ring']['z_bounds'], 
                    self.moving_average_queues['ring'], 
                    desired_joint_angles
                )

                # Movement for the Thumb
                if cv2.pointPolygonTest(np.float32(self.calibrated_bounds[4:]), np.float32(finger_tip_coords['thumb'][:2]), False) > -1:
                    transformed_thumb_coordinate = perform_persperctive_transformation(
                        finger_tip_coords['thumb'],
                        self.calibrated_bounds[4 : ],
                        self.allegro_bounds['thumb'],
                        self.allegro_bounds['height']
                    )

                    position["thumb"], desired_joint_angles = self.ik_solver.bounded_finger_motion(
                        "thumb",
                        transformed_thumb_coordinate,
                        self.moving_average_queues['thumb'],
                        desired_joint_angles
                    )
                    # finger_types += ["thumb"]

                # if you need to visualize
                for finger_type in finger_types:
                    current_finger_position = position[finger_type]
                    my_chain = self.ik_solver.allegro_ik.chains[finger_type]
                    my_chain.plot(my_chain.inverse_kinematics(current_finger_position), ax, target = current_finger_position)
                    ax.scatter3D(current_finger_position[0], current_finger_position[1], current_finger_position[2])
                plt.xlim(-0.1, 0.1)
                plt.ylim(-0.1, 0.1)
                ax.set_zlim(-0.1, 0.1)

                plt.draw()
                plt.pause(0.01)
                ax.cla()
                # plt.close()

                # Publishing the required joint angles
                self.go_to_goal(start = self.current_joint_state, goal = desired_joint_angles)
                # print(f"current_joint_state: {self.current_joint_state}\ndesired joint state: {desired_joint_angles}")
                if cv2.waitKey(5) & 0xFF == 27:
                    print("Terminating PD policy...")
                    state_log = self.robot.terminate_current_policy()
                    break
