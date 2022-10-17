# Standard imports
import numpy as np

# Parameter management imports
from hydra.experimental import initialize, compose

# Imports to get the current joint state
import rospy 
from sensor_msgs.msg import JointState

# Allegro Inverse Kinematics Library
from ik_teleop.ik_core.allegro_ik import AllegroInvKDL

# Debugging imports
from IPython import embed

class FingerState (object):
    def __init__(self, cfg = None, frequency = 10, urdf_path = None):
        # To ignore scientific notations
        np.set_printoptions(suppress = True)

        # Importing configs from yaml file
        if cfg is None:
            initialize(config_path = "../parameters/allegro")
            self.cfg = compose(config_name = "allegro_link_info")
        else:
            self.cfg = cfg

        # Initializing a ROS node
        try:
            rospy.init_node('allegro_finger_debugger')
        except:
            pass

        # Creating an IK object for the allegro hand
        self.allegro_ik = AllegroInvKDL(self.cfg, urdf_path = urdf_path)

        # ROS stuff to get the current joint state of the Allegro hand
        self.current_joint_state = None
        rospy.Subscriber(self.cfg.joint_angle_topic, JointState, self._callback_current_joint_state, queue_size = 1)
        
        # Setting a frequency for the motion
        self.rate = rospy.Rate(frequency)

    def _callback_current_joint_state(self, data):
        self.current_joint_state = data

    def finger_coordinate_state(self, finger_type):
        while True:
            if self.current_joint_state is not None:
                # Getting the current set of angles from the joint state rostopic
                current_angles = list(self.current_joint_state.position)[self.cfg.fingers[finger_type].offset : self.cfg.fingers[finger_type].offset + self.cfg.joints_per_finger]

                # Getting the translation and rotation values using the forward kinematics function
                translation, rotation = self.allegro_ik.finger_forward_kinematics(finger_type, current_angles)
                print('{} position: {}'.format(finger_type, translation))

            self.rate.sleep


if __name__ == '__main__':
    allegro = FingerState()
    embed()