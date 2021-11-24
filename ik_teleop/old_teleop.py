# Standard imports
import os
import numpy as np

# Parameter management imports
from hydra import initialize, compose

# Image based imports
import cv2
import mediapipe

# ROS imports
import rospy
from sensor_msgs.msg import JointState

# ROS wrapper to publish joint angles
from move_dexarm import DexArmControl

# Allegro Inverse Kinematics based controller
from ik_teleop.ik_core.allegro_controller import AllegroIKController

# Other utility imports
import utils.camera as camera
import utils.joint_handling as joint_handlers
from utils.transformations import perform_persperctive_transformation

# Debugging imports
from IPython import embed

class TeleOperation (object):
    def __init__(self, cfg = None, urdf_path = os.path.join(os.getcwd(), "urdf_template", "allegro_right.urdf"), rotation_angle = 0, enable_moving_average = True):
        # Getting the configurations
        if cfg is None:
            initialize(config_path = "./parameters/")
            self.cfg = compose(config_name = "teleop")
        else:
            self.cfg = cfg

        # Initializing a ROS node
        try:
            rospy.init_node('teleop_node')
        except:
            pass

        # Creating a realsense pipeline
        self.pipeline, config = camera.create_realsense_pipeline(self.cfg.realsense.serial_numbers[0], self.cfg.realsense.resolution, self.cfg.realsense.fps)

        self.pipeline.start(config)

        # Creating mediapipe objects
        self.mediapipe_drawing = mediapipe.solutions.drawing_utils
        self.mediapipe_hands = mediapipe.solutions.hands

        # Initializing the Alegro Inverse Kinematics based controller
        self.allegro_control = AllegroIKController(self.cfg.allegro, urdf_path = urdf_path)

        # Arm controller
        self.arm_controller = DexArmControl()
        
        # Homing the robot
        self.arm_controller.home_robot()

        # Initializing a current joint state variable and creating a subscriber to get the current allegro joint angles
        self.current_joint_state = None
        rospy.Subscriber(self.cfg.allegro.joint_angle_topic, JointState, self._callback_current_joint_state, queue_size = 1)

        # Moving average arrays
        self.enable_moving_average = enable_moving_average
        if self.enable_moving_average is True:
            self.moving_average_queues = {
                'thumb': [],
                'index': [],
                'middle': [],
                'ring': []
            }

        # Storing the specified camera rotation
        self.rotation_angle = rotation_angle

    def _callback_current_joint_state(self, data):
        self.current_joint_state = data

    def check_bounds_to_compute_angles(self, wrist_position, pinky_knuckle_position, finger_tip_positions):
        # Check if the wrist position and pinky knuckle position are aligned
        if joint_handlers.check_hand_position(wrist_position, 
            pinky_knuckle_position, 
            self.cfg.mediapipe.contours.wrist_circle, 
            self.cfg.mediapipe.contours.pinky_knuckle_bounds):

            # TODO
            # Updating the index finger motion based on the mediapipe index finger position
            # Updating the middle fingger motion based on the mediapipe middle finger position

            # Updating the allegro ring finger angles based on the mediapipe ring finger and pinky values
            desired_angles = self.allegro_control.bounded_linear_fingers_motion(
                'ring', 
                self.cfg.allegro_bounds.height,         # x value for allegro hand
                finger_tip_positions['pinky'][1],       # y value for the allegro hand
                finger_tip_positions['ring'][1],        # z value for the allegro hand
                self.cfg.mediapipe_bounds.pinky_linear, 
                self.cfg.mediapipe_bounds.ring_linear, 
                self.cfg.allegro_bounds.ring[0],  
                self.cfg.allegro_bounds.ring[1], 
                self.moving_average_queues['ring'], 
                list(self.current_joint_state.position))

            # Checking if the thumb is inside the bound
            if cv2.pointPolygonTest(np.array(self.cfg.mediapipe_bounds.thumb), finger_tip_positions['thumb'], False) > -1:
                # Getting the transformed thumb tip coordinate using projective transformation on the finger tip coordinate
                transformed_thumb_coordinate = perform_persperctive_transformation(
                    finger_tip_positions['thumb'], 
                    self.cfg.mediapipe_bounds.thumb,
                    self.cfg.allegro_bounds.thumb,
                    self.cfg.allegro_bounds.height)

                # Finding the desired thumb angles and updating them on the ring finger angles using the bounded inverse kinematics finction
                desired_angles = self.allegro_control.bounded_finger_motion(
                    'thumb', 
                    transformed_thumb_coordinate, 
                    self.moving_average_queues['thumb'], 
                    desired_angles)

            return desired_angles

        print('Hand not inside bound!')
        return 

    def hand_movement_processor(self):
        # Setting the mediapipe hand parameters
        with self.mediapipe_hands.Hands(
            max_num_hands = 1, # Limiting the number of hands detected in the image to 1
            min_detection_confidence = 0.95,
            min_tracking_confidence = 0.95) as hand:

            while True:
                # Getting the image to process
                image = camera.getting_image_data(self.pipeline)

                if image is None:
                    print('Did not receive an image. Please wait!')
                    continue

                # Rotate image if needed
                if self.rotation_angle != 0:
                    image = camera.rotate_image(image, self.rotation_angle)

                # Getting the hand pose results out of the image
                image.flags.writeable = False
                estimate = hand.process(image)
                image.flags.writeable = True

                # Converting the image back from RGB to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Create all the bounds in the image
                image = camera.create_contours(
                    image, 
                    self.cfg.mediapipe.contours.wrist_circle, 
                    self.cfg.mediapipe.contours.pinky_knuckle_bounds, 
                    self.cfg.mediapipe.contours.thumb_tip_bounds,
                    self.cfg.mediapipe.contours.thickness
                )

                # If there is a mediapipe hand estimate
                if estimate.multi_hand_landmarks is not None:  

                    # Getting the hand coordinate values for the only detected hand
                    hand_landmarks = estimate.multi_hand_landmarks[0]

                    # Embedding the hand drawind in the image
                    self.mediapipe_drawing.draw_landmarks(
                            image, hand_landmarks, self.mediapipe_hands.HAND_CONNECTIONS)

                    if self.current_joint_state is not None:
                        # Getting the mediapipe wrist and fingertip positions
                        wrist_position, thumb_knuckle_position, index_knuckle_position, middle_knuckle_position, ring_knuckle_position, pinky_knuckle_position, finger_tip_positions = joint_handlers.get_joint_positions(hand_landmarks, self.cfg.realsense.resolution, self.cfg.mediapipe)

                        # Getting the desired angles based on the current joint and tip positions
                        desired_angles = self.check_bounds_to_compute_angles(wrist_position, 
                                pinky_knuckle_position, 
                                finger_tip_positions)

                        if desired_angles is not None:
                            # Using the arm controller to publish the desired angles
                            self.arm_controller.move_hand(desired_angles)

                # Printing the image
                cv2.imshow('Teleop - Mediapipe screen', image)

                # Condition to break the loop incase of keyboard interrupt
                if cv2.waitKey(5) & 0xFF == 27:
                    break

if __name__ == '__main__':
    teleop = TeleOperation(rotation_angle = 180)
    # embed()
    teleop.hand_movement_processor()
