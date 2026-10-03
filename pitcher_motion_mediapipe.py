import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import math

# Preprocessing
# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, # it's for video, not static image
                    model_complexity=2, # 0: light, 1: heavy, 2: very heavy
                    min_detection_confidence=0.85, # Confidence threshold for detection
                    min_tracking_confidence=0.85) # Confidence threshold for tracking

# Function to calculate distance between two points
def distance_computing (x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def process (video_path, left_handed=False):
    cap = cv2.VideoCapture(video_path) # Load video file
    print("Video loaded:", video_path)

    # Get video frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0

    # for video saving
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if h > 720 or w > 1280: # the frame would resize to smaller size if it's too big
        w = w * 2 // 3
        h = h * 2 // 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter('output.mp4', fourcc, fps, (w,h), True)
    output_frames = []

    # Initialize variables
    # for foot
    frame_nums = []
    contact_frame_num = 0
    left_foot_xs = []
    left_foot_ys = []
    left_foot_prev_x = 0
    left_foot_prev_y = 0
    left_foot_status = "Start"
    pitcher_leg_length = 0

    left_foot_velocity_xs = []
    left_foot_velocity_ys = []

    left_foot_highest_x = 0
    left_foot_highest_y = 0

    right_hip_init_x = 0
    right_hip_init_y = 0
    right_heel_init_x = 0
    right_heel_init_y = 0

    # for upper body
    right_elbow_xs = []
    right_elbow_ys = []
    right_wrist_xs = []
    right_wrist_ys = []
    right_forearem_angles = []
    max_external_rotation_angle = 0
    max_external_rotation_frame_num = 0
    external_rotation_flag = False
    
    release_flag = False
    right_arm_angle_difference = []
    release_frame_num = 0

    leg_angles = []

    center_mass_xs = []
    center_mass_ys = []

    # Flags for play/pause control
    paused = False

    while cap.isOpened():
        if not paused:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret: # Exit when video ends
                break

            if left_handed:
                frame = cv2.flip(frame, 1)

            # Caulculate time in seconds
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Show a processing video
            proc_img = np.zeros((100, 600, 3), dtype=np.uint8)
            proc_img_text = f"Processing frame {frame_number}/{total_frames}"
            cv2.putText(proc_img, proc_img_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
            cv2.namedWindow('Processing', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Processing', 600, 100)
            cv2.imshow('Processing', proc_img)
            
            # Resize frame to smaller size if it's too big
            original_height, original_width = frame.shape[:2]
            if original_height > 720 or original_width > 1280:
                frame = cv2.resize(frame, (original_width * 2 // 3, original_height * 2 // 3))
            height, width, _ = frame.shape
            last_frame = frame.copy()

            # Process the frame with MediaPipe Pose
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame to RGB
            result = pose.process(rgb_frame) # Process the frame with MediaPipe Pose

            left_foot_x, left_foot_y = 0, 0

            # Check pitcher leg length
            if pitcher_leg_length == 0:
                if result.pose_landmarks:
                    # Calculate pitcher leg length (distance between right heel and right hip)
                    right_heel_init_x = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x * width
                    right_hip_init_x = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * width
                    right_heel_init_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y * height
                    right_hip_init_y = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * height
                    pitcher_leg_length = distance_computing(right_heel_init_x, right_heel_init_y, right_hip_init_x, right_hip_init_y)
                    pitcher_leg_length = int(pitcher_leg_length) # Convert to integer
            
            frame_nums.append(frame_number)
            # Draw landmarks if a person is detected, and process the frame
            if result.pose_landmarks:
                # Draw pose landmarks on the frame
                mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                        mp_draw.DrawingSpec(color=(0, 100, 255), thickness=0, circle_radius=0),
                                        mp_draw.DrawingSpec(color=(200, 200, 180), thickness=2))

                # Get the body positions
                nose_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * width)
                right_elbow_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * width)
                right_elbow_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * height)
                right_wrist_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * width)
                right_wrist_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height)
                right_shoulder_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width)
                right_shoulder_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height)
                right_knee_x =  int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * width)
                left_foot_index_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * width)
                left_foot_index_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * height)
                right_foot_index_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * width)
                right_foot_index_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * height)
                left_heel_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x * width)
                left_heel_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y * height)
                right_heel_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x * width)
                right_heel_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y * height)
                left_hip_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * width)
                left_hip_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * height)
                right_hip_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * width)
                right_hip_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * height)
                left_knee_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * width)
                left_knee_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * height)
                left_ankle_x = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * width)
                left_ankle_y = int(result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * height)

                # Define left foot position and right foot position = (foot index + heel) / 2
                left_foot_x = int((left_foot_index_x + left_heel_x) / 2)
                left_foot_y = int((left_foot_index_y + left_heel_y) / 2)
                right_foot_x = int((right_foot_index_x + right_heel_x) / 2)
                right_foot_y = int((right_foot_index_y + right_heel_y) / 2)

                if (left_foot_prev_x != 0 and left_foot_prev_y != 0):
                    # Calculate the distance between the current and previous left foot positions
                    distance = distance_computing(left_foot_x, left_foot_y, left_foot_prev_x, left_foot_prev_y)
                    if distance > pitcher_leg_length / 2:
                        # Drop this point if it moves too far
                        left_foot_x = left_foot_prev_x
                        left_foot_y = left_foot_prev_y

                # Append left foot position to the list
                left_foot_xs.append(left_foot_x)
                left_foot_ys.append(left_foot_y)

                # Compute velocity
                if left_foot_prev_x != 0 and left_foot_prev_y != 0:
                    velocity_x = (left_foot_x - left_foot_prev_x) / (1 / fps)
                    velocity_y = (left_foot_y - left_foot_prev_y) / (1 / fps)
                    left_foot_velocity_xs.append(velocity_x)
                    left_foot_velocity_ys.append(velocity_y)
                else:
                    left_foot_velocity_xs.append(None)
                    left_foot_velocity_ys.append(None)

                # if status is contact, not release and all velocity of the last 5 frames are positive then it's not contact
                if (left_foot_status == "contact" and not(release_flag) 
                    and (all(v > 0 for v in left_foot_velocity_ys[-6:-1]) or all(v > 0 for v in left_foot_velocity_xs[-6:-1])) 
                    and distance_computing(left_foot_x, left_foot_y, left_foot_xs[-5], left_foot_ys[-5]) > 5
                    and frame_number - contact_frame_num < 20):
                    left_foot_status = "almost contact"
                    max_external_rotation_angle = 0
                    print(f"frame {frame_number}: contact -> almost contact")
                    print(left_foot_velocity_xs[-6:-1])
                    print(left_foot_velocity_ys[-6:-1])
                    print(distance_computing(left_foot_x, left_foot_y, left_foot_xs[-5], left_foot_ys[-5]))


                # Find when the foot contact the ground
                if (left_foot_status != "contact"):
                    # set initial values for left foot position
                    if left_foot_highest_x == 0 and left_foot_highest_y == 0:
                        left_foot_highest_x = left_foot_x
                        left_foot_highest_y = left_foot_y
                    elif left_foot_y < left_foot_highest_y:
                        left_foot_highest_x = left_foot_x
                        left_foot_highest_y = left_foot_y
                        left_foot_status = "lifting"
                    elif left_foot_status == "lifting" and left_foot_y > left_foot_highest_y and distance_computing(left_foot_highest_x, left_foot_highest_y, left_foot_x, left_foot_y) > (pitcher_leg_length / 2):
                        left_foot_status = "dropping"

                    if (left_foot_status == "dropping" or left_foot_status == "almost contact") and len(left_foot_velocity_xs) > 10 and all(v != None for v in left_foot_velocity_xs[-11:-1]):
                        # if velocity of last 10 frames are positive, and now is 0, then it's contact 
                        if all(v >= 0 for v in left_foot_velocity_xs[-11:-1]) and velocity_x <= 0 and (left_foot_x - right_foot_x) > pitcher_leg_length:
                            if velocity_y <= 0:
                                left_foot_status = "contact"
                                contact_frame_num = frame_number-2
                                if max_external_rotation_frame_num < contact_frame_num: max_external_rotation_angle = 0
                            else:
                                left_foot_status = "almost contact"

                        elif left_foot_status == "almost contact" and velocity_y <= 0:
                            left_foot_status = "contact"
                            contact_frame_num = frame_number-2 if velocity_x <= 0 else frame_number-1
                            if max_external_rotation_frame_num < contact_frame_num: max_external_rotation_angle = 0

                        if left_foot_status == "contact":
                            print(f"foot contact\ncurrent ({frame_number}): {left_foot_x, left_foot_y}")
                            print(f"foot contact ({contact_frame_num}): {left_foot_xs[contact_frame_num-1], left_foot_ys[contact_frame_num-1]}")

                # find knee flexion angle
                if left_foot_status != "lifting" and left_foot_status != "Start":
                    ux, uy = left_hip_x - left_knee_x, left_hip_y - left_knee_y
                    vx, vy = left_knee_x - left_ankle_x, left_knee_y - left_ankle_y

                    thigh_angle = math.degrees(math.atan2(-uy, ux))
                    shank_angle = math.degrees(math.atan2(-vy, vx))

                    leg_angles.append(thigh_angle - shank_angle)
                else:
                    leg_angles.append(None)

                ux, uy = right_shoulder_x - right_elbow_x, right_shoulder_y - right_elbow_y
                vx, vy = right_wrist_x    - right_elbow_x, right_wrist_y   - right_elbow_y

                # Find Maximum external rotation of right forearm
                if right_elbow_x > right_knee_x and right_elbow_x > right_wrist_x and not(release_flag): 
                    external_rotation_flag = True
                    # draw a horizontal line on right elbow
                    # cv2.line(frame, (left_foot_x, right_elbow_y), (right_elbow_x, right_elbow_y), (0, 255, 0), 2)
                    # draw a line from right elbow to right wrist
                    cv2.line(frame, (right_elbow_x, right_elbow_y), (right_wrist_x, right_wrist_y), (0, 0, 255), 5)
                    # draw a line from right shoulder to right elbow
                    cv2.line(frame, (right_shoulder_x, right_shoulder_y), (right_elbow_x, right_elbow_y), (255, 0, 0), 5)

                    # elbow angle
                    dot = ux*vx + uy*vy
                    mag_u = math.hypot(ux, uy)
                    mag_v = math.hypot(vx, vy)
                    cos_theta = max(-1.0, min(1.0, dot / (mag_u * mag_v)))
                    elbow_angle = math.degrees(math.acos(cos_theta))

                    # angle difference between forearm and upper arm
                    angle_difference = math.degrees(math.atan2(vx, vy)) - math.degrees(math.atan2(-ux, -uy))

                    # Calculate the angle between the vertical line and the line from right elbow to right wrist
                    arm_angle = math.degrees(math.atan2(right_elbow_y - right_wrist_y, right_wrist_x - right_elbow_x)) # y is greater when the point is lower
                    if arm_angle < 0:
                        arm_angle += 360
                    right_forearem_angles.append(arm_angle)

                    if (arm_angle > max_external_rotation_angle):
                        max_external_rotation_angle = arm_angle
                        max_external_rotation_frame_num = frame_number
                else:
                    right_forearem_angles.append(None)

                # Find ball release frame
                forearm_angle = math.degrees(math.atan2(-vy, vx))
                upperarm_angle = math.degrees(math.atan2(uy, -ux))
                
                if not(release_flag) and forearm_angle < 90 and upperarm_angle < 90 and right_wrist_x > nose_x and right_wrist_x > right_elbow_x and right_elbow_x > right_shoulder_x:
                    right_arm_angle_difference.append(forearm_angle - upperarm_angle)
                    if forearm_angle <= upperarm_angle:
                        release_flag = True
                        if max_external_rotation_frame_num == frame_number-1 or external_rotation_flag:
                            release_frame_num = frame_number
                        else:
                            release_frame_num = frame_number-1
                        print(f"ball realease ({release_frame_num}): {left_foot_xs[release_frame_num-1], left_foot_ys[release_frame_num-1]}")
                    external_rotation_flag = False
                        
                else:
                    right_arm_angle_difference.append(forearm_angle - upperarm_angle)

                # Find the center of mass
                center_mass_x = int((left_hip_x + right_hip_x) / 2)
                center_mass_y = int((left_hip_y + right_hip_y) / 2)

                center_mass_xs.append(center_mass_x)
                center_mass_ys.append(center_mass_y)

                if len(center_mass_xs) > 1:
                    for i in range(1, len(center_mass_xs)):
                        if center_mass_xs[i-1] != None and center_mass_xs[i] != None:
                            cv2.line(frame,
                                    (center_mass_xs[i-1], center_mass_ys[i-1]),
                                    (center_mass_xs[i], center_mass_ys[i]),
                                    (255, 225, 150), thickness=3)
            else:
                left_foot_xs.append(None)
                left_foot_ys.append(None)
                left_foot_velocity_xs.append(None)
                left_foot_velocity_ys.append(None)
                right_forearem_angles.append(None)
                right_arm_angle_difference.append(None)
                leg_angles.append(None)
                center_mass_xs.append(None)
                center_mass_ys.append(None)

            if left_handed:
                frame = cv2.flip(frame, 1)
            output_frames.append(frame)
            if left_foot_x != 0 and left_foot_y != 0:
                left_foot_prev_x = left_foot_x
                left_foot_prev_y = left_foot_y

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Press 'q' to exit
            break
        elif key == ord('p'):  # Press 'p' to pause/play
            paused = not paused

    # fix the contact frame if it's too far from the position when max_external_rotation_frame_num
    if contact_frame_num > 0 and max_external_rotation_frame_num > 0:
        if abs(left_foot_ys[max_external_rotation_frame_num-1] - left_foot_ys[contact_frame_num-1]) > 5:
            for i in range(contact_frame_num, max_external_rotation_frame_num-1):
                if abs(left_foot_ys[i-1] - left_foot_ys[max_external_rotation_frame_num-1]) < 2:
                    contact_frame_num = i
                    print("fix")
                    break

    if contact_frame_num > 0:
        if contact_frame_num == max_external_rotation_frame_num:
            contact_frame_num -= 1

        for i in range (0, contact_frame_num-1):
            cv2.putText(output_frames[i],
                        "Phase 1",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 180, 255), 3
                        )
        cv2.putText(output_frames[contact_frame_num-1],
            f"foot contact",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 180, 255), 3)
        
    if max_external_rotation_frame_num > 0:
        if contact_frame_num >= max_external_rotation_frame_num:
            max_external_rotation_angle = 0
            for i in range (contact_frame_num, release_frame_num):
                if right_forearem_angles[i] is not None and right_forearem_angles[i] > max_external_rotation_angle:
                    max_external_rotation_angle = right_forearem_angles[i]
                    max_external_rotation_frame_num = i + 1
        if contact_frame_num < max_external_rotation_frame_num:
            for i in range (contact_frame_num, max_external_rotation_frame_num-1):
                cv2.putText(output_frames[i],
                            "Phase 2",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 180, 255), 3
                            )
        cv2.putText(output_frames[max_external_rotation_frame_num-1],
            f"max external rotation",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 180, 255), 3)
        
    if release_frame_num > 0:
        if contact_frame_num < max_external_rotation_frame_num and max_external_rotation_frame_num < release_frame_num:
            for i in range (max_external_rotation_frame_num, release_frame_num-1):
                cv2.putText(output_frames[i],
                            "Phase 3",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 180, 255), 3
                            )
            for i in range (release_frame_num, total_frames):
                cv2.putText(output_frames[i],
                            "Phase 4",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 180, 255), 3
                            )
        cv2.putText(output_frames[release_frame_num-1],
            f"release",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 180, 255), 3)

    for frame in output_frames:
        out.write(frame)

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("fps: ", fps)
    print(f"frame={len(frame_nums)} leg{len(leg_angles)}")

    return (
        contact_frame_num, left_foot_velocity_xs, left_foot_velocity_ys, frame_nums,
        max_external_rotation_frame_num, max_external_rotation_angle, right_forearem_angles,
        release_frame_num, right_arm_angle_difference,
        leg_angles, center_mass_xs, center_mass_ys
    )