import cv2
import mediapipe as mp
import numpy as np
import json

# Set to true if live recording
LIVE = True

# List of joint sets for angle calculations
JOINT_LIST = [[4, 3, 2], [8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]


def draw_finger_angles(image, hand, joint_list):
    """
    Draws angles between finger joints on the image and returns the angle values.

    Args:
        image (numpy.ndarray): Input image.
        hand (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Detected hand landmarks.
        joint_list (list): List of joint sets for which angles are calculated.

    Returns:
        dict: Dictionary containing angles for each joint set in hand landmarks.
    """
    hand_angles = {}
    for joint in joint_list:
        a = np.array(
            [hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]
        )  # First coord
        b = np.array(
            [hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]
        )  # Second coord
        c = np.array(
            [hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]
        )  # Third coord

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        # Draw angle
        cv2.putText(
            image,
            str(round(angle, 2)),
            tuple(np.multiply(b, [image.shape[1], image.shape[0]]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Collect angles
        joint_name = f"{joint[1]}"
        hand_angles[joint_name] = round(angle, 2)

    return hand_angles


def custom_draw_landmarks(image, hand_landmarks, connections):
    """
    Draws landmarks and joint numbers on the image.

    Args:
        image (numpy.ndarray): Input image.
        hand_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Detected hand landmarks.
        connections (list): List of connections between landmarks.

    Returns:
        None
    """
    # Draw the hand landmarks
    mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, connections)

    # Draw joint numbers
    for i, landmark in enumerate(hand_landmarks.landmark):
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.putText(
            image,
            str(i),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    # Initialize MediaPipe Hands and Drawing Utilities
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Open the camera or video capture device
    if LIVE:
        cap = cv2.VideoCapture(0)
    else:
        # Your input video name
        input_video = "clip.MOV"
        cap = cv2.VideoCapture(input_video)

    # Output video filename
    output_video = "output.mp4"

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # List to hold angles for each frame
    all_angles = []

    # Main loop to process each frame
    with mp_hands.Hands(
        min_detection_confidence=0.8, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # BGR to RGB conversion
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip horizontally for selfie-view display
            image = cv2.flip(image, 1)

            # Set flag to prevent in-place modifications
            image.flags.writeable = False

            # Run hand detection
            results = hands.process(image)

            # Allow modifications to the image
            image.flags.writeable = True

            # RGB to BGR conversion for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Process each detected hand
            frame_angles = []
            if results.multi_hand_landmarks:
                for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks and joint numbers
                    custom_draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # Draw angles and collect angle data
                    angles = draw_finger_angles(image, hand_landmarks, JOINT_LIST)
                    frame_angles.append(angles)

            # Append angles data for the frame
            all_angles.append(frame_angles)

            # Write the frame to the output video file
            out.write(image)

            # Display the annotated frame
            cv2.imshow("Hand Tracking", image)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Save all angles data to a JSON file
    with open("output_angles.json", "w") as f:
        json.dump(all_angles, f)
