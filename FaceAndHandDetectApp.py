import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import mediapipe as mp

# Create application titile and file uploader widget
st.title("Face Detection Application😃")
img_file_buffer = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])


# Function for detecting faces in and image
def detectFaceOpenCVDnn(net, frame):
    # create a blot from image and some pre-processing
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob input to the model
    net.setInput(blob)
    detections = net.forward()
    return detections

# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detection and draw bounding boxes around each faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            # Draw bounding boxes around detected faces
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), bb_line_thickness, cv2.LINE_8)

    return frame, bboxes
# Function to load the DNN model.
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net


# Function to generate download link
def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    hex_data = buffered.getvalue()
    return hex_data

net = load_model()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print handedness and draw hand landmarks on the image.
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks:', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        if not results.multi_hand_world_landmarks:
            continue
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)


if img_file_buffer is not None:
    # Read the file and convert it to opencv image
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    # load image in BGR channel order
    image_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)

    # create placeholder to display input and output image.
    col1, col2 = st.columns(2)
    # Display input image
    col1.image(image_bgr, channels='BGR')
    col1.text("Input Image")

    # Create Slider and get the threshold from the slider.
    conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

    # Call detection function
    detections = detectFaceOpenCVDnn(net, image_bgr)
    # Process the detection based on the current confidence threshold.
    out_image, _ = process_detections(image_bgr, detections)

    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
        results = hands.process(out_image)

        out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    out_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


    # Display the output image
    col2.image(out_image, channels='BGR')
    col2.text("Output Image")

    # Convert opencv image to PIL.
    out_image = Image.fromarray(out_image[:, :, ::-1])  # convert to RGB image
    # create a Download Button
    col2.download_button("Download",data=get_image_download_link(out_image),file_name="output_image.jpeg")


image_path = "chi.jpg"

st.header("FOUNDER OF CHI👨🏻‍💻")
founder_img = cv2.imread(image_path)
st.image(founder_img[:, :, ::-1], width=350)
st.markdown("""We are two brothers **[ ZEN || CHI ]** . Very passionate about learning and building Artificial 
              Intelligence models. Same as you like to eat your favorite food.**We believe Artificial Intelligence 
              solved human any problem in the 21st century.**""")
