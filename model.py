import os
import cv2
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from call import make_twilio_call  # Import the function from the call module
from moviepy.editor import *


from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

#Preprocess the data

# Height and width to which each video frame will be resized.
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

# Frame count that will be fed to the model
SEQUENCE_LENGTH = 20
 
DATASET_DIR =  r'D:\Anomaly Detection\Dataset'

# CLASSES_LIST = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Normal_Videos_for_Event_Recognition", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
# CLASSES_LIST = ["Explosion", "Fighting", "Normal_Videos_for_Event_Recognition", "RoadAccidents", "Robbery"]
# CLASSES_LIST = ["Explosion", "Fighting", "Shooting", "Normal_Videos_for_Event_Recognition", "RoadAccidents", "Robbery", "Burglary"]
# CLASSES_LIST = ["Abuse", "Arrest", "Arson", "Normal_Videos_for_Event_Recognition"]
CLASSES_LIST = ['Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Normal']
#Created a Function to extract, resize & normalize
def frames_extraction(video_path):

    frames_list = []
    
    video_reader = cv2.VideoCapture(video_path)

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        success, frame = video_reader.read() 

        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        normalized_frame = resized_frame / 255
        
        frames_list.append(normalized_frame)
    
    video_reader.release()

    return frames_list

#Created a Function for dataset creation
def create_dataset():

    features = []
    labels = []
    video_files_paths = []

    for class_index, class_name in enumerate(CLASSES_LIST):
        
        print(f'Extracting Data of Class: {class_name}')
        
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        for file_name in files_list:
            
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)

            frames = frames_extraction(video_file_path)

            if len(frames) == SEQUENCE_LENGTH:

                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)

    features = np.asarray(features)
    labels = np.array(labels)  

    return features, labels, video_files_paths

def create_LRCN_model():
    '''
    This function will construct the required LRCN model.
    Returns:
        model: It is the required constructed LRCN model.
    '''

    # We will use a Sequential model for model construction.
    model = Sequential()
    
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    
    model.add(TimeDistributed(MaxPooling2D((4, 4)))) 
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    #model.add(TimeDistributed(Dropout(0.25)))
                                      
    model.add(TimeDistributed(Flatten()))
                                      
    model.add(LSTM(32))
                                      
    model.add(Dense(len(CLASSES_LIST), activation = 'softmax'))

    # Display the models summary.
    model.summary()
    
    # Return the constructed LRCN model.
    return model

# Construct the required LRCN model.
LRCN_model = create_LRCN_model()

# Display the success message.
print("Model Created Successfully!")

# Plot the structure of the contructed LRCN model.
plot_model(LRCN_model, to_file = 'LRCN_model_structure_plot.png', show_shapes = True, show_layer_names = True)

#Load the model
LRCN_model_path = r"D:\Anomaly Detection\Untitled Folder\lrcn_model__Date_Time_2023_10_24_22_37_08_Loss_0.86_Accuracy_0.78.h5"

from tensorflow.keras.models import load_model

# Load the pre-trained LRCN model
LRCN_model = load_model(LRCN_model_path)
model = load_model(LRCN_model_path)


from keras.models import load_model
def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    
    video_reader = cv2.VideoCapture(video_file_path)
    
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('H', '2', '6', '4'),
                                  video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
    
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)
    
    predicted_class_name = ""
    
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    while video_reader.isOpened():
        ok,frame = video_reader.read()
        
        if not ok:
            break
        
        frame_bg = fgbg.apply(frame)
        
        frame_bg = cv2.GaussianBlur(frame_bg,(3,3),cv2.BORDER_DEFAULT)
        
        resized_frame = cv2.resize(frame_bg, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        normalized_frame = resized_frame/255
        
        
        frames_queue.append(normalized_frame)
        
        if len(frames_queue)==SEQUENCE_LENGTH:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            
            predicted_label = np.argmax(predicted_labels_probabilities)
            
            predicted_class_name = CLASSES_LIST[predicted_label]
            
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        video_writer.write(frame)
        
    video_reader.release()
    video_writer.release()


# def predict_on_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
#     # Initialize the VideoCapture object to read from the video file.
#     video_reader = cv2.VideoCapture(video_file_path)

#     # Get the width and height of the video.
#     original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Initialize the VideoWriter Object to store the output video in MP4 format.
#     video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('H', '2', '6', '4'), 
#                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

#     # Rest of your code for video processing goes here...
#     # Declare a queue to store video frames.
#     frames_queue = deque(maxlen = SEQUENCE_LENGTH)

#     # Initialize a variable to store the predicted action being performed in the video.
#     predicted_class_name = ''

#     # Iterate until the video is accessed successfully.
#     while video_reader.isOpened():

#         # Read the frame.
#         ok, frame = video_reader.read() 
        
#         # Check if frame is not read properly then break the loop.
#         if not ok:
#             break

#         # Resize the Frame to fixed Dimensions.
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
#         # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
#         normalized_frame = resized_frame / 255

#         # Appending the pre-processed frame into the frames list.
#         frames_queue.append(normalized_frame)

#         # Check if the number of frames in the queue are equal to the fixed sequence length.
#         if len(frames_queue) == SEQUENCE_LENGTH:

#             # Pass the normalized frames to the model and get the predicted probabilities.
#             predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

#             # Get the index of class with highest probability.
#             predicted_label = np.argmax(predicted_labels_probabilities)

#             # Get the class name using the retrieved index.
#             predicted_class_name = CLASSES_LIST[predicted_label]

#         # Write predicted class name on top of the frame.
#         cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Write The frame into the disk using the VideoWriter Object.
#         video_writer.write(frame)
        
#     # Release the VideoCapture and VideoWriter objects.
#     video_reader.release()
#     video_writer.release()



def predict_single_action(video_file_path, SEQUENCE_LENGTH):

    video_reader = cv2.VideoCapture(video_file_path)
 
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Declare a list to store video frames we will extract.
    frames_list = []
    
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''
 
    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
    
    fgbg = cv2.createBackgroundSubtractorMOG2()
 
    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
 
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
 
        # Read a frame.
        success, frame = video_reader.read() 
 
        # Check if frame is not read properly then break the loop.
        if not success:
            break
         
        frame_bg = fgbg.apply(frame)
        
        frame = cv2.GaussianBlur(frame_bg,(3,3),cv2.BORDER_DEFAULT)

        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
 
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)

    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    predicted_confidence = predicted_labels_probabilities[predicted_label]
    # Display the predicted action along with the prediction confidence.
    print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_confidence}')
    if predicted_class_name in ['Burglary', 'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 'Shooting'] and predicted_confidence >= 0.5:
        make_twilio_call()  # Call the function from the call module    
    # Release the VideoCapture object. 
    video_reader.release()
    return predicted_class_name, predicted_confidence


# #Function To Perform a Single Prediction on Videos
# def predict_single_action(video_file_path, SEQUENCE_LENGTH):

#     # Initialize the VideoCapture object to read from the video file.
#     video_reader = cv2.VideoCapture(video_file_path)

#     # Get the width and height of the video.
#     original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
#     original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     # Declare a list to store video frames we will extract.
#     frames_list = []
    
#     # Initialize a variable to store the predicted action being performed in the video.
#     predicted_class_name = ''

#     # Get the number of frames in the video.
#     video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Calculate the interval after which frames will be added to the list.
#     skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)

#     # Iterating the number of times equal to the fixed length of sequence.
#     for frame_counter in range(SEQUENCE_LENGTH):

#         # Set the current frame position of the video.
#         video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

#         # Read a frame.
#         success, frame = video_reader.read() 

#         # Check if frame is not read properly then break the loop.
#         if not success:
#             break

#         # Resize the Frame to fixed Dimensions.
#         resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        
#         # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
#         normalized_frame = resized_frame / 255
        
#         # Appending the pre-processed frame into the frames list
#         frames_list.append(normalized_frame)

#     # Passing the  pre-processed frames to the model and get the predicted probabilities.
#     predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_list, axis = 0))[0]

#     # Get the index of class with highest probability.
#     predicted_label = np.argmax(predicted_labels_probabilities)

#     # Get the class name using the retrieved index.
#     predicted_class_name = CLASSES_LIST[predicted_label]
#     predicted_confidence = predicted_labels_probabilities[predicted_label]
#     # Display the predicted action along with the prediction confidence.
#     print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_confidence}')
#     if predicted_class_name in ["Explosion", "Fighting", "RoadAccidents", "Robbery"] and predicted_confidence >= 0.5:
#         make_twilio_call()  # Call the function from the call module    
#     # Release the VideoCapture object. 
#     video_reader.release()
#     return predicted_class_name, predicted_confidence


#Live Camera Logic
def predict_and_display_live_video(SEQUENCE_LENGTH):
    # Initialize the VideoCapture object to capture video from the default camera (camera index 0).
    video_reader = cv2.VideoCapture(0)  # Change 0 to the appropriate camera index if needed.

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    while True:
        # Read a frame from the camera.
        ok, frame = video_reader.read()

        if not ok:
            break

        # Resize and normalize the frame.
        resized_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame in the "Predicted Video" window.
        cv2.imshow("Predicted Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_reader.release()
    cv2.destroyAllWindows()