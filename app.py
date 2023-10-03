import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

classifier = cv2.CascadeClassifier("E:\Deep Learning\PROJECTS\MaskDetector\haarcascade_frontalface_default.xml")
detector = load_model("E:\Deep Learning\PROJECTS\MaskDetector\dl-model.save")

def load_image(img_file):
    img = Image.open(img_file)
    return img

def main():
    st.title("Welcome to Face Mask Detector")
    menu = ["Detector", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Detector":
        st.subheader("Upload the image to check mask")
        image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
        if image_file is not None:
            file_details = {
                "filename": image_file.name,
                "filetype": image_file.type,
                "filesize": image_file.size,
            }
            st.write(file_details)

            image_array = np.array(load_image(image_file))
            new_image = image_array.copy()
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
            faces = classifier.detectMultiScale(new_image, 1.1, 4)

            new_image_2 = new_image.copy()  # Initialize new_image_2

            for x, y, w, h in faces:
                face_img = new_image[y:y+h, x:x+w] # crop the face
                resized = cv2.resize(face_img, (224, 224))
                image_arr = tf.keras.preprocessing.image.img_to_array(resized)
                image_arr = tf.expand_dims(image_arr, 0)
                pred = detector.predict(image_arr)
                score = tf.nn.softmax(pred[0])
                label = np.argmax(score)
        
                if label == 0:
                    cv2.rectangle(new_image_2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(new_image_2, "mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                elif label == 1:
                    cv2.rectangle(new_image_2, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(new_image_2, "no mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Save the modified image using OpenCV
            output_path = "out.png"
            cv2.imwrite(output_path, cv2.cvtColor(new_image_2, cv2.COLOR_RGB2BGR))

            # Display the saved image
            st.image(load_image(output_path))

    elif choice == "About":
        st.subheader("About Project")

if __name__ == "__main__":
    main()
