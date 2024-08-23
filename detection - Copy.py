import os
import subprocess
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import time
import adafruit_dht
import board
import digitalio
import RPi.GPIO as GPIO
import time

# Directory to save captured images
image_dir = os.path.expanduser('~/images/')

# Ensure the directory exists
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Function to capture an image using the USB camera
def capture_image(image_path):
    capture_command = f"fswebcam -r 1280x720 --no-banner {image_path}"
    subprocess.run(capture_command, shell=True, check=True)

# Function to preprocess the image
def preprocess_image(image_path, target_size=(225, 225)):
    print(f"Preprocessing image: {image_path}")
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    
# Define GPIO pins for the RGB LED
RED_PIN = 17
GREEN_PIN = 27
BLUE_PIN = 22

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_PIN, GPIO.OUT)
GPIO.setup(GREEN_PIN, GPIO.OUT)
GPIO.setup(BLUE_PIN, GPIO.OUT)

# Setup PWM for each color
red_pwm = GPIO.PWM(RED_PIN, 100)
green_pwm = GPIO.PWM(GREEN_PIN, 100)
blue_pwm = GPIO.PWM(BLUE_PIN, 100)

# Start PWM with 0 duty cycle (off)
red_pwm.start(0)
green_pwm.start(0)
blue_pwm.start(0)
    
def set_color(red, green, blue):
    red_pwm.ChangeDutyCycle(red)
    green_pwm.ChangeDutyCycle(green)
    blue_pwm.ChangeDutyCycle(blue)
    
#Define GPIO pin for the buzzer
BUZZER_PIN = 18

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Setup PWM for the buzzer
buzzer_pwm = GPIO.PWM(BUZZER_PIN, 1000)  # Set frequency to 1kHz

def buzz(duration):
    buzzer_pwm.start(50)  # Start PWM with 50% duty cycle
    time.sleep(duration)  # Keep buzzing for the duration
    buzzer_pwm.stop()     # Stop the buzzer


# Function to check for the presence of plant using edge detection and color filtering
def check_for_plant(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range for green in HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Mask green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixel_count = np.sum(mask > 0)
    
    # Edge detection
    edges = cv2.Canny(image, 100, 200)
    edge_pixel_count = np.sum(edges > 0)

    total_pixel_count = image.shape[0] * image.shape[1]
    
    # Consider it as a plant image if more than 5% of the pixels are green or if there are significant edges
    has_plant = (green_pixel_count > 0.05 * total_pixel_count) or (edge_pixel_count > 0.05 * total_pixel_count)
    
    return has_plant

# Capture and preprocess the image
def capture_and_preprocess():
    # Define the path for the captured image
    captured_image_path = os.path.join(image_dir, 'captured_image.jpg')
    
    # Capture an image
    capture_image(captured_image_path)
    
    # Preprocess the captured image for disease prediction model
    img_array = preprocess_image(captured_image_path, target_size=(225, 225))
    
    # Return the preprocessed image array and the path
    return img_array, captured_image_path

# Load the pre-trained disease prediction model
disease_model_path = "model.h5"
disease_model = load_model(disease_model_path)

# Compile the model to avoid warnings
disease_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Predict the class of the captured image
def predict_image(disease_model, confidence_threshold=0.7):
    img_array, image_path = capture_and_preprocess()
    
    # Check if the image contains a plant
    has_plant = check_for_plant(image_path)
    
    if not has_plant:
        predicted_label = "No Plant Detected"
        set_color(100, 100, 100)  # White
        time.sleep(1)
        confidence = 0.0
    else:
        predictions = disease_model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Define your labels
        labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}
        
        if confidence < confidence_threshold:
            predicted_label = "No Plant Detected"
        else:
            predicted_label = labels[predicted_class]
    
    print(f"Predicted Disease: {predicted_label} (Confidence: {confidence:.2f})")
    
    # Display the image and prediction
    display_image(image_path, predicted_label)

    return predicted_label

# Function to display the image
def display_image(image_path, predicted_label):
    img = load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicted Disease: {predicted_label}")
    plt.axis('off')
    plt.show()

# GPIO setup for DHT11 sensor
def setup_gpio():
    try:
        pin = digitalio.DigitalInOut(board.D4)
        pin.direction = digitalio.Direction.INPUT
        print("GPIO pin 4 is set as input successfully.")
        return pin
    except Exception as e:
        print(f"Failed to set GPIO pin 4 as input: {e}")
        return None

# Function to provide recommendations based on disease and environmental conditions
def provide_recommendations(disease, temp, humidity):
    recommendations = []

    # General recommendations for Powdery Mildew
    if disease == 'Powdery':
        set_color(0, 100, 0)  # Red
        time.sleep(1)
        buzz(1)
        time.sleep(1)
        if temp > 30 or temp < 15:
            recommendations.append("Temperature is favorable to prevent powdery mildew.")
        else:
            recommendations.append(f"Adjust temperature to below 15°C or above 30°C to prevent powdery mildew. The current temperature is {temp:.2f}°C")
        
        if humidity <= 50:
            recommendations.append("Humidity is favorable to prevent powdery mildew.")
        else:
            recommendations.append(f"Adjust humidity to below 50% to prevent powdery mildew. The current humidity is {humidity:.2f}%")
    
    # General recommendations for Rust
    if disease == 'Rust':
        set_color(100, 100, 0)  # Yellow
        time.sleep(1)
        buzz(1)
        time.sleep(1)
        if temp < 10 or temp > 15:
            recommendations.append("Temperature is favorable to prevent rust.")
        else:
            recommendations.append(f"Adjust temperature to below 10°C or above 15°C to prevent rust. The current temperature is {temp:.2f}°C")
        
        if 60 <= humidity <= 70:
            recommendations.append("Humidity is favorable to prevent rust.")
        else:
            recommendations.append(f"Adjust humidity to between 60% to 70% to prevent rust. The current humidity level is {humidity:.2f}%")
    
    if disease == 'Healthy':
        set_color(100, 0, 0)  # Green
        time.sleep(1)
        recommendations.append("The plant appears healthy. Maintain current conditions.")

    if disease == 'No Plant Detected':
        recommendations.append("No plant detected. Ensure a plant is in the camera's view.")

    return recommendations

pin = setup_gpio()
if pin is not None:
    # Initialize the DHT11 sensor
    dht11 = adafruit_dht.DHT11(board.D4)

    while True:
        try:
            # Predict disease
            disease = predict_image(disease_model)
            print(f"Detected Disease: {disease}")  # Debug statement
            
            # Read temperature and humidity
            temperature = dht11.temperature
            humidity = dht11.humidity
            print(f"Temp: {temperature:.1f}C  Humidity: {humidity:.1f}%")  # Debug statement

            # Provide recommendations based on temperature, humidity, and predicted disease
            recommendations = provide_recommendations(disease, temperature, humidity)

            print("Recommendations:")  # Debug statement
            for rec in recommendations:
                print(rec)

        except RuntimeError as error:
            # Errors happen fairly often, DHT's are hard to read, just keep going
            print(error.args[0])

        time.sleep(2)
        



