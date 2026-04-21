from __future__ import annotations
import cv2
import mediapipe as mp
import time
import paho.mqtt.client as mqtt
import json

# --- MQTT Setup (add near the top, after your imports) ---
BROKER   = "127.0.0.1"   # change to broker IP if on another machine | empx
PORT     = 1883
TOPIC_PUB = "arm/servos"      # topic your script publishes angles to
TOPIC_SUB = "arm/feedback"    # optional: topic the arm sends status back on

