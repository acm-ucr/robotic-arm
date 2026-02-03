import paho.mqtt.client as mqtt
import json

# MQTT Configuration
BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "robotic_arm/command"

# Initialize and connect to broker
client = mqtt.Client()
client.connect(BROKER, PORT, 60)

def send_coordinates(hand_x, hand_y, openness_percentage=None):
    """Publish hand coordinates to the MQTT broker."""
    message = {
        "x": hand_x,
        "y": hand_y
    }
    
    # Add openness percentage if provided
    if openness_percentage is not None:
        message["openness"] = openness_percentage
    
    client.publish(TOPIC, json.dumps(message))
    print(f"📡 Hand data sent: {message}")

# Optional test (run this file alone to send a test message)
#if __name__ == "__main__":
#    send_coordinates(500, 300, 45)
#    client.disconnect()