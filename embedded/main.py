
# python 3.11
import json
import random
from testmotionprofile import move_arm

from paho.mqtt import client as mqtt_client


broker = 'broker.hivemq.com'
port = 1883
topic = "arm/servos"
# Generate a Client ID with the subscribe prefix.
client_id = f'subscribe-{random.randint(0, 100)}'
# username = 'emqx'
# password = 'public'


def connect_mqtt() -> mqtt_client.Client:
    # 1. Update the signature to accept 'reason_code' and 'properties'
    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {reason_code}\n")

    # 2. Add CallbackAPIVersion.VERSION2 as the FIRST argument
    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id)
    
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

#Format: {"x": 0.233, "y": 0.593, "z": 0, "grip": 0.857, "palm_orientation": "left", "orientation_angle": -90.0}
def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        decoded = json.loads(msg.payload.decode())
        print(decoded)

    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()

