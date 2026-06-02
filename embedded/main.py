# python 3.11

# python 3.11
import json
import random
from testmotionprofile import enqueue_move
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

# payload:
#"x": round(wrist.x, 3) - 0.5,
#"y": round(wrist.y, 3),
#"z": round(z_value, 3),
#"grip": grip, (0, 1)
#"orientation_angle": orientation_angle, - motor 5, palm facing you - 0, palm facing camera right - 90, 
# palm facing camera left - -90, palm facing camera - oscillates -180 and 180
#"pitch_angle": pitch_angle, - motor 4 (-90, 90) degrees, 0 = upright, 90 = palm down, -90 = palm up

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        decoded = json.loads(msg.payload.decode())
        print(decoded)

        x = decoded.get("x", 0.0)
        y = decoded.get("y", 0.0)
        z = decoded.get("z", 0.0)
        grip = decoded.get("grip", 0.0)
        orientation_angle = decoded.get("orientation_angle", 0.0)
        pitch_angle = decoded.get("pitch_angle", 0.0)

        enqueue_move(x, y, z, grip, orientation_angle, pitch_angle)
        

    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()

