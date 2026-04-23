#USED TO FIND THE POSITION OF THE MOTORS MANUALLY
import time
import random
import serial
import json
# import keyboard

# import paho.mqtt.client as mqtt
# broker = 'broker.emqx.io'
# port = 1883
# topic = "python/mqtt"
# client_id = f'python-mqtt-{random.randint(0, 1000)}'
# username = 'emqx'
# password = 'public'


# ====== CONFIG ======
#change to your port name and check device manager if unsure
PORT = "COM3"  
BAUDRATE = 1000000 #keep at 1 million
SERVO_IDS = [1, 2, 3, 4, 5, 6]
MOVE_TIME = 200  # faster for live tracking
# Expected camera ranges
CAM_X_MIN, CAM_X_MAX = 0.0, 1.0
CAM_Y_MIN, CAM_Y_MAX = 0.0, 1.0
CAM_Z_MIN, CAM_Z_MAX = 0.0, 1.0
OP_MIN, OP_MAX = 0, 300
# =====================


ser = serial.Serial(PORT, BAUDRATE, timeout=0.1)

def checksum(data):
    return (~sum(data) & 0xFF)


def write_position(servo_id, position, move_time=None):
    if ser is None:
        print("Serial not connected")
        return
    position = max(0, min(4095, position))
    pos_low   = position & 0xFF
    pos_high  = (position >> 8) & 0xFF
    # allow override of MOVE_TIME if provided
    mt = MOVE_TIME if move_time is None else move_time
    time_low  = mt & 0xFF
    time_high = (mt >> 8) & 0xFF
    packet = [0xFF, 0xFF, servo_id, 7, 0x03, 0x2A,
            pos_low, pos_high, time_low, time_high]
    packet.append(checksum(packet[2:]))
    ser.write(bytearray(packet))

def read_position(servo_id):
    """Read actual current position from a servo"""
    if ser is None:
        return None
    ser.reset_input_buffer()
    start_address = 0x38
    read_length = 2
    packet = [
        0xFF, 0xFF,
        servo_id,
        4,
        0x02,
        start_address,
        read_length
    ]
    packet.append(checksum(packet[2:]))
    ser.write(bytearray(packet))
    time.sleep(0.05)
    if ser.in_waiting >= 7:
        resp = ser.read(ser.in_waiting)
        if len(resp) >= 7:
            pos = resp[5] + (resp[6] << 8)
            return pos
    return None

def read_all_positions():
    positions = {}

    for servo_id in SERVO_IDS:
        pos = read_position(servo_id)

        if pos is None:
            print(f"Servo {servo_id}: No response")
        else:
            print(f"Servo {servo_id}: {pos}")

        positions[servo_id] = pos

        time.sleep(0.02)  # small delay so servos don't collide on serial

    return positions

# ctrl c to quit
while True:
    print("---- Servo Positions ----")
    print("Press Ctrl-C to quit")
    write_position(6,random.randint(847,3212))
    time.sleep(1)
    write_position(5,random.randint(836,3205))
    time.sleep(1)
    write_position(4,random.randint(933,1036))
    time.sleep(1)
    write_position(3,random.randint(1036,2846))
    time.sleep(1)
    write_position(2,random.randint(144,3175)) # forward-back
    time.sleep(1)
    write_position(1,random.randint(1641,2765)) # left-right
    time.sleep(1)
    read_all_positions()
    time.sleep(1)
    
    
# motor min max
# 1: 847 3212
# 2: 836 3205
# 3: 933 1856
# 4: 1036 2846
# 5: 144 3175
# 6: 1641 2765
