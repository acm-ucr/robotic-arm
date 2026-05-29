#USED TO FIND THE POSITION OF THE MOTORS MANUALLY

import time

import serial
import json
import os
from dotenv import load_dotenv
load_dotenv()

# ====== CONFIG ======
#change to your port name and check device manager if unsure
PORT = os.getenv("SERIAL_PORT")  
BAUDRATE = 1000000
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


while True:
    print("---- Servo Positions ----")
    read_all_positions()
    time.sleep(1)

# write_position(3, 3000, 1)
