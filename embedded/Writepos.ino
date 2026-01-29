/*
The normal write example passed the test in ST3215 Servo, 
and if testing other models of ST series servos
please change the appropriate position, speed and delay parameters.
*/
#include <Adafruit_NeoPixel.h>
#include <SCServo.h>
#define PIN        48 
#define NUMPIXELS  1 

SMS_STS st;

Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);


// the UART used to control servos.
// GPIO 18 - S_RXD, GPIO 19 - S_TXD, as default.
#define S_RXD 44
#define S_TXD 43

void setup()
{
  Serial.begin(115200);
  Serial1.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(1000);
  pixels.begin();
}

void loop()
{
  st.WritePosEx(4, 2500, 3400, 50); // servo(ID1) speed=3400，acc=50，move to position=4095.
  delay(2000);
  
  st.WritePosEx(4, 2000, 1500, 50); // servo(ID1) speed=3400，acc=50，move to position=2000.
  delay(2000);
  pixels.clear();

  pixels.setPixelColor(0, pixels.Color(0, 150, 150));
  pixels.show();
  Serial.println("LED ON");
  delay(1000);
  //prev

  pixels.clear();
  pixels.show();
  Serial.println("LED OFF");
  delay(1000);
}
