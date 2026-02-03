#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "";          // Change this
const char* password = "";  // Change this

const char* mqtt_server = "broker.emqx.io";
WiFiClient espClient;
PubSubClient client(espClient);

void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  // OUTPUT THE MESSAGE
  Serial.println("====================");
  Serial.print("Topic: ");
  Serial.println(topic);
  Serial.print("Message: ");
  Serial.println(message);
  Serial.println("====================");
}

void setup() {
  Serial.begin(115200);
  delay(3000);
  
  Serial.println("\n=== Connecting to WiFi ===");
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\n✓ WiFi Connected!");
  Serial.println(WiFi.localIP());
  
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  
  String clientId = "Arm_" + String(random(0xffff), HEX);
  if (client.connect(clientId.c_str())) {
    Serial.println("✓ MQTT Connected!");
    client.subscribe("robotic_arm/command");
    Serial.println("Listening for messages on: robotic_arm/command\n");
  }
}

void loop() {
  if (!client.connected()) {
    String clientId = "Arm_" + String(random(0xffff), HEX);
    client.connect(clientId.c_str());
    client.subscribe("robotic_arm/command");
  }
  client.loop();
}
