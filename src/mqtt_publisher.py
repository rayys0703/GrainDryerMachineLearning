import paho.mqtt.client as mqtt
import time
import json
import random

# Konfigurasi MQTT
broker = "broker.hivemq.com"
port = 1883
username = "graindryer"
password = "polindra"
topics = ["iot/sensor/datagabah/1", "iot/sensor/pembakaran/2"]
client_id = f"python-client-{random.randint(0, 1000)}"

# Inisialisasi klien MQTT
client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5)
client.username_pw_set(username, password)

# Koneksi ke broker
client.connect(broker, port)
print("Connected to MQTT broker")

# Loop pengiriman data
try:
    while True:
        # Data untuk tombak pengering
        data_gabah = {
            "panel_id": 1,
            "grain_temperature": round(random.uniform(25.0, 40.0), 1),
            "grain_moisture": round(random.uniform(15.0, 30.0), 1),
            "room_temperature": round(random.uniform(20.0, 30.0), 1),
            "timestamp": int(time.time())
        }
        client.publish(topics[0], json.dumps(data_gabah))
        print(f"Published to {topics[0]}: {data_gabah}")

        # Data untuk tombak pembakaran
        data_pembakaran = {
            "panel_id": 2,
            "burning_temperature": round(random.uniform(50.0, 70.0), 1),
            "stirrer_status": random.choice([True, False]),
            "timestamp": int(time.time())
        }
        client.publish(topics[1], json.dumps(data_pembakaran))
        print(f"Published to {topics[1]}: {data_pembakaran}")

        # Tunggu 5 detik
        time.sleep(5)

except KeyboardInterrupt:
    print("Stopping MQTT publisher")
    client.disconnect()