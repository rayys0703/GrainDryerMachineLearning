# import paho.mqtt.client as mqtt
# import time
# import json
# import random

# # Konfigurasi MQTT
# broker = "127.0.0.1"
# port = 4321
# username = "graindryer"
# password = "polindra"
# topics = ["iot/sensor/datagabah/1", "iot/sensor/pembakaran/5"]
# client_id = f"python-client-{random.randint(0, 1000)}"

# # Inisialisasi klien MQTT
# client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5)
# # client.username_pw_set(username, password)

# # Koneksi ke broker
# client.connect(broker, port)
# print("Connected to MQTT broker")

# # Loop pengiriman data
# try:
#     while True:
#         # Data untuk tombak pengering
#         data_gabah = {
#             "panel_id": 1,
#             "grain_temperature": 26,
#             "grain_moisture": 28,
#             "room_temperature": 30,
#             "timestamp": int(time.time())
#         }
#         client.publish(topics[0], json.dumps(data_gabah))
#         print(f"Published to {topics[0]}: {data_gabah}")

#         # Data untuk tombak pembakaran
#         data_pembakaran = {
#             "panel_id": 5,
#             "burning_temperature": 300,
#             "stirrer_status": False,
#             "timestamp": int(time.time())
#         }
#         client.publish(topics[1], json.dumps(data_pembakaran))
#         print(f"Published to {topics[1]}: {data_pembakaran}")

#         # Tunggu 5 detik
#         time.sleep(5)

# except KeyboardInterrupt:
#     print("Stopping MQTT publisher")
#     client.disconnect()

import paho.mqtt.client as mqtt
import time
import json
import random
import pandas as pd

# === Load Data dari Excel ===
# df = pd.read_excel(r"D:\TA\PY\graindryer\src\Data_Sintetis_Pengeringan_Gabah_FDM_Variabel.xlsx")
df = pd.read_excel(r"D:\TA\PY\graindryer\src\Data_Sintetis_Pengeringan_Gabah_FDM_Variabel_15_Data_NEW.xlsx")

# Ambil hanya kolom yang diperlukan
df_filtered = df[[
    'Kadar Air Gabah (%)', 
    'Suhu Gabah (°C)', 
    'Suhu Ruangan (°C)', 
    'Suhu Pembakaran (°C)'
]]

# === Konfigurasi MQTT ===
broker = "127.0.0.1"
port = 4321
username = "graindryer"
password = "polindra"
topics = ["iot/sensor/datagabah/1", "iot/sensor/pembakaran/5"]
client_id = f"python-client-{random.randint(0, 1000)}"

# Inisialisasi MQTT client
client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5)
# client.username_pw_set(username, password)
client.connect(broker, port)
print("Connected to MQTT broker")

# === Loop Kirim Data dari File ===
try:
    for index, row in df_filtered.iterrows():
        # Kirim data gabah
        data_gabah = {
            "panel_id": 1,
            "grain_temperature": round(row['Suhu Gabah (°C)'], 2),
            "grain_moisture": round(row['Kadar Air Gabah (%)'], 2),
            "room_temperature": round(row['Suhu Ruangan (°C)'], 2),
            "timestamp": int(time.time())
        }
        client.publish(topics[0], json.dumps(data_gabah))
        print(f"Published to {topics[0]}: {data_gabah}")

        # Kirim data pembakaran
        data_pembakaran = {
            "panel_id": 5,
            "burning_temperature": round(row['Suhu Pembakaran (°C)'], 2),
            "stirrer_status": False, 
            "timestamp": int(time.time())
        }
        client.publish(topics[1], json.dumps(data_pembakaran))
        print(f"Published to {topics[1]}: {data_pembakaran}")

        time.sleep(5)

except KeyboardInterrupt:
    print("Stopping MQTT publisher")
    client.disconnect()