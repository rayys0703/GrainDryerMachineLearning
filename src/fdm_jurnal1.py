import numpy as np
import pandas as pd

# Parameter simulasi
jenis_gabah = 1
status_pengaduk = 0
bobot_gabah = 20000  # kg

interval_detik = 5
total_menit = 1200  # durasi total simulasi
total_detik = total_menit * 60
n_time_steps = total_detik // interval_detik + 1

# Waktu simulasi dalam menit per step
waktu_menit = np.linspace(0, total_menit, n_time_steps)

# Kadar air gabah menurun linear dari 28% ke 14%
kadar_air = 28 - ((28 - 14) / total_menit) * waktu_menit

# Suhu gabah naik linear dari 26°C ke 40°C
suhu_gabah = 26 + ((40 - 26) / total_menit) * waktu_menit

# Suhu ruangan dengan fluktuasi sinusoidal ±2°C di sekitar 28°C
suhu_ruangan = 28 + 2 * np.sin(2 * np.pi * (waktu_menit / 30))  # periode 30 menit

# Suhu pembakaran sekitar 300°C dengan fluktuasi sinusoidal dan noise Gaussian
np.random.seed(42)
suhu_pembakaran = 300 + 10 * np.sin(2 * np.pi * (waktu_menit / 15)) + np.random.normal(0, 2, n_time_steps)

# Status pengaduk konstan 0
status_pengaduk_arr = np.zeros(n_time_steps, dtype=int)

# Bobot gabah konstan 20000 kg
bobot_arr = np.full(n_time_steps, bobot_gabah)

# Waktu pengeringan menurun dari 1200 menit ke 0
waktu_pengeringan = total_menit - waktu_menit

# Data ke dalam DataFrame
data = pd.DataFrame({
    "Interval (detik)": np.arange(0, total_detik + 1, interval_detik),
    "Estimasi (menit)": waktu_pengeringan,
    "Jenis_Gabah_ID": jenis_gabah,
    "Kadar Air Gabah (%)": kadar_air,
    "Suhu Gabah (°C)": suhu_gabah,
    "Suhu Ruangan (°C)": suhu_ruangan,
    "Suhu Pembakaran (°C)": suhu_pembakaran,
    "Massa Gabah (Kg)": bobot_arr,
    "Status Pengaduk": status_pengaduk_arr
})

# Simpan ke file Excel dengan format angka 5 desimal menggunakan XlsxWriter
filename = "Data_Sintetis_Pengeringan_Gabah_FDM_NEW.xlsx"
with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
    data.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    # Format angka dengan 5 desimal
    format_5desimal = workbook.add_format({'num_format': '0.00000'})

    # Atur format kolom yang berisi angka desimal
    # Kolom mulai dari indeks 3 (Kadar Air Gabah) sampai 6 (Suhu Pembakaran)
    worksheet.set_column('D:G', None, format_5desimal)

print(f"Data sintetis berhasil disimpan ke {filename} dengan format 5 desimal pada Excel.")
