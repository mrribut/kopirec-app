import pandas as pd
import sqlite3
import os

# 1. Membaca file CSV (menggunakan sep=';' karena data kamu dipisah titik koma)
nama_file = 'database coffee shop.csv'

if not os.path.exists(nama_file):
    print(f"Error: File '{nama_file}' tidak ditemukan di folder ini. Coba cek lagi ya!")
else:
    df = pd.read_csv(nama_file, sep=';')

    # 2. Membersihkan typo 'Non Coffe' menjadi 'Non Coffee' agar seragam
    df['Menu'] = df['Menu'].str.replace('Non Coffe', 'Non Coffee', case=False)

    # 3. Membuat folder 'instance' jika belum ada (tempat standar Flask menyimpan DB)
    if not os.path.exists('instance'):
        os.makedirs('instance')

    # 4. Menghubungkan ke SQLite (file database akan otomatis terbuat di dalam folder instance)
    conn = sqlite3.connect('instance/coffee_shop.db')

    # 5. Memasukkan seluruh data ke dalam tabel bernama 'coffee_shop'
    # Jika tabel sudah ada, akan ditimpa dengan yang baru (replace)
    df.to_sql('coffee_shop', conn, if_exists='replace', index=False)

    print("--- PROSES SELESAI ---")
    print(f"Berhasil mengimport {len(df)} data coffee shop ke SQLite!")
    conn.close()