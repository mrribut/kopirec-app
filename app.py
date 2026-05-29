from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def ambil_data_base():
    conn = sqlite3.connect('instance/coffee_shop.db')
    df = pd.read_sql_query("SELECT * FROM coffee_shop", conn)
    conn.close()
    return df

def ambil_list_atribut(df):
    fasilitas_set = set()
    df['Fasilitas'].str.lower().str.split(',').dropna().apply(lambda x: [fasilitas_set.add(i.strip()) for i in x])
    
    menu_set = set()
    df['Menu'].str.lower().str.split(',').dropna().apply(lambda x: [menu_set.add(i.strip()) for i in x])
    
    return sorted(list(fasilitas_set)), sorted(list(menu_set))

# ----------------- 1. HALAMAN REKOMENDASI (SISTEM INTI) -----------------
@app.route('/')
def index():
    df = ambil_data_base()
    list_fasilitas, list_menu = ambil_list_atribut(df)
    list_harga = sorted(df['Harga'].unique())
    return render_template('index.html', fasilitas=list_fasilitas, menu=list_menu, harga=list_harga, aktif='rekomendasi')

@app.route('/cari_rekomendasi', methods=['POST'])
def cari_rekomendasi():
    harga_dipilih = request.form.get('harga')
    menu_dipilih = request.form.getlist('menu')
    fasilitas_dipilih = request.form.getlist('fasilitas')

    df = ambil_data_base()

    if harga_dipilih and harga_dipilih != 'semua':
        nominal_float = float(harga_dipilih)
        df_filtered = df[df['Harga'] == nominal_float].copy()
        label_harga = f"Rp {int(nominal_float)}.000"
    else:
        df_filtered = df.copy()
        label_harga = "Semua Harga"

    if df_filtered.empty:
        return render_template('hasil.html', hasil=[], pesan=f"Tidak ada coffee shop dengan harga {label_harga}.", preferensi={"harga": label_harga, "menu": ", ".join(menu_dipilih), "fasilitas": ", ".join(fasilitas_dipilih)}, aktif='rekomendasi')

    df_filtered = df_filtered.reset_index(drop=True)

    teks_preferensi_user = " ".join(fasilitas_dipilih) + " " + " ".join(menu_dipilih)
    teks_preferensi_user = teks_preferensi_user.lower().strip()

    if not teks_preferensi_user:
        df_filtered['Skor_Kemiripan'] = 0.0
        top_5 = df_filtered.head(5)
    else:
        df_filtered['fasilitas_clean'] = df_filtered['Fasilitas'].str.lower().str.replace(',', ' ')
        df_filtered['menu_clean'] = df_filtered['Menu'].str.lower().str.replace(',', ' ')
        df_filtered['metadata_cafe'] = df_filtered['fasilitas_clean'] + ' ' + df_filtered['menu_clean']

        list_metadata = df_filtered['metadata_cafe'].tolist()
        list_metadata.append(teks_preferensi_user)

        cv = CountVectorizer()
        count_matrix = cv.fit_transform(list_metadata)
        cosine_sim = cosine_similarity(count_matrix)

        idx_user = len(list_metadata) - 1
        skor_user_vs_cafe = cosine_sim[idx_user][:-1]

        df_filtered['Skor_Kemiripan'] = skor_user_vs_cafe
        top_5 = df_filtered.sort_values(by='Skor_Kemiripan', ascending=False).head(5)

    hasil_rekomendasi = top_5.to_dict(orient='records')

    return render_template('hasil.html', hasil=hasil_rekomendasi, preferensi={
        "harga": label_harga,
        "menu": ", ".join(menu_dipilih) if menu_dipilih else "Semua Menu",
        "fasilitas": ", ".join(fasilitas_dipilih) if fasilitas_dipilih else "Semua Fasilitas"
    }, aktif='rekomendasi')


# ----------------- 2. HALAMAN PENCARIAN MANUAL & DAFTAR DATA -----------------
@app.route('/coffee_shop')
def coffee_shop():
    query_cari = request.args.get('search', '').strip()
    df = ambil_data_base()
    
    # Jika ada keyword pencarian manual yang diketik user
    if query_cari:
        # Filter berdasarkan nama cafe (tidak sensitif huruf besar/kecil)
        df_hasil = df[df['Nama'].str.contains(query_cari, case=False, na=False)].copy()
    else:
        df_hasil = df.copy()
        
    list_cafe = df_hasil.to_dict(orient='records')
    return render_template('database.html', coffee_shops=list_cafe, keyword=query_cari, aktif='database')

if __name__ == '__main__':
    app.run(debug=True)