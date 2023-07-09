from flask import Flask, request, render_template
from requests.structures import CaseInsensitiveDict

from naiveBayes import summaries, predict, accuracy
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_data = False

    if request.method == 'POST':
        

        # User input
        user_input = []
        rating_bahan = float(request.form["pertanyaan1"])
        rating_tampilan = float(request.form["pertanyaan2"])
        rating_fitur = float(request.form["pertanyaan3"])
        rating_proses_pelayanan = float(request.form["pertanyaan4"])
        rating_nama_brand = float(request.form["pertanyaan5"])
        user_input.extend([rating_bahan, rating_tampilan, rating_fitur, rating_proses_pelayanan, rating_nama_brand])

        prediction = predict(summaries, user_input)

        # Mencetak prediksi kualitas
        quality = "baik" if prediction == 1 else "buruk"
        print(f"Prediksi kualitas produk: {quality}")

        prediction_data = quality
    return render_template('index.html', prediction = prediction_data, accuracy=round(accuracy,3))

# Uncomment code below if you want to host it locally
# if __name__ == '__main__':
#    app.run(debug=False,host='0.0.0.0')
 
