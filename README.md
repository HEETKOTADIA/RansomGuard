📘 About the Project

RansomGuard is a machine learning–powered ransomware detection and classification system designed to identify ransomware families based on behavioral and network metadata. Unlike signature-based antivirus engines, RansomGuard analyzes patterns such as communication protocol, network traffic, flags, clustering patterns, and ransom payment indicators to accurately classify malware families such as WannaCry, Locky, Petya, CryptoLocker, and more.

The project includes:

🔥 A trained LightGBM ML model

🎨 A modern PyQt6 GUI with a sleek cyber-security theme

📁 CSV-based prediction system

🛠 Fully automated preprocessing pipeline

RansomGuard demonstrates how machine learning can be applied to cyber-security for behavior-based ransomware detection.

🚀 Features

✔ Machine-learning-based ransomware family classification

✔ Fast and efficient LightGBM model

✔ Modern, eye-catching PyQt6 GUI

✔ CSV file input for bulk prediction

✔ Handles preprocessing (encoding, scaling, imputation) automatically

✔ Offline detection (no internet required)

✔ Detects multiple ransomware families

RansomGuard/
│── checkpoints/              # Trained model & encoders
│── gui.py                    # Modern PyQt6 GUI
│── train_ransomguard.py       # Model training script
│── data/                      # Dataset (optional)
│── requirements.txt
│── inputs.csv                # Example input
│── README.md

🔧 Tech Stack

Python 3.11

LightGBM

Scikit-learn

NumPy / Pandas

PyQt6

Joblib

🧠 How RansomGuard Works

User uploads a CSV file containing ransomware behavioral data

System validates required fields

Columns like Protcol and Flag are label-encoded

Missing values are imputed

Values are scaled to match training distributions

LightGBM predicts the ransomware family

GUI displays results instantly in a cyber-themed window

📊 Detected Ransomware Families

RansomGuard can classify well-known ransomware families such as:

🟦 WannaCry

🟥 CryptoLocker

🟩 Locky

🟨 CryptoWall

🟪 Petya

🟧 SamSam

🟫 Cerber

🟦 Ryuk

🟩 Maze

🟨 GandCrab

▶️ How to Use
🔹 Step 1 — Install dependencies
pip install -r requirements.txt

🔹 Step 2 — Run GUI
python ransomguard_csv_gui.py

🔹 Step 3 — Upload your CSV

Must follow this format:

Time,Protcol,Flag,Clusters,BTC,USD,Netflow_Bytes,Port
40,TCP,A,1,1,500,12,5061
57,TCP,A,1,1,540,18,5061

🔹 Step 4 — View predictions

The GUI displays ransomware families row-by-row.

📸 Screenshots
<img width="1812" height="1443" alt="image" src="https://github.com/user-attachments/assets/41c4b112-df13-460a-89ee-4e809873ceb5" />

📚 Dataset Sources

You may include any dataset sources used, such as:

Kaggle Cyber Security Datasets

UNB CIC Malware & Ransomware Datasets

CSE Ransomware Dataset

Custom simulated data

🔮 Future Enhancements

Real-time ransomware network monitoring

API-based detection engine

Integration with SIEM / SOC tools

Deep learning integration (LSTM, CNN, Autoencoders)

Early detection system for live network packets

Web dashboard using Streamlit or FastAPI

Ransomware heatmap visualization

🤝 Contributing

Contributions, issues, and feature requests are welcome!

🛡️ License

This project is released under the MIT License.
