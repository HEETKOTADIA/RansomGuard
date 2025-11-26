import sys
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QTextEdit, QFileDialog,
    QVBoxLayout, QMessageBox, QGraphicsDropShadowEffect
)
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve

CHECKPOINT_DIR = "checkpoints"

# Load ML Components
lgb_model = lgb.Booster(model_file=os.path.join(CHECKPOINT_DIR, "lgb_model.txt"))
label_encoder = joblib.load(os.path.join(CHECKPOINT_DIR, "label_encoder.joblib"))
imputer = joblib.load(os.path.join(CHECKPOINT_DIR, "imputer.joblib"))
scaler = joblib.load(os.path.join(CHECKPOINT_DIR, "scaler.joblib"))
enc_protcol = joblib.load(os.path.join(CHECKPOINT_DIR, "encoder_Protcol.joblib"))
enc_flag = joblib.load(os.path.join(CHECKPOINT_DIR, "encoder_Flag.joblib"))

REQUIRED_FIELDS = [
    "Time", "Protcol", "Flag", "Clusters", "BTC",
    "USD", "Netflow_Bytes", "Port"
]

class RansomGuardCSV(QWidget):
    def __init__(self):
        super().__init__()

        # Window Styling
        self.setWindowTitle("🛡️ RansomGuard – Cyber UI CSV Predictor")
        self.setGeometry(250, 60, 900, 700)
        self.setStyleSheet("""
            QWidget {
                background-color: #12131A;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Title with glow
        title = QLabel("🛡️ RansomGuard – Ransomware Predictor")
        title.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            color: #6CB8FF;
            text-shadow: 0px 0px 20px #6CB8FF;
        """)
        layout.addWidget(title)

        subtitle = QLabel("AI-Powered Ransomware Detection (CSV Input Only)")
        subtitle.setFont(QFont("Segoe UI", 13))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #CDD6F4;")
        layout.addWidget(subtitle)

        # Upload Button - Neon Glow
        self.btn_upload = QPushButton("📁 Upload CSV File")
        self.btn_upload.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.btn_upload.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_upload.setFixedHeight(55)
        self.btn_upload.setStyleSheet("""
            QPushButton {
                background-color: #3B82F6;
                color: white;
                border-radius: 12px;
            }
            QPushButton:hover {
                background-color: #60A5FA;
                border: 2px solid #93C5FD;
            }
            QPushButton:pressed {
                background-color: #2563EB;
            }
        """)
        self.btn_upload.clicked.connect(self.load_csv)
        layout.addWidget(self.btn_upload)

        # Output Box with Glass Effect
        self.output_box = QTextEdit()
        self.output_box.setFont(QFont("Consolas", 11))
        self.output_box.setReadOnly(True)
        self.output_box.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 255, 255, 0.07);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 12px;
                color: #E3E3E3;
                padding: 15px;
            }
        """)
        layout.addWidget(self.output_box)

        # Drop Shadow for Output Box
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        self.output_box.setGraphicsEffect(shadow)

    # -------------------------------------------
    # Load CSV + Predict
    # -------------------------------------------
    def load_csv(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select CSV File", "", "CSV Files (*.csv)"
            )
            if not file_path:
                return

            df = pd.read_csv(file_path)

            # Validate required columns
            for col in REQUIRED_FIELDS:
                if col not in df.columns:
                    raise ValueError(f"Missing column in CSV: {col}")

            # Encode categorical fields
            df["Protcol"] = enc_protcol.transform(df["Protcol"])
            df["Flag"] = enc_flag.transform(df["Flag"])

            # Impute + scale
            X_imp = imputer.transform(df[REQUIRED_FIELDS])
            X_scaled = scaler.transform(X_imp)

            # Predict
            preds = lgb_model.predict(X_scaled)
            labels = np.argmax(preds, axis=1)
            families = label_encoder.inverse_transform(labels)

            # Display results
            self.output_box.clear()
            self.output_box.append("=== 🔐 Prediction Results ===\n")

            for i, fam in enumerate(families):
                self.output_box.append(
                    f"Row {i+1}:   <b style='color:#6CB8FF;'>🔒 {fam}</b>"
                )

            self.output_box.append("\n--- ✔ Completed Successfully ---")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# -------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RansomGuardCSV()
    gui.show()
    sys.exit(app.exec())
