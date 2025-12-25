# services/ft_transformer_service.py

import os
import torch
import joblib
import pandas as pd
import numpy as np
from rtdl_revisiting_models import FTTransformer


class FTTransformerService:
    def __init__(self, model_name: str = "ft_transformer_model2"):
        # =====================================================
        # PATH
        # =====================================================
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.save_dir = os.path.join(base_dir, "models", model_name)

        if not os.path.exists(self.save_dir):
            raise FileNotFoundError(f"Model folder not found: {self.save_dir}")

        self.device = torch.device("cpu")

        # =====================================================
        # LOAD CONFIG
        # =====================================================
        self.config = joblib.load(os.path.join(self.save_dir, "config.pkl"))

        self.num_cols = self.config["numerical_features"]
        self.cat_cols = self.config["categorical_features"]

        # =====================================================
        # LOAD PREPROCESSING
        # =====================================================
        self.scaler = joblib.load(os.path.join(self.save_dir, "scaler.pkl"))
        self.cat_encoders = joblib.load(os.path.join(self.save_dir, "cat_encoders.pkl"))

        # =====================================================
        # BUILD MODEL
        # =====================================================
        self.model = FTTransformer(
            n_cont_features=len(self.num_cols),
            cat_cardinalities=self.config["cat_cardinalities"],
            d_block=self.config["d_block"],
            n_blocks=self.config["n_blocks"],
            attention_n_heads=self.config["attention_n_heads"],
            attention_dropout=self.config["attention_dropout"],
            ffn_d_hidden_multiplier=self.config["ffn_d_hidden_multiplier"],
            ffn_dropout=self.config["ffn_dropout"],
            residual_dropout=self.config["residual_dropout"],
            d_out=self.config["num_classes"]
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(
                os.path.join(self.save_dir, "model.pt"),
                map_location=self.device
            )
        )
        self.model.eval()

    # =====================================================
    # METADATA
    # =====================================================
    def get_features(self):
        return {
            "numerical": self.num_cols,
            "categorical": self.cat_cols
        }

    def get_categorical_options(self):
        return {
            col: list(self.cat_encoders[col].classes_)
            for col in self.cat_cols
        }

    # =====================================================
    # PREPROCESS
    # =====================================================
    def _preprocess(self, df: pd.DataFrame):
        df = df.copy()

        for col in self.cat_cols:
            encoder = self.cat_encoders[col]
            known = set(encoder.classes_)
            df[col] = df[col].apply(
                lambda x: encoder.transform([x])[0] if x in known else -1
            )

        X_num = self.scaler.transform(df[self.num_cols])
        X_cat = df[self.cat_cols].values.astype("int64")

        X_num = torch.tensor(X_num, dtype=torch.float32).to(self.device)
        X_cat = torch.tensor(X_cat, dtype=torch.long).to(self.device)

        return X_num, X_cat

    # =====================================================
    # SINGLE PREDICTION
    # =====================================================
    def predict(self, input_data: dict):
        df = pd.DataFrame([input_data])
        X_num, X_cat = self._preprocess(df)

        with torch.no_grad():
            logits = self.model(X_num, X_cat)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(probs.argmax())

        return pred_class, probs

    # =====================================================
    # BATCH PREDICTION
    # =====================================================
    def predict_batch(self, df: pd.DataFrame):
        X_num, X_cat = self._preprocess(df)

        with torch.no_grad():
            logits = self.model(X_num, X_cat)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        preds = probs.argmax(axis=1)

        df_out = df.copy()
        df_out["prediction"] = preds
        df_out["prob_gagal_bayar"] = probs[:, 1]
        df_out["prediction_label"] = np.where(
            preds == 1, "Gagal Bayar", "Lancar Bayar"
        )

        return df_out
