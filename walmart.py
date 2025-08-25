import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras

layers = keras.layers
models = keras.models

class WalmartSalesPredictor:
    def __init__(self):
        self.df = None
        self.features = None
        self.target = "Weekly_Sales"
        self.numeric_scaler = StandardScaler()
        self.target_scaler = StandardScaler()



    def load_data(self, file_path="Walmart.csv"):
        self.df = pd.read_csv(file_path)
        print(f"数据加载完成，共 {len(self.df)} 行")
        return self.df



    def create_features(self):
        self.df["Date"] = pd.to_datetime(self.df["Date"], format="%d-%m-%Y")
        self.df["Year"] = self.df["Date"].dt.year
        self.df["Month"] = self.df["Date"].dt.month
        self.df["Week"] = self.df["Date"].dt.isocalendar().week.astype(int)

        self.df.sort_values(["Store", "Date"], inplace=True)

        for lag in [1, 2, 3, 4]:
            self.df[f"Sales_Lag_{lag}"] = self.df.groupby("Store")["Weekly_Sales"].shift(lag)

        for window in [4, 8, 12]:
            self.df[f"Sales_MA_{window}"] = self.df.groupby("Store")["Weekly_Sales"].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        self.features = [
            "Holiday_Flag", "Temperature", "Fuel_Price",
            "CPI", "Unemployment", "Year", "Month", "Week"
        ]
        for lag in [1, 2, 3, 4]:
            self.features.append(f"Sales_Lag_{lag}")
        for window in [4, 8, 12]:
            self.features.append(f"Sales_MA_{window}")

        self.df.fillna(method="bfill", inplace=True)
        return self.df



    def create_sequences(self, seq_length=12):
        X, y, stores = [], [], []

        for store_id in self.df["Store"].unique():
            store_data = self.df[self.df["Store"] == store_id].sort_values("Date")

            for i in range(len(store_data) - seq_length):
                seq_x = store_data.iloc[i:i+seq_length][self.features].values
                seq_y = store_data.iloc[i+seq_length][self.target]

                X.append(seq_x)
                y.append(seq_y)
                stores.append(store_id)

        self.X, self.y, self.store_ids = np.array(X), np.array(y), np.array(stores)
        print(f"创建了 {len(X)} 个序列，每个序列长度 {seq_length}")
        return self.X, self.y, self.store_ids


    def prepare_data(self, test_size=0.2):
        split_idx = int(len(self.X) * (1 - test_size))

        train_X, test_X = self.X[:split_idx], self.X[split_idx:]
        train_y, test_y = self.y[:split_idx], self.y[split_idx:]
        train_store, test_store = self.store_ids[:split_idx], self.store_ids[split_idx:]

        # 标准化数值特征
        train_X_reshaped = train_X.reshape(-1, train_X.shape[-1])
        test_X_reshaped = test_X.reshape(-1, test_X.shape[-1])

        train_X_scaled = self.numeric_scaler.fit_transform(train_X_reshaped)
        test_X_scaled = self.numeric_scaler.transform(test_X_reshaped)

        train_X = train_X_scaled.reshape(train_X.shape)
        test_X = test_X_scaled.reshape(test_X.shape)

        # 目标值缩放
        train_y = self.target_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()
        test_y = self.target_scaler.transform(test_y.reshape(-1, 1)).flatten()

        # store_id 作为单个输入
        self.train_inputs = [train_store, train_X]
        self.test_inputs = [test_store, test_X]
        self.train_labels, self.test_labels = train_y, test_y

        print(f"训练集 {len(train_y)} 测试集 {len(test_y)}")
        return self.train_inputs, self.train_labels, self.test_inputs, self.test_labels



    def build_model(self):
        store_input = layers.Input(shape=(1,), name="store_id")
        numeric_input = layers.Input(shape=(self.train_inputs[1].shape[1], self.train_inputs[1].shape[2]), name="numeric")

        embed = layers.Embedding(input_dim=self.df["Store"].nunique()+1, output_dim=8)(store_input)
        embed = layers.Flatten()(embed)
        embed = layers.RepeatVector(numeric_input.shape[1])(embed)

        x = layers.Concatenate(axis=-1)([embed, numeric_input])

        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        output = layers.Dense(1)(x)

        model = models.Model(inputs=[store_input, numeric_input], outputs=output)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model


    def train_model(self, model, epochs=30, batch_size=32):
        history = model.fit(
            self.train_inputs, self.train_labels,
            validation_data=(self.test_inputs, self.test_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=1
        )
        return history

        

    def evaluate_model(self, model):
        y_pred_scaled = model.predict(self.test_inputs)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        y_true = self.target_scaler.inverse_transform(self.test_labels.reshape(-1, 1))

        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print("模型评估结果:")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.4f}")
        print(f"RMSE: {rmse:.2f}")

        print("\n预测样本:")
        for i in range(5):
            print(f"Predicted: {y_pred[i][0]:.2f}, Actual: {y_true[i][0]:.2f}")
        self.plot_results(y_true, y_pred)

        return y_true, y_pred

    def plot_results(self, y_true, y_pred):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(y_true[:100], label="Actual", marker="o", alpha=0.7)
        plt.plot(y_pred[:100], label="Predicted", marker="x", alpha=0.7)
        plt.title("Predicted vs Actual (First 100 samples)")
        plt.xlabel("Sample Index")
        plt.ylabel("Weekly Sales")
        plt.legend()

if __name__ == "__main__":
    predictor = WalmartSalesPredictor()
    predictor.load_data("Walmart.csv")
    predictor.create_features()
    predictor.create_sequences(seq_length=12)
    predictor.prepare_data()
    model = predictor.build_model()
    model.summary()
    predictor.train_model(model, epochs=20)
    predictor.evaluate_model(model)
