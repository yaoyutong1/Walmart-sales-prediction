import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

layers = keras.layers
models = keras.models

class WalmartSalesPredictor:
    def __init__(self):
        self.df = None
        self.features = None
        self.target = "Weekly_Sales"
        self.numeric_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    # Load data
    def load_data(self, file_path="Walmart.csv"):
        print("正在加载数据...")
        self.df = pd.read_csv(file_path)
        print(f"数据加载完成，共有 {len(self.df)} 行记录")
        return self.df

    # Data exploration
    def explore_data(self):
        print("\n数据概览:")
        print(self.df.head())

        print("\n数据信息:")
        print(self.df.info())

        print("\n描述性统计:")
        print(self.df.describe())

        print("\n缺失值检查:")
        print(self.df.isnull().sum())

        print(f"\n共有 {self.df['Store'].nunique()} 个店铺")
        print(f"日期范围: {self.df['Date'].min()} 到 {self.df['Date'].max()}")

    # Feature engineering
    def create_features(self):
        print("\n正在创建特征...")
        self.df["Date"] = pd.to_datetime(self.df["Date"], format="%d-%m-%Y")
        self.df["Year"] = self.df["Date"].dt.year
        self.df["Month"] = self.df["Date"].dt.month
        self.df["Week"] = self.df["Date"].dt.isocalendar().week.astype(int)
        self.df["DayOfYear"] = self.df["Date"].dt.dayofyear

        self.df.sort_values(['Store', 'Date'], inplace=True)
        for lag in [1, 2, 3, 4]:
            self.df[f"Sales_Lag_{lag}"] = self.df.groupby('Store')['Weekly_Sales'].shift(lag)

        for window in [4, 8, 12]:
            self.df[f"Sales_MA_{window}"] = self.df.groupby('Store')['Weekly_Sales'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        self.features = [
            "Store", "Holiday_Flag", "Temperature", "Fuel_Price",
            "CPI", "Unemployment", "Year", "Month", "Week", "DayOfYear"
        ]
        for lag in [1, 2, 3, 4]:
            self.features.append(f"Sales_Lag_{lag}")
        for window in [4, 8, 12]:
            self.features.append(f"Sales_MA_{window}")

        self.df.fillna(method='bfill', inplace=True)
        print("特征创建完成")
        return self.df

    # Create sequences
    def create_sequences(self, seq_length=12):
        print("\n正在创建序列...")
        X, y = [], []

        for store_id in self.df["Store"].unique():
            store_data = self.df[self.df["Store"] == store_id].copy()
            store_data = store_data.sort_values("Date")

            for i in range(len(store_data) - seq_length):
                seq_x = store_data.iloc[i:i + seq_length][self.features].values
                seq_y = store_data.iloc[i + seq_length][self.target]
                X.append(seq_x)
                y.append(seq_y)

        self.X, self.y = np.array(X), np.array(y)
        print(f"创建了 {len(X)} 个序列，每个序列长度为 {seq_length}")
        return self.X, self.y

    # Prepare data
    def prepare_data(self, test_size=0.2):
        print("\n正在准备训练和测试数据...")
        split_idx = int(len(self.X) * (1 - test_size))

        store_ids = self.X[:, :, 0].astype("int32")
        numeric_features = self.X[:, :, 1:].astype("float32")
        labels = self.y.astype("float32")

        train_store, test_store = store_ids[:split_idx], store_ids[split_idx:]
        train_numeric, test_numeric = numeric_features[:split_idx], numeric_features[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]

        train_numeric_reshaped = train_numeric.reshape(-1, train_numeric.shape[-1])
        test_numeric_reshaped = test_numeric.reshape(-1, test_numeric.shape[-1])

        train_numeric_scaled = self.numeric_scaler.fit_transform(train_numeric_reshaped)
        test_numeric_scaled = self.numeric_scaler.transform(test_numeric_reshaped)

        train_numeric = train_numeric_scaled.reshape(train_numeric.shape)
        test_numeric = test_numeric_scaled.reshape(test_numeric.shape)

        train_labels = self.target_scaler.fit_transform(train_labels.reshape(-1, 1)).flatten()
        test_labels = self.target_scaler.transform(test_labels.reshape(-1, 1)).flatten()

        self.train_inputs = [train_store, train_numeric]
        self.train_labels = train_labels
        self.test_inputs = [test_store, test_numeric]
        self.test_labels = test_labels

        print(f"训练集: {len(train_labels)} 个样本")
        print(f"测试集: {len(test_labels)} 个样本")
        return self.train_inputs, self.train_labels, self.test_inputs, self.test_labels

    # Build model
    def build_model(self):
        print("\n正在构建模型...")
        store_input = layers.Input(shape=(self.train_inputs[0].shape[1],), name="store_id")
        numeric_input = layers.Input(shape=(self.train_inputs[1].shape[1], self.train_inputs[1].shape[2]), name="numeric")

        embed = layers.Embedding(
            input_dim=self.df["Store"].nunique() + 1,
            output_dim=8,
            embeddings_regularizer=keras.regularizers.l2(1e-5)
        )(store_input)

        x = layers.Concatenate(axis=-1)([embed, numeric_input])
        x = layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                       kernel_regularizer=keras.regularizers.l2(1e-5))(x)
        x = layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2,
                       kernel_regularizer=keras.regularizers.l2(1e-5))(x)
        x = layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-5))(x)

        output = layers.Dense(1)(x)
        model = models.Model(inputs=[store_input, numeric_input], outputs=output)
        optimizer = keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        print("模型构建完成")
        return model

    # Train model
    def train_model(self, model, epochs=50, batch_size=32):
        print("\n开始训练模型...")
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        history = model.fit(
            self.train_inputs, self.train_labels,
            validation_data=(self.test_inputs, self.test_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print("模型训练完成")
        return history

    # Evaluate model
    def evaluate_model(self, model):
        print("\n正在评估模型...")
        y_pred_scaled = model.predict(self.test_inputs)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
        y_true = self.target_scaler.inverse_transform(self.test_labels.reshape(-1, 1))

        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print("=" * 50)
        print("模型评估结果:")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.4f}")
        print(f"RMSE: {rmse:.2f}")
        print("=" * 50)

        print("\n预测样例（真实尺度）:")
        for i in range(5):
            print(f"预测: {y_pred[i][0]:.2f}, 实际: {y_true[i][0]:.2f}")
        return y_true, y_pred, r2, mape, rmse

    # Visualization
    def plot_results(self, history, y_true, y_pred):
        print("\n正在生成可视化图表...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        axes[0, 0].plot(history.history["loss"], label="Train Loss")
        axes[0, 0].plot(history.history["val_loss"], label="Validation Loss")
        axes[0, 0].set_xlabel("Epochs")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(y_true[:200], label="True Sales", alpha=0.7)
        axes[0, 1].plot(y_pred[:200], label="Predicted Sales", alpha=0.7)
        axes[0, 1].set_title("True vs Predicted (First 200 Samples)")
        axes[0, 1].set_xlabel("Sample Index")
        axes[0, 1].set_ylabel("Weekly Sales")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        residuals = y_true - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel("Predicted")
        axes[1, 0].set_ylabel("Residuals")
        axes[1, 0].set_title("Residual Plot")
        axes[1, 0].grid(True)

        axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
        max_val = max(np.max(y_true), np.max(y_pred))
        min_val = min(np.min(y_true), np.min(y_pred))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 1].set_xlabel("True Values")
        axes[1, 1].set_ylabel("Predicted Values")
        axes[1, 1].set_title("True vs Predicted")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("walmart_sales_predictions.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("图表已保存为 walmart_sales_predictions.png")

    # Save model
    def save_model(self, model, file_path="walmart_sales_model.h5"):
        model.save(file_path)
        print(f"\n模型已保存为 {file_path}")

    # Full pipeline
    def run_pipeline(self, file_path="Walmart.csv", seq_length=12, epochs=50, batch_size=32):
        print("开始沃尔玛销售额预测流程")
        print("=" * 50)
        self.load_data(file_path)
        self.explore_data()
        self.create_features()
        self.create_sequences(seq_length)
        self.prepare_data()
        model = self.build_model()
        model.summary()
        history = self.train_model(model, epochs, batch_size)
        y_true, y_pred, r2, mape, rmse = self.evaluate_model(model)
        self.plot_results(history, y_true, y_pred)
        self.save_model(model)
        print("\n流程完成!")
        return model, history, y_true, y_pred


if __name__ == "__main__":
    predictor = WalmartSalesPredictor()
    model, history, y_true, y_pred = predictor.run_pipeline(
        file_path="Walmart.csv",
        seq_length=12,
        epochs=50,
        batch_size=32
    )
