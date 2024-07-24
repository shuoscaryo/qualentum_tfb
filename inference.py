import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import json

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1600, 128)  # Adjusted input size for the fully connected layer
        self.fc2 = nn.Linear(128, 64)
        self.fc3= nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)  # Get batch size
        data_len = x.size(1)  # Get data length (number of features)
        x = x.unsqueeze(2)  # Add a channel dimension (5 channels)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(batch_size, -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

last_model= 'BTCUSD_dataset_1H_100candles_balanced_results/BTCUSD_dataset_1H_100candles_balanced_results_12.pth'
model = Model()
model.load_state_dict(torch.load(last_model))
def predict():
    # 1. Poner el modo eval
    model.eval()

    # 2. Descargar las últimas 100 velas de BTC en la frecuencia de 1H (ohlc+volume)
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': 100
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Convertir los datos a un DataFrame de pandas
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Convertir las columnas a float y guardar solo las columnas ohlc y volume
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df = df[::-1]  # Reverse the DataFrame

    # 3. Normalizar el volumen con MinMaxScaler utilizando todos los volúmenes
    scaler = MinMaxScaler()
    volumes = scaler.fit_transform(df['volume'].values.reshape(-1, 1))
    # 4. Crear una lista con esta forma Oi,Hi,Li,Ci y normalizar con MinMaxScaler
    features = df[['open', 'high', 'low', 'close']].values.reshape(-1, 1)
    features_scaled = scaler.fit_transform(features).reshape(-1, 4)
    # 5. Juntar el volumen con la lista de forma que queda Oi, Hi, Li, Ci, Vi
    features_with_volume = np.hstack((features_scaled, volumes))
    features_with_volume = features_with_volume.transpose(1,0)
    # Convertir a tensor de PyTorch
    x = torch.tensor(features_with_volume, dtype=torch.float32).unsqueeze(0)

    # 6. Hacer model.eval() output=model(x).squeeze(dim=-1)
    with torch.no_grad():
        output = model(x).squeeze(dim=-1).item()
    # 7. Si el output es más que 0.8, calcular el TP y SL
        close_prices = df['close'].values[-14:]
        std_dev = np.std(close_prices)
        last_close = df['close'].values[-1]

        tp = last_close + 0.25 * std_dev * 3
        sl = last_close - 0.125 * std_dev * 3
        result = {
            'confidence': f'{output * 100:.2f}%',
            'take_profit': tp,
            'stop_loss': sl
        }

    # 8. Devolver un JSON indicando el TP, SL si es long y si no, lo que se te ocurra
    return result

def update():
    pass

from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
async def root(name: str, timeframe: str):
    if name != "BTCUSD" or timeframe != "1H":
        raise HTTPException(status_code=501, detail="Not implemented :3 try with BTCUSD and 1H")
    return predict()

@app.post("/update_model")
async def update_model(name: str, timeframe: str):
    raise HTTPException(status_code=501, detail="Not implemented yet :3")
    if name != "BTCUSD" or timeframe != "1H":
        raise HTTPException(status_code=501, detail="Not implemented :3 try with BTCUSD and 1H")
    return update()