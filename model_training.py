from MASTER_model import MASTER
import numpy as np
from constants import PREDICTION_HORIZON, TRAINING_EPOCHS, ENCODINGS, FEATURES_OF_INTEREST, GATES
import torch
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    
    training_data = np.load('data/training_dataset.npy')
    training_labels = np.load('data/training_labels.npy')
    model = MASTER(timesteps=PREDICTION_HORIZON,
                   features = len(FEATURES_OF_INTEREST),#sell
                   gate_inputs = len(GATES),#sellVolume
                   encodings = ENCODINGS)
    
    x_train, x_test, y_train, y_test = train_test_split(training_data, training_labels, train_size = 0.8)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    verbose = False
    for i in range(TRAINING_EPOCHS):
        optimizer.zero_grad()
        output = model(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        test_loss = criterion(model(x_test), y_test)
        if(verbose):
            print(f"TRAIN LOSS: {loss}, TEST LOSS: {test_loss}")
    torch.save(model.state_dict(), "model.pt")
