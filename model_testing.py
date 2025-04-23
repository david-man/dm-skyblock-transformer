import torch
from MASTER_model import MASTER
from constants import PREDICTION_HORIZON, COLUMNS_OF_INTEREST, ENCODINGS
import numpy as np
if __name__ == '__main__':
    model = MASTER(timesteps=PREDICTION_HORIZON,
                   features = len(COLUMNS_OF_INTEREST),
                   encodings = ENCODINGS)
    model.load_state_dict(torch.load('model.pt', weights_only=True))
    data = np.load('data/training_dataset.npy')
    labels = np.load('data/training_labels.npy')

    data = torch.from_numpy(data).float()

    indices_to_choose = np.random.choice(np.arange(len(labels)), 30)#pick 30 random indices
    overall = 0
    for k in indices_to_choose:
        test_labels = labels[k]
        test_result = model(data[k]).detach().numpy()

        test_labels = (test_labels > 1)#stock go up or stock go down
        test_result = (test_result > 1)#stock go up or stock go down
        overall += np.sum(test_labels == test_result)/len(test_result)
    print(f"INFO COEFFICIENT: {overall / len(indices_to_choose) * 2 - 1}")