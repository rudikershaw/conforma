"""MNIST classification with pytorch and conformal prediction (without the conforma wrapper classes)."""

from tempfile import gettempdir

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from conforma import calibrate_classifier
from conforma.core import compute_p_values

RANDOM_SEED = 42
COVERAGE = 0.95
MAX_SET_SIZE = 1.05


def test_mnist_classification():
    torch.manual_seed(RANDOM_SEED)
    # Load a real dataset
    full_data = datasets.MNIST(gettempdir(), download=True, transform=transforms.ToTensor())

    # Split our data into training and a held-out reserve set for calibration + testing
    train_data, reserve_data = random_split(full_data, [50000, 10000])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Define a simple NN model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 10),
    )
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for images, labels in train_loader:
        optimiser.zero_grad()
        loss_fn(model(images), labels).backward()
        optimiser.step()

    # Define a prediction function that returns per-class probabilities
    def predict(images):
        model.eval()
        with torch.no_grad():
            return model(images).softmax(dim=1)

    # Split the reserve set into calibration and test sets arbitrarily
    cal_indexes, test_indexes = range(200), range(200, len(reserve_data))
    cal_inputs = torch.stack([reserve_data[i][0] for i in cal_indexes])
    cal_labels = torch.tensor([reserve_data[i][1] for i in cal_indexes])
    test_inputs = torch.stack([reserve_data[i][0] for i in test_indexes])
    test_labels = torch.tensor([reserve_data[i][1] for i in test_indexes])

    # Full conformal prediction without constructing a wrapper:
    # compute nonconformity scores, p-values, and prediction sets directly
    calibration = calibrate_classifier(predict(cal_inputs).numpy(), cal_labels.numpy())
    test_outputs = predict(test_inputs).numpy()
    one = np.array(1, dtype=test_outputs.dtype)
    nonconformity_scores = one - test_outputs
    p_values = compute_p_values(calibration, nonconformity_scores)
    prediction_sets = p_values >= 1 - COVERAGE

    # The coverage guarantee: the true class is included at least `COVERAGE` of the time
    correct_class_included = prediction_sets[np.arange(len(test_labels)), test_labels]
    empirical_coverage = correct_class_included.mean()
    assert empirical_coverage >= COVERAGE

    # And most predictions are singletons, as requested by max_set_size
    mean_set_size = prediction_sets.sum(axis=1).mean()
    assert mean_set_size <= MAX_SET_SIZE

    # P-values are valid probabilities
    assert np.all((p_values >= 0) & (p_values <= 1))
