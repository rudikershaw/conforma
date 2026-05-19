# Performing Conformal Prediction without the Wrapper class

The `conforma` wrappers provide the plumbing for conformal prediction steps, data & input validation, and the means to ensure your data precision is maintained. It is possible, using this library, to perform comformal prediction without using the wrapper classes (ConformalClassifier or ConformalRegressor). You may choose to do this to gain more control over how predictions are made. For example, to change the nonconformity score function, to produce p-values alongside predictions each time, or to introduce custom validation per prediction.

The following code is an example of using conforma to calibrate a pytorch NN classifier from the project's integration tests. You can find the code file in this project under [tests/integration/test_no_wrapper_workflow.py](tests/integration/test_no_wrapper_workflow.py). After reading through the example below, you can find a discussion of what each step achieves below the code example.

<!-- INSERT_CODE:tests/integration/test_no_wrapper_workflow.py -->
```py
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
```
<!--CODE_END -->

First we pull in our dataset, split our data into training and a reserve sets, define our NN architecture, and then train our model on the training set. This is all achieved outside of the scope of the conformal library (in this case using pytorch).

In this example, we assume that we have already determined optimal coverage, a maximum average predicition set size, and the required size of the calibration set. For a sensible default for determining these values, please see our other examples using the calibration plan functions.

Next, we split our reserve dataset into test and calibration sets using a calibration set size of 200. 

The next logical section of code contains the complete steps for performing conformal prediction on the test set, without using the `conforma` wrapper classes. First, we calibrate the model using the calibration set. The `calibrate_classifier` function takes the predicted probabilities and true labels from the calibration set and computes the necessary calibration data. Notice the use of the `.numpy()` functions on the pytorch Tensors. `conforma` uses numpy arrays for it's API, so we are required to convert values. 

Next we produce nonconformity scores from our predictions on the test set. Here we use the "Absolute Residual" to produce the scores. Absolute Residual (1 - outputs) is the default score used by the classifier wrapper. If you wish to switch to a specialised noncomformity score, this is where you would do it. Notice that we produce a numpy array for the value `1`, with a `dtype` matching the datatype of our model output. We do this to ensure that our nonconformity score maintains user specified precisions. This is handled automatically by the wrapper, and development time guarentees ensure that the public APIs of the wrappers always preserve user provided precision. Without the wrappers, we will have to remember to do this manually.

Now that we have our nonconformity scores for the model output, we can now produce p-values using the `compute_p_values` function provided by `conforma.core`. With our p-values and our required coverage, we then convert the p-values into a prediction set by producing booleans for each value by checking that they are `>= 1 - COVERAGE`. The prediction set is a boolean array indicating which classes are included in the prediction for each input. The coverage guarantee ensures that the true class is included in the prediction set at least as often as the specified coverage level. This means that the new conformal predictions may include multiple true values when the model is uncertain (which is how it maintains the coverage guarantee).

In the case where a prediction set contains multiple true values, you may still need to select a single class for your use-case. Without the wrapper we already have the p-values for each output. We may use these values to determine which single class the model is most sure of, and also to compare the model's confidence between classes.