from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from esn import EchoStateNetwork as ESN


image_path = Path('./images').resolve()
image_path.mkdir(mode=0o775, parents=True, exist_ok=True)

init_length = 1000
train_length = 3000
test_length = 1000
tmax = train_length + test_length

x = np.linspace(0, 100 * np.pi, tmax)
data = np.sin(x).reshape(-1, 1)

model = ESN(
    init_length=init_length,
    train_length=train_length,
    test_length=test_length,
    tmax=tmax
)

d = data[init_length + 1:train_length + 1]

model.fit(data[:train_length], d)
predict_result = model.predict(data[train_length:])

# Plot options
bbox_to_anchor = (1.02, 0)
bbox_loc = 'lower left'

fig = plt.figure(figsize=(10, 8))

# Plot prediction
ax1 = fig.add_subplot(211)
index = np.arange(train_length, tmax)
ax1.plot(index, data[train_length:], label='input data')
ax1.plot(index, predict_result[train_length:-1, 0], label='predict result')
ax1.set_ylim(-1.1, 1.1)
ax1.set_xlabel(r'$n$')
ax1.set_ylabel(r'$y_n$')
ax1.legend(bbox_to_anchor=bbox_to_anchor, loc=bbox_loc, borderaxespad=0)
ax1.set_title('Prediction')

# Plot prediction (enlarged)
ax2 = fig.add_subplot(212)
enlarge_length = 200
enlarge_tmax = train_length + enlarge_length
index = np.arange(train_length, enlarge_tmax)
ax2.plot(index, data[train_length:enlarge_tmax], label='input data')
ax2.plot(index, predict_result[train_length:enlarge_tmax, 0], label='predict result')
ax2.legend(bbox_to_anchor=bbox_to_anchor, loc=bbox_loc, borderaxespad=0)
ax1.set_ylim(-1.1, 1.1)
ax2.set_xlabel(r'$n$')
ax2.set_ylabel(r'$y_n$')
ax2.set_title('Prediction (enlarged)')

fig.subplots_adjust(right=0.8, hspace=0.5)

plt.savefig(image_path / 'sinusoid.png', bbox_inches='tight', pad_inches=0.1)
