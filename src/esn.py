from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class EchoStateNetwork:
    N_in: int = 1
    N_out: int = 1
    N: int = 1000

    init_length: int = 1000
    train_length: int = 3000
    test_length: int = 1000
    tmax: int = train_length + test_length

    spr: float = 0.9
    p_connect: float = 0.1
    la_ridge: float = 1e-3

    W_in: np.ndarray = field(init=False)
    W_out: np.ndarray = field(init=False)
    W: np.ndarray = field(init=False)
    X: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    f: Callable = np.tanh

    def __post_init__(self):
        self.W_in = np.random.rand(self.N, self.N_in) * 2 - 1

        self.W_out = np.zeros((self.N, self.N_out))

        self.W = np.random.choice([0., 1.], (self.N, self.N), p=[(1 - self.p_connect), self.p_connect])
        self.W *= np.random.rand(self.N, self.N) * 2 - 1
        self.W = self.spr * self.W / np.max(np.abs(np.linalg.eigvals(self.W)))

        self.X = np.zeros((self.tmax + 1, self.N))
        self.y = np.zeros((self.tmax + 1, self.N_out))

    def one_step(self, u: np.ndarray, t: int):
        self.X[t + 1] = self.f(self.W @ self.X[t] + self.W_in @ u)
        self.y[t + 1] = self.W_out.T @ self.X[t + 1]

    def update_weight(self, d: np.ndarray):
        X_train = self.X[self.init_length:self.train_length]
        reg_term = self.la_ridge * np.eye(self.N)
        self.W_out = np.linalg.inv(X_train.T @ X_train + reg_term) @ X_train.T @ d

    def fit(self, train_data: np.ndarray, d: np.ndarray):
        for t in range(self.train_length):
            self.one_step(train_data[t], t)

        self.update_weight(d)

        self.y[t + 1] = self.W_out.T @ self.X[t + 1]

    def predict(self, test_data) -> np.ndarray:
        for t in range(self.test_length):
            self.one_step(test_data[t], t + self.train_length)

        return self.y
