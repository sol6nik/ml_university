import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Создание искусственного набора данных
np.random.seed(0)
n_samples = 100
X = np.linspace(-10, 10, n_samples)
# Генерация данных с квадратичной зависимостью: y = 2x^2 + 3x + 5 + шум
a, b, c = 2, 3, 5
noise = np.random.normal(0, 10, n_samples)
y = a * X**2 + b * X + c + noise

# 2. Инициализация переменных и определение функции потерь
# Модель будет иметь форму y_pred = w2 * X^2 + w1 * X + w0
W2 = tf.Variable(np.random.randn(), name="weight2")
W1 = tf.Variable(np.random.randn(), name="weight1")
W0 = tf.Variable(np.random.randn(), name="bias")


# Определение функции для предсказания
def model(X):
    return W2 * X**2 + W1 * X + W0


# Функция потерь: среднеквадратичная ошибка (MSE)
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# 3. Реализация градиентного спуска
optimizer = tf.optimizers.SGD(learning_rate=0.0001)


# Процесс обучения
def train_step(X, y):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = loss_fn(y, y_pred)
    gradients = tape.gradient(loss, [W2, W1, W0])
    optimizer.apply_gradients(zip(gradients, [W2, W1, W0]))
    return loss


# 4. Тренировка модели
epochs = 1000
loss_history = []

for epoch in range(epochs):
    loss = train_step(X, y)
    loss_history.append(loss.numpy())

# Визуализация изменения функции потерь
plt.figure(figsize=(12, 6))
plt.plot(loss_history)
plt.title("Изменение функции потерь в процессе обучения")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Построение предсказанной модели
y_pred = model(X)

# Визуализация аппроксимации исходных данных
plt.figure(figsize=(12, 6))
plt.scatter(X, y, label="Изначальные данные", color="blue")
plt.plot(X, y_pred, label="Аппроксимация моделью", color="red")
plt.title("Аппроксимация моделью квадратичной зависимости")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Возвращаем последние значения коэффициентов модели
W2.numpy(), W1.numpy(), W0.numpy()