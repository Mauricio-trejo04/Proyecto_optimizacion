import numpy as np
import cv2
import matplotlib.pyplot as plt


imagen = cv2.imread('C:/Users/EdgarMauricioTrejoDe/Documents/4toSemestre/optimizacion_matemarica/tigre.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Agregar ruido
def ruido(imagen, sigma):
    ruido = np.random.normal(0, sigma, imagen.shape)
    imagenRuido = imagen + ruido * 255
    return np.clip(imagenRuido, 0, 255)

imagenRuido = ruido(imagen, sigma=0.5)

# Función objetivo
def funcion_objetivo(u, f, lambda_param):
    diff = u - f
    grad_u_x = np.roll(u, -1, axis=1) - u
    grad_u_y = np.roll(u, -1, axis=0) - u
    return 0.5 * np.sum(diff**2) + 0.5 * lambda_param * (np.sum(grad_u_x**2) + np.sum(grad_u_y**2))

# Gradiente
def gradiente(u, f, lambda_param):
    laplaciano = (
        -4 * u +
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)
    )
    return (u - f) - lambda_param * laplaciano

# Descenso de gradiente simple
def descenso_gradiente_simple(f, grad_f, x0, alpha, lambda_param, max_iter, eps):
    for i in range(max_iter):
        grad = grad_f(x0, imagenRuido, lambda_param)
        if np.linalg.norm(grad) < eps:
            break
        x0 = x0 - alpha * grad
    return x0

# Descenso con momentum
def descenso_gradiente_momentum(f, grad_f, x0, alpha, beta, max_iter, eps):
    x = x0.copy()
    v = np.zeros_like(x)
    for i in range(max_iter):
        grad = grad_f(x, imagenRuido, lambda_param)
        if np.linalg.norm(grad) < eps:
            break
        v = beta * v + (1 - beta) * grad
        x = x - alpha * v
    return x

# Descenso con Nesterov
def descenso_gradiente_nesterov(f, grad_f, x0, alpha, beta, max_iter, eps):
    x = x0.copy()
    v = np.zeros_like(x)
    for i in range(max_iter):
        grad = grad_f(x + beta * v, imagenRuido, lambda_param)
        if np.linalg.norm(grad) < eps:
            break
        v = beta * v + alpha * grad
        x = x - v
    return x

# Parámetros
alpha = 0.1
beta = 0.9
lambda_param = 0.1
max_iter = 100
eps = 1e-3

# Aplicar métodos
resultado_simple = descenso_gradiente_simple(funcion_objetivo, gradiente, imagenRuido.copy(), alpha, lambda_param, max_iter, eps)
resultado_momentum = descenso_gradiente_momentum(funcion_objetivo, gradiente, imagenRuido.copy(), alpha, beta, max_iter, eps)
resultado_nesterov = descenso_gradiente_nesterov(funcion_objetivo, gradiente, imagenRuido.copy(), alpha, beta, max_iter, eps)

# Mostrar imágenes
plt.figure(figsize=(15, 6))
plt.subplot(1, 5, 1); plt.imshow(imagen, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(1, 5, 2); plt.imshow(imagenRuido, cmap='gray'); plt.title("Con ruido"); plt.axis('off')
plt.subplot(1, 5, 3); plt.imshow(resultado_simple, cmap='gray'); plt.title("Simple"); plt.axis('off')
plt.subplot(1, 5, 4); plt.imshow(resultado_momentum, cmap='gray'); plt.title("Momentum"); plt.axis('off')
plt.subplot(1, 5, 5); plt.imshow(resultado_nesterov, cmap='gray'); plt.title("Nesterov"); plt.axis('off')
plt.tight_layout()
plt.show()

