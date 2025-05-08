import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Leer imagen en escala de grises
imagen = cv2.imread('C:/Users/EdgarMauricioTrejoDe/Documents/4toSemestre/optimizacion_matemarica/tigre.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
imagen = cv2.resize(imagen, (256, 256)).astype(np.float32)

# Función para agregar ruido a la imagen
def ruido(imagen, sigma):
    ruido = np.random.normal(0, sigma, imagen.shape)
    imagenRuido = imagen + ruido 
    return np.clip(imagenRuido, 0, 255)

imagenRuido = ruido(imagen, sigma=40)  

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

# Descenso con momentum (corregido)
def descenso_gradiente_momentum(f, grad_f, x0, alpha, beta, lambda_param, max_iter, eps):
    x = x0.copy()
    v = np.zeros_like(x)
    for i in range(max_iter):
        grad = grad_f(x, imagenRuido, lambda_param)
        if np.linalg.norm(grad) < eps:
            break
        v = beta * v + (1 - beta) * grad
        x = x - alpha * v
    return x

def descenso_gradiente_nesterov(f, grad_f, x0, alpha, beta, lambda_param, max_iter, eps):
    x = x0.copy()
    y = x0.copy()
    v = np.zeros_like(x)
    for i in range(max_iter):
        x_prev = x.copy()
        grad = grad_f(y, imagenRuido, lambda_param)
        v = beta * v - alpha * grad
        x = x + v
        y = x + beta * (x - x_prev)
        if np.linalg.norm(grad) < eps:
            break
    return x

# Parámetros ajustados para mejorar la restauración
alpha = 0.01  # Tasa de aprendizaje reducida
beta = 0.9  
lambda_param = 0.1  # Regularización más fuerte
max_iter = 500  # Más iteraciones para permitir una mejor convergencia
eps = 1e-3  # Umbral de convergencia

# Aplicar métodos
resultado_simple = descenso_gradiente_simple(funcion_objetivo, gradiente, imagenRuido.copy(), alpha, lambda_param, max_iter, eps)
resultado_momentum = descenso_gradiente_momentum(funcion_objetivo, gradiente, imagenRuido.copy(), alpha, beta, lambda_param, max_iter, eps)
resultado_nesterov = descenso_gradiente_nesterov(funcion_objetivo, gradiente, imagenRuido.copy(), alpha, beta, lambda_param, max_iter, eps)

# Asegurarse de que las imágenes restauradas estén en el rango adecuado
resultado_simple = np.clip(resultado_simple, 0, 255).astype(np.uint8)
resultado_momentum = np.clip(resultado_momentum, 0, 255).astype(np.uint8)
resultado_nesterov = np.clip(resultado_nesterov, 0, 255).astype(np.uint8)

# Mostrar imágenes
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1); plt.imshow(imagen, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(imagenRuido, cmap='gray'); plt.title("Con ruido"); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(resultado_simple, cmap='gray'); plt.title("Simple"); plt.axis('off')
plt.subplot(2, 3, 4); plt.imshow(resultado_momentum, cmap='gray'); plt.title("Momentum"); plt.axis('off')
plt.subplot(2, 3, 5); plt.imshow(resultado_nesterov, cmap='gray'); plt.title("Nesterov"); plt.axis('off')
plt.tight_layout()
plt.show()

# Métricas
def calcular_metricas(ref, proc):
    ref = ref.astype(np.uint8)
    proc = np.clip(proc, 0, 255).astype(np.uint8)
    return psnr(ref, proc), ssim(ref, proc)

psnr_s, ssim_s = calcular_metricas(imagen, resultado_simple)
psnr_m, ssim_m = calcular_metricas(imagen, resultado_momentum)
psnr_n, ssim_n = calcular_metricas(imagen, resultado_nesterov)

print("\nMétricas de calidad:")
print(f"Gradiente simple:   PSNR = {psnr_s:.2f}, SSIM = {ssim_s:.4f}")
print(f"Con momentum:       PSNR = {psnr_m:.2f}, SSIM = {ssim_m:.4f}")
print(f"Con Nesterov:       PSNR = {psnr_n:.2f}, SSIM = {ssim_n:.4f}")
