import numpy as np
import cv2
import matplotlib.pyplot as plt


imagen = cv2.imread('C:/Users/EdgarMauricioTrejoDe/Documents/4toSemestre/optimizacion_matemarica/tigre.jpg')

imageRgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

#Función para agregar ruido a la imagen
def ruido(imagen, sigma):  
    ruido = np.random.normal(0, sigma, imagen.shape)  
    imagenRuido = imagen + ruido * 255  
    imagenRuido = np.clip(imagenRuido, 0, 255)  
    return imagenRuido

imagenRuido = ruido(imagen, sigma=0.5)

#Convertir la imagen con ruido a RGB para graficar
imagenRuidoRgb = cv2.cvtColor(imagenRuido.astype(np.uint8), cv2.COLOR_BGR2RGB)

def descenso_gradiente_momentum(f, grad_f, x0, alpha, beta, max_iter, eps):
    x = np.array(x0, dtype=float)  
    v = np.zeros_like(x)  
    for i in range(max_iter):
        grad = grad_f(x, f)  
        norma_grad = np.linalg.norm(grad)  
        if norma_grad < eps:
            break  
        v = beta * v + (1 - beta) * grad  
        x = x - alpha * v  
    return x

# Descenso de gradiente simple
def descenso_gradiente_simple(f, grad_f, x0, alpha, lambda_param, max_iter, eps):
    for i in range(max_iter):
        grad_f_i = grad_f(x0, f, lambda_param)  
        norma_grad = np.linalg.norm(grad_f_i)
        if norma_grad < eps:
            break
        xi = x0 - alpha * grad_f_i
        x0 = xi.copy()
    return x0

# Descenso de gradiente con Nesterov
def descenso_gradiente_nesterov(f, grad_f, x0, alpha, beta, max_iter, eps):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    for i in range(max_iter):
        grad = grad_f(x + beta * v, f)  
        norma_grad = np.linalg.norm(grad)  
        if norma_grad < eps:
            break  
        v = beta * v + alpha * grad  
        x = x - v  
    return x


plt.figure(figsize=(12, 6))

#Imagen original
plt.subplot(1, 2, 1)
plt.imshow(imageRgb)
plt.title("Imagen Original")
plt.axis('off')

#Imagen con ruido 
plt.subplot(1, 2, 2)
plt.imshow(imagenRuidoRgb)
plt.title("Imagen con ruido")
plt.axis('off')

plt.show()
