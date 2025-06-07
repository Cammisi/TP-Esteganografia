import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def imagen_a_bits(img):
    datos = np.array(img).flatten()
    bits = ''.join([format(pixel, '08b') for pixel in datos])
    return bits

def bits_a_imagen(bits, shape):
    total_bits = shape[0] * shape[1] * 8
    bits = bits[:total_bits]  # Solo tomar los bits necesarios
    pixels = [int(bits[i:i+8], 2) for i in range(0, total_bits, 8)]
    arr = np.array(pixels, dtype=np.uint8).reshape(shape)
    return Image.fromarray(arr)

def obtener_indices_frecuencia_validos(shape, n):
    rows, cols = shape
    f_base = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if not (i == 0 and j == 0):  # Excluir DC
                f_base[i, j] = np.sqrt((i - rows//2)**2 + (j - cols//2)**2)
    indices = np.dstack(np.unravel_index(np.argsort(-f_base.ravel()), (rows, cols)))[0]
    
    usados = set()
    final = []
    for idx in indices:
        u, v = idx
        if len(final) >= n:
            break
        # Solo agrega si no está ya en la lista
        if (u, v) not in usados:
            final.append((u, v))
            usados.add((u, v))
    return final

def filtro_pasa_altos(img, radio=30):
    arr = np.array(img, dtype=np.float64)
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    rows, cols = arr.shape
    crow, ccol = rows // 2, cols // 2

    # Crear máscara pasa altos
    mask = np.ones((rows, cols), np.uint8)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= radio**2
    mask[mask_area] = 0

    # Aplicar máscara
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_filtrada = np.fft.ifft2(f_ishift).real
    img_filtrada = np.clip(img_filtrada, 0, 255).astype(np.uint8)
    return Image.fromarray(img_filtrada)

def ocultar_imagen_fourier(cover_path, mensaje_path, salida_path):
    cover = Image.open(cover_path).convert('L')
    mensaje = Image.open(mensaje_path).convert('L').resize(cover.size)
    shape = np.array(mensaje).shape  # Usar shape de la imagen mensaje
    bits = imagen_a_bits(mensaje)
    arr = np.array(cover, dtype=np.float64)
    f = np.fft.fft2(arr)

    # Obtener los índices donde esconder los bits
    total_bits = shape[0] * shape[1] * 8
    indices = obtener_indices_frecuencia_validos(arr.shape, total_bits)

    for k, (u, v) in enumerate(indices):
        bit = bits[k]
        a = f[u, v].real
        b = f[u, v].imag
        if bit == '0':
            f[u, v] = abs(a) + 1j * abs(b)
        else:
            f[u, v] = -abs(a) - 1j * abs(b)

        # Modificar el conjugado simétrico
        u_sym = (-u) % arr.shape[0]
        v_sym = (-v) % arr.shape[1]
        if bit == '0':
            f[u_sym, v_sym] = abs(f[u_sym, v_sym].real) + 1j * abs(f[u_sym, v_sym].imag)
        else:
            f[u_sym, v_sym] = -abs(f[u_sym, v_sym].real) - 1j * abs(f[u_sym, v_sym].imag)

    # Transformada inversa
    img_mod = np.fft.ifft2(f).real
    img_mod = np.clip(img_mod, 0, 255).astype(np.uint8)
    Image.fromarray(img_mod).save(salida_path)
    print(f"Imagen estego guardada en {salida_path}")

    # Guardamos los índices por si queremos extraer
    return indices, shape  # shape es (alto, ancho)

def extraer_imagen_fourier(estego_path, indices, shape):
    estego = Image.open(estego_path).convert('L')
    arr = np.array(estego, dtype=np.float64)
    f = np.fft.fft2(arr)
    bits = []
    total_bits = shape[0] * shape[1] * 8  # alto × ancho × 8
    for (u, v) in indices[:total_bits]:
        a = f[u, v].real
        b = f[u, v].imag
        bit = '0' if (a >= 0 and b >= 0) else '1'
        bits.append(bit)
    bits = ''.join(bits)
    # Solo tomar los bits necesarios para formar la imagen
    pixels = [int(bits[i:i+8], 2) for i in range(0, total_bits, 8)]
    print(f"Total bits extraídos: {len(bits)}")
    print(f"Total píxeles a reconstruir: {len(pixels)} (shape: {shape})")
    arr = np.array(pixels, dtype=np.uint8).reshape(shape)
    return Image.fromarray(arr)

def mostrar_espectros(img_path, titulo):
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float64)
    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitud = np.log(np.abs(fshift) + 1)
    plt.figure(figsize=(6,5))
    plt.imshow(magnitud, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def mostrar_inversa_fourier(img_path, titulo):
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float64)
    f = np.fft.fft2(arr)
    img_rec = np.fft.ifft2(f).real
    img_rec = np.clip(img_rec, 0, 255).astype(np.uint8)
    plt.figure(figsize=(6,5))
    plt.imshow(img_rec, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

# === Ejemplo de uso ===

# 0. Mostrar espectro e inversa original antes del filtro
mostrar_espectros('original.png', 'Espectro Original')
mostrar_inversa_fourier('original.png', 'Reconstrucción Inversa Original')

# 1. Aplicar filtro pasa altos a la imagen cover y guardar
cover_filtrada = filtro_pasa_altos(Image.open('original.png').convert('L'), radio=30)
cover_filtrada.save('original_pasa_altos.png')

# 2. Mostrar espectro e inversa después del filtro pasa altos
mostrar_espectros('original_pasa_altos.png', 'Espectro Pasa Altos')
mostrar_inversa_fourier('original_pasa_altos.png', 'Reconstrucción Inversa Pasa Altos')

# 3. Ocultar imagen mensaje en la imagen cover filtrada
indices, shape_mensaje = ocultar_imagen_fourier('original_pasa_altos.png', 'secreta.png', 'estego_fourier.png')
total_bits = shape_mensaje[0] * shape_mensaje[1] * 8
print(f"Índices generados: {len(indices)}, Bits necesarios: {total_bits}")
if len(indices) < total_bits:
    raise ValueError("No hay suficientes índices únicos para ocultar todos los bits.")

# 4. Extraer imagen mensaje desde la imagen estego
recuperada = extraer_imagen_fourier('estego_fourier.png', indices, shape_mensaje)
recuperada.save('mensaje_recuperado.png')

# 5. Mostrar imágenes y espectros
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(Image.open('original_pasa_altos.png').convert('L'), cmap='gray')
plt.title('Original Pasa Altos')
plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(Image.open('estego_fourier.png').convert('L'), cmap='gray')
plt.title('Estego')
plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(recuperada, cmap='gray')
plt.title('Recuperada')
plt.axis('off')
plt.show()

mostrar_espectros('original_pasa_altos.png', 'Espectro Original Pasa Altos')
mostrar_espectros('estego_fourier.png', 'Espectro Estego')
mostrar_inversa_fourier('estego_fourier.png', 'Reconstrucción Inversa Estego')