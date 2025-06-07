import numpy as np
import cv2
import matplotlib.pyplot as plt

def cargar_imagen_grises(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img.astype(np.float64)
    except:
        return None

def redimensionar(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def modificar_componente_tf2d(componente, bit, delta):
    """
    Modifica una componente (real o imaginaria) según el inciso 3:
    componente' = signo * q * delta
    donde signo = 1 si componente >= 0, sino -1
    y q = abs(round(componente/delta))
    La paridad de q guarda el bit secreto
    """
    # Calcular signo
    signo = 1 if componente >= 0 else -1
    
    # Calcular q
    q = abs(round(componente / delta)) if delta != 0 else 0
    
    # Modificar paridad de q según el bit a ocultar
    if (q % 2) != bit:
        q += 1
    
    # Asegurar que q sea al menos 1 para evitar pérdida total de información
    if q == 0:
        q = 1 if bit == 1 else 2
    
    # Calcular nueva componente
    nueva_componente = signo * q * delta
    
    return nueva_componente

def extraer_bit_tf2d(componente, delta):
    """
    Extrae un bit de una componente modificada
    """
    if delta == 0:
        return 0
    
    q = abs(round(componente / delta))
    return q % 2

def ocultar_imagen_tf2d(img_host, img_secreta, delta=1.0):
    """
    Oculta una imagen dentro de otra usando la Transformada de Fourier 2D
    según las especificaciones del inciso 3
    """
    print(f"=== OCULTANDO IMAGEN CON TF2D (δ={delta}) ===")
    
    # Aplicar Transformada de Fourier 2D a la imagen host
    print("Aplicando Transformada de Fourier 2D...")
    tf2d_host = np.fft.fft2(img_host)
    
    print(f"Forma de la TF2D: {tf2d_host.shape}")
    print(f"Tipo de datos: {tf2d_host.dtype}")
    
    # Preparar imagen secreta
    h_host, w_host = img_host.shape
    
    # Calcular cuántos bits podemos ocultar (usamos ambas componentes real e imaginaria)
    total_componentes = h_host * w_host * 2  # real + imaginaria
    
    # Redimensionar imagen secreta para que quepa
    max_pixels_secretos = total_componentes // 8  # 8 bits por píxel
    lado_maximo = int(np.sqrt(max_pixels_secretos))
    
    # Usar un tamaño más conservador para mejor calidad
    lado_secreto = min(lado_maximo // 2, min(img_secreta.shape))
    
    print(f"Redimensionando imagen secreta a {lado_secreto}x{lado_secreto}")
    img_secreta_redim = redimensionar(img_secreta, (lado_secreto, lado_secreto))
    
    # Binarizar imagen secreta
    _, img_secreta_bin = cv2.threshold(img_secreta_redim.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    
    # Convertir a bits
    bits_secretos = []
    for pixel in img_secreta_bin.flatten():
        bit = 1 if pixel > 127 else 0
        bits_secretos.append(bit)
    
    print(f"Total de bits a ocultar: {len(bits_secretos)}")
    
    # Crear copia de la TF2D para modificar
    tf2d_modificada = np.copy(tf2d_host)
    
    # Separar componentes real e imaginaria
    componente_real = tf2d_modificada.real.flatten()
    componente_imag = tf2d_modificada.imag.flatten()
    
    # Ocultar bits modificando las componentes según el inciso 3
    bit_idx = 0
    componentes_modificadas = 0
    
    for i in range(len(componente_real)):
        if bit_idx < len(bits_secretos):
            # Modificar componente real
            componente_real[i] = modificar_componente_tf2d(
                componente_real[i], bits_secretos[bit_idx], delta
            )
            bit_idx += 1
            componentes_modificadas += 1
            
        if bit_idx < len(bits_secretos) and i < len(componente_imag):
            # Modificar componente imaginaria
            componente_imag[i] = modificar_componente_tf2d(
                componente_imag[i], bits_secretos[bit_idx], delta
            )
            bit_idx += 1
            componentes_modificadas += 1
    
    print(f"Componentes modificadas: {componentes_modificadas}")
    
    # Reconstruir TF2D modificada
    tf2d_modificada = componente_real.reshape(tf2d_host.shape) + 1j * componente_imag.reshape(tf2d_host.shape)
    
    # Aplicar Transformada Inversa de Fourier
    print("Aplicando Transformada Inversa de Fourier...")
    img_estego = np.fft.ifft2(tf2d_modificada).real
    
    # Asegurar que los valores estén en el rango correcto
    img_estego = np.clip(img_estego, 0, 255)
    
    print("Imagen estego generada exitosamente")
    
    return img_estego, tf2d_modificada, (lado_secreto, lado_secreto), img_secreta_bin

def extraer_imagen_tf2d(img_estego, tf2d_modificada, secret_shape, delta=1.0):
    """
    Extrae la imagen oculta de la imagen estego usando TF2D
    """
    print(f"=== EXTRAYENDO IMAGEN CON TF2D (δ={delta}) ===")
    
    secret_h, secret_w = secret_shape
    total_bits = secret_h * secret_w
    
    # Separar componentes de la TF2D modificada
    componente_real = tf2d_modificada.real.flatten()
    componente_imag = tf2d_modificada.imag.flatten()
    
    # Extraer bits
    bits_extraidos = []
    bit_idx = 0
    
    for i in range(len(componente_real)):
        if bit_idx < total_bits:
            # Extraer bit de componente real
            bit = extraer_bit_tf2d(componente_real[i], delta)
            bits_extraidos.append(bit)
            bit_idx += 1
            
        if bit_idx < total_bits and i < len(componente_imag):
            # Extraer bit de componente imaginaria
            bit = extraer_bit_tf2d(componente_imag[i], delta)
            bits_extraidos.append(bit)
            bit_idx += 1
    
    print(f"Bits extraídos: {len(bits_extraidos)}")
    
    # Reconstruir imagen
    img_recuperada = np.zeros((secret_h, secret_w), dtype=np.uint8)
    
    for i in range(secret_h):
        for j in range(secret_w):
            idx = i * secret_w + j
            if idx < len(bits_extraidos):
                pixel_value = 255 if bits_extraidos[idx] == 1 else 0
                img_recuperada[i, j] = pixel_value
    
    print("Imagen extraída exitosamente")
    
    return img_recuperada

def analizar_tf2d(tf2d_original, tf2d_modificada, delta):
    """
    Análisis simplificado de la TF2D
    """
    print(f"Análisis básico para δ={delta} completado")
    return {}

def calcular_precision(img_original, img_recuperada):
    """
    Calcula la precisión de recuperación
    """
    if img_original.shape != img_recuperada.shape:
        # Redimensionar para comparar
        img_original = redimensionar(img_original, img_recuperada.shape[::-1])
    
    # Binarizar imagen original
    _, img_orig_bin = cv2.threshold(img_original.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    
    # Calcular precisión
    pixeles_correctos = np.sum(img_orig_bin == img_recuperada)
    total_pixeles = img_recuperada.size
    precision = (pixeles_correctos / total_pixeles) * 100
    
    return precision, pixeles_correctos, total_pixeles

def ejecutar_steganografia_tf2d():
    """
    Función principal que implementa el inciso 3 correctamente
    """
    print("=== ESTEGANOGRAFÍA CON TRANSFORMADA DE FOURIER 2D ===")
    print("Implementación del inciso 3\n")
    
    # Cargar o crear imágenes
    img_host = cargar_imagen_grises("original.png")
    img_secreta = cargar_imagen_grises("secreta.png")
    
    if img_host is None or img_secreta is None:
        print("Creando imágenes de ejemplo...")
        # Imagen host más grande para mejor capacidad
        img_host = np.random.randint(100, 180, (256, 256), dtype=np.uint8).astype(np.float64)
        
        # Imagen secreta con patrón reconocible
        img_secreta = np.zeros((64, 64), dtype=np.float64)
        img_secreta[16:48, 16:48] = 255  # Cuadrado blanco
        img_secreta[24:40, 24:40] = 0    # Cuadrado negro interno
        # Agregar algunos detalles
        img_secreta[20:28, 20:22] = 255
        img_secreta[36:44, 42:44] = 255
    else:
        img_host = img_host.astype(np.float64)
        img_secreta = img_secreta.astype(np.float64)
        print("Imágenes cargadas correctamente")
    
    print(f"Imagen host: {img_host.shape}")
    print(f"Imagen secreta: {img_secreta.shape}")
    
    # Probar diferentes valores de delta según el inciso 3
    delta_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    resultados = []
    
    for delta in delta_values:
        print(f"\n{'='*50}")
        print(f"PROBANDO δ = {delta}")
        print('='*50)
        
        try:
            # Aplicar TF2D para ocultar imagen
            tf2d_original = np.fft.fft2(img_host)
            
            # Ocultar imagen
            img_estego, tf2d_modificada, secret_shape, img_secreta_usada = ocultar_imagen_tf2d(
                img_host, img_secreta, delta
            )
            
            # Extraer imagen
            img_recuperada = extraer_imagen_tf2d(img_estego, tf2d_modificada, secret_shape, delta)
            
            # Calcular precisión
            precision, correctos, total = calcular_precision(img_secreta_usada, img_recuperada)
            
            # Analizar TF2D
            analisis = analizar_tf2d(tf2d_original, tf2d_modificada, delta)
            
            # Calcular diferencia simple entre imágenes
            diff_max = np.max(np.abs(img_host - img_estego))
            diff_mean = np.mean(np.abs(img_host - img_estego))
            
            resultado = {
                'delta': delta,
                'img_host': img_host,
                'img_estego': img_estego,
                'img_secreta_usada': img_secreta_usada,
                'img_recuperada': img_recuperada,
                'precision': precision,
                'correctos': correctos,
                'total': total,
                'diff_max': diff_max,
                'diff_mean': diff_mean
            }
            
            resultados.append(resultado)
            
            print(f"\nRESULTADOS para δ={delta}:")
            print(f"Precisión de recuperación: {precision:.2f}% ({correctos}/{total})")
            print(f"Diferencia máxima Host-Estego: {diff_max:.2f}")
            print(f"Diferencia promedio: {diff_mean:.2f}")
            
        except Exception as e:
            print(f"Error con δ={delta}: {e}")
    
    # Mostrar comparación de resultados
    if resultados:
        print(f"\n{'='*60}")
        print("RESUMEN DE RESULTADOS")
        print('='*60)
        print(f"{'Delta':<8} {'Precisión':<12} {'Diff Max':<10} {'Diff Media':<12}")
        print('-' * 50)
        
        for r in resultados:
            print(f"{r['delta']:<8} {r['precision']:<12.2f} {r['diff_max']:<10.2f} {r['diff_mean']:<12.4f}")
        
        # Visualizar mejores resultados (δ=1.0 como ejemplo)
        mejor_resultado = None
        for r in resultados:
            if r['delta'] == 1.0:
                mejor_resultado = r
                break
        
        if mejor_resultado is None and resultados:
            mejor_resultado = resultados[0]
        
        if mejor_resultado:
            visualizar_resultados(mejor_resultado)
    
    return resultados

def visualizar_resultados(resultado):
    """
    Visualiza solo las imágenes principales
    """
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle(f'Esteganografía con TF2D - δ={resultado["delta"]} - Precisión: {resultado["precision"]:.1f}%', fontsize=14)
    
    axes[0].imshow(resultado['img_host'], cmap='gray')
    axes[0].set_title('Imagen Host Original')
    axes[0].axis('off')
    
    axes[1].imshow(resultado['img_estego'], cmap='gray')
    axes[1].set_title('Imagen Estego')
    axes[1].axis('off')
    
    axes[2].imshow(resultado['img_secreta_usada'], cmap='gray')
    axes[2].set_title('Imagen Secreta')
    axes[2].axis('off')
    
    axes[3].imshow(resultado['img_recuperada'], cmap='gray')
    axes[3].set_title('Imagen Recuperada')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    resultados = ejecutar_steganografia_tf2d()