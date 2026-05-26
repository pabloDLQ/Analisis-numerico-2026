import cv2
import os
from src.cargar_imagenes_jpg import cargar_imagen


def rotar_y_guardar_imagen(num_imagen, angulo_rotacion):
    """
    Carga una imagen, la rota el ángulo especificado y la guarda en "/imagenes creadas".
    
    Args:
        num_imagen (int): Número de la imagen a cargar (1-7)
        angulo_rotacion (float): Ángulo de rotación en grados (positivo = rotación antihoraria)
    
    Returns:
        dict: Diccionario con información del proceso:
            - 'exito': Boolean indicando si la operación fue exitosa
            - 'ruta_guardada': Ruta completa del archivo guardado (si fue exitoso)
            - 'mensaje': Mensaje descriptivo del resultado
            - 'angulo_aplicado': Ángulo de rotación aplicado
    """
    
    try:
        # Cargar imagen
        imagen = cargar_imagen(num_imagen)
        
        if imagen is None:
            return {
                'exito': False,
                'ruta_guardada': None,
                'mensaje': f"Error: No se pudo cargar la imagen número {num_imagen}",
                'angulo_aplicado': None
            }
        
        # Obtener dimensiones de la imagen
        height, width = imagen.shape[:2]
        center = (width // 2, height // 2)
        
        # Obtener matriz de rotación
        # cv2.getRotationMatrix2D(center, angle, scale)
        # angle: ángulo en grados (positivo = rotación antihoraria)
        # scale: factor de escala (1 = sin cambio de escala)
        matriz_rotacion = cv2.getRotationMatrix2D(center, angulo_rotacion, 1.0)
        
        # Aplicar rotación usando warpAffine
        imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (width, height))
        
        # Definir directorio de salida
        directorio_salida = os.path.join(os.path.dirname(__file__), "..", "imagenes creadas")
        
        # Crear directorio si no existe
        if not os.path.exists(directorio_salida):
            os.makedirs(directorio_salida)
        
        # Definir nombre del archivo
        nombre_archivo = f"imagen_{num_imagen}_rotacion_{angulo_rotacion:.1f}deg.jpg"
        ruta_completa = os.path.join(directorio_salida, nombre_archivo)
        
        # Guardar imagen rotada
        exito_guardado = cv2.imwrite(ruta_completa, imagen_rotada)
        
        if exito_guardado:
            return {
                'exito': True,
                'ruta_guardada': ruta_completa,
                'mensaje': f"Imagen rotada guardada correctamente en: {ruta_completa}",
                'angulo_aplicado': float(angulo_rotacion)
            }
        else:
            return {
                'exito': False,
                'ruta_guardada': None,
                'mensaje': f"Error: No se pudo guardar la imagen en {ruta_completa}",
                'angulo_aplicado': float(angulo_rotacion)
            }
    
    except Exception as e:
        return {
            'exito': False,
            'ruta_guardada': None,
            'mensaje': f"Error durante el proceso: {str(e)}",
            'angulo_aplicado': float(angulo_rotacion)
        }


if __name__ == "__main__":
    # Prueba: rotar imagen 1 un ángulo de 45 grados
    print("Prueba: Rotación de imagen")
    print("=" * 60)
    
    resultado = rotar_y_guardar_imagen(1, 45.0)
    
    print(f"Éxito: {resultado['exito']}")
    print(f"Mensaje: {resultado['mensaje']}")
    if resultado['exito']:
        print(f"Ruta guardada: {resultado['ruta_guardada']}")
    print(f"Ángulo aplicado: {resultado['angulo_aplicado']}")
