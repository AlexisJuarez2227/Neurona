import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button, Text, END
# from sklearn.preprocessing import MinMaxScaler  # Comentado para desactivar la normalización
import os
import cv2
from natsort import natsorted

class VideoGenerator:
    def __init__(self, image_folder, output_video):
        self.image_folder = image_folder
        self.output_video = output_video

    def generate_video(self):
        image_files = [f'{self.image_folder}/{img}' for img in os.listdir(self.image_folder) if img.endswith('.png')]
        image_files = natsorted(image_files)  # Usar natsorted para ordenar correctamente los archivos

        if not image_files:
            print("No se encontraron imágenes para crear el video.")
            return

        frame = cv2.imread(image_files[0])
        height, width, layers = frame.shape

        # Mantén el códec 'mp4v' y aumenta la tasa de fotogramas a 24 fps
        video = cv2.VideoWriter(self.output_video, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))

        for image in image_files:
            video.write(cv2.imread(image))

        video.release()
        print("Video generado exitosamente.")

class PerceptronRegresion:
    # Asigna los argumentos a las variables de la instancia y inicializa los historiales de pesos y predicciones
    def __init__(self, tasa_aprendizaje, numero_epocas):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.numero_epocas = numero_epocas
        self.pesos = None
        self.historial_pesos = []
        self.historial_predicciones = []

    # Inicializa los pesos aleatoriamente y guarda el primer historial de pesos.
    def ajustar(self, entradas, salidas):
        n_muestras, n_caracteristicas = entradas.shape
        self.pesos = np.random.randn(n_caracteristicas + 1)
        self.historial_pesos.append(self.pesos.copy())
        historial_errores = []

        if not os.path.exists('resultados_epocas'):
            os.makedirs('resultados_epocas')

        for epoca in range(self.numero_epocas):
            error_total = 0
            predicciones_epoca = []

            for muestra, salida_esperada in zip(entradas, salidas):
                salida_estimacion = np.dot(muestra, self.pesos[:-1]) + self.pesos[-1]
                predicciones_epoca.append(salida_estimacion)
                error = salida_esperada - salida_estimacion
                error_total += error ** 2
                ajuste = self.tasa_aprendizaje * error
                self.pesos[:-1] += ajuste * muestra
                self.pesos[-1] += ajuste

            self.historial_pesos.append(self.pesos.copy())
            self.historial_predicciones.append(predicciones_epoca)
            historial_errores.append(error_total / n_muestras)

            PlotGenerator.plot_ydeseada_ycalculada(salidas, predicciones_epoca, epoca)

            if error_total / n_muestras < 0.01:
                break

        video_generator = VideoGenerator('resultados_epocas', 'resultados_epocas/resultados_video.mp4')
        video_generator.generate_video()

        return historial_errores

    # Calcula la predicción multiplicando las entradas por los pesos y sumando el sesgo
    def predecir(self, entradas):
        return np.dot(entradas, self.pesos[:-1]) + self.pesos[-1]

class PlotGenerator:
    @staticmethod
    # Crea y guarda un gráfico de los valores reales y predicciones.
    def plot_ydeseada_ycalculada(salidas, predicciones_epoca, epoca):
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(salidas)), salidas, 'go-', label='Valor Real')
        plt.plot(range(len(predicciones_epoca)), predicciones_epoca, 'rx--', label='Predicción')
        plt.title(f'Ydeseada & Ycalculada - Número de época {epoca}')
        plt.xlabel('Muestras')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'resultados_epocas/epoca_{epoca}.png')
        plt.close()

    @staticmethod
    # Crea y guarda un gráfico del error por época
    def plot_error_epocas(errores):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(errores) + 1), errores, color='blue')
        plt.title('Error de las épocas')
        plt.xlabel('Épocas')
        plt.ylabel('Error')
        plt.grid(True)
        plt.savefig('resultados_epocas/error_epocas.png')
        plt.close()

    @staticmethod
    # Crea y guarda un gráfico de la evolución de los pesos por época
    def plot_pesos_epocas(historial_pesos, colores):
        plt.figure(figsize=(10, 5))
        for i in range(len(historial_pesos[0])):
            plt.plot(range(len(historial_pesos)), [w[i] for w in historial_pesos], label=f'peso {i+1}', color=colores[i % len(colores)])
        plt.title('Comportamiento de los pesos')
        plt.xlabel('Épocas')
        plt.ylabel('Peso')
        plt.legend()
        plt.grid(True)
        plt.savefig('resultados_epocas/pesos_epocas.png')
        plt.close()

class AplicacionPerceptron:
    # Establece la ventana raíz, los datos y llama a la función para inicializar la interfaz
    def __init__(self, root, X, y):
        self.root = root
        self.root.title("Perceptrón")
        self.X = X
        self.y = y

        self.inicializar_interfaz()

    # Crea etiquetas, campos de entrada, botones y un cuadro de texto para mostrar los resultados
    def inicializar_interfaz(self):
        Label(self.root, text="Curva de aprendizaje:").grid(row=0, column=0)
        self.entrada_tasa_aprendizaje = Entry(self.root, validate="key", validatecommand=(self.root.register(self.validar_tasa_aprendizaje), '%P'))
        self.entrada_tasa_aprendizaje.grid(row=0, column=1)

        Label(self.root, text="Número de épocas:").grid(row=1, column=0)
        self.entrada_epocas = Entry(self.root, validate="key", validatecommand=(self.root.register(self.validar_epocas), '%P'))
        self.entrada_epocas.grid(row=1, column=1)

        Button(self.root, text="Iniciar Entrenamiento", command=self.iniciar_entrenamiento).grid(row=2, column=0, columnspan=2)

        Label(self.root, text="Resultados:").grid(row=3, column=0, columnspan=2)
        self.texto_resultados = Text(self.root, height=15, width=50)
        self.texto_resultados.grid(row=4, column=0, columnspan=2)

    # Intenta convertir el valor a float y verifica que esté en el rango [0, 1]
    def validar_tasa_aprendizaje(self, valor):
        try:
            val = float(valor)
            return 0 <= val <= 1
        except ValueError:
            return False

    # Verifica que el valor sea un número entero positivo
    def validar_epocas(self, valor):
        return valor.isdigit() and int(valor) > 0

    # Obtiene la tasa de aprendizaje y el número de épocas desde la interfaz
    def iniciar_entrenamiento(self):
        tasa_aprendizaje = float(self.entrada_tasa_aprendizaje.get())
        epocas = int(self.entrada_epocas.get())

        perceptron = PerceptronRegresion(tasa_aprendizaje, epocas)
        
        # escalador = MinMaxScaler()
        # X_normalizado = escalador.fit_transform(self.X)  # Normalizó los datos

        perceptron.pesos = np.random.randn(self.X.shape[1] + 1)
        errores = perceptron.ajustar(self.X, self.y)
        y_calculada_final = perceptron.historial_predicciones[-1]

        pesos_iniciales = perceptron.historial_pesos[0]
        pesos_finales = perceptron.pesos

        self.imprimir_tabla_pesos("Pesos Iniciales", pesos_iniciales)
        self.imprimir_tabla_pesos("Pesos Finales", pesos_finales)

        PlotGenerator.plot_error_epocas(errores)
        colores = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        PlotGenerator.plot_pesos_epocas(perceptron.historial_pesos, colores)

    # Inserta el título y los pesos en el cuadro de texto de resultados de la interfaz gráfica
    def imprimir_tabla_pesos(self, titulo, pesos):
        self.texto_resultados.insert(END, f"{titulo}:\n")
        tabla_pesos = ""
        for i, peso in enumerate(pesos):
            tabla_pesos += f"Peso {i+1}: {peso:.4f}\n"
        self.texto_resultados.insert(END, tabla_pesos)
        self.texto_resultados.see(END)

if __name__ == "__main__":
    filepath = "arredondo.csv"
    df = pd.read_csv(filepath, delimiter=',')  # Leer archivo CSV con delimitador ','

    # Elimina espacios adicionales en los nombres de las columnas
    df.columns = df.columns.str.strip()

    # Verifica si la columna 'y' existe en el DataFrame
    if 'y' not in df.columns:
        raise KeyError("La columna 'y' no se encontró en el archivo CSV.")

    X = df.drop(columns=['y']).values
    y = df['y'].values

    root = Tk()
    app = AplicacionPerceptron(root, X, y)
    root.mainloop()
