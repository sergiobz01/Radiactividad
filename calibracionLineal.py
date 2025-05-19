#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:08:12 2025

@author: sergiobarrioszamora
"""

#!/usr/bin/env python3
import numpy as np
from scipy.odr import ODR, Model, RealData
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

enabled = True
ruta = ""

def read_data(filename):
    """
    Lee los datos de un fichero con formato:
    x, y, error_x, error_y
    separador: coma y espacio (",").
    Devuelve cuatro arrays: x, y, ex, ey.
    """
    data = np.loadtxt(filename, delimiter=',')
    x, y, ex, ey = data.T
    return x, y, ex, ey

# Modelo lineal: y = m*x + b
def linear_model(B, x):
    return B[0] * x 


def main():
    # Usa la ruta definida en la variable 'ruta'
    filename = ruta
    try:
        x, y, ex, ey = read_data(filename)
    except Exception as e:
        print(f"Error al leer el fichero '{filename}': {e}")
        return

    # Configuración del modelo y datos para ODR
    model = Model(linear_model)
    data = RealData(x, y, sx=ex, sy=ey)
    odr = ODR(data, model, beta0=[1.0, 0.0])  # beta0: estimación inicial [pendiente, intercepto]
    output = odr.run()
    m, b = output.beta
    m_err, b_err = output.sd_beta
    chi2_red = output.res_var
    
    # Cálculo del coeficiente de correlación de Pearson y R²
    r, p_val = pearsonr(x, y)
    r2 = r**2


    m, b = output.beta
    m_err, b_err = output.sd_beta

    # Resultados
    print(f"Pendiente (m): {m:.6f} ± {m_err:.6f}")
    print(f"Intercepto (b): {b:.6f} ± {b_err:.6f}")
    print(f"Chi-cuadrado reducido: {chi2_red:.6f}")
    print(f"Coeficiente de correlación de Pearson: r = {r:.6f}, p-valor = {p_val:.3g}")
    print(f"Coeficiente de determinación (R²): {r2:.6f}\n")

    # Interpretación
    print("Interpretación del ajuste:")
    if abs(chi2_red - 1) < 0.2:
        print(" - El chi-cuadrado reducido está cercano a 1, indicando que los errores estimados son consistentes y el ajuste es bueno.")
    elif chi2_red > 1:
        print(" - Chi-cuadrado reducido > 1 sugiere que la dispersión de los datos es mayor de lo esperado, o que los errores en y podrían estar infraestimados.")
    else:
        print(" - Chi-cuadrado reducido < 1 indica que los errores podrían estar sobreestimados o que el modelo se ajusta más de lo esperado.")
    
    if r2 > 0.9:
        print(" - R² alto (>0.9) confirma una fuerte correlación lineal entre x e y.")
    elif r2 > 0.5:
        print(" - R² moderado (0.5-0.9): existe correlación lineal, pero con cierta dispersión.")
    else:
        print(" - R² bajo (<0.5) sugiere una débil relación lineal, quizá convenga revisar el modelo o los datos.")
    
    # Gráfica de datos y ajuste
    plt.errorbar(x, y, yerr=ey, fmt='o', label='Datos', markersize = 2, color = 'red')
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = m * x_fit + b
    label_fit = (
        f'Ajuste: E = ({m:.6f}±{m_err:.3f})x, ' \
        f'R² = {r2:.5f}'
    )
    plt.plot(x_fit, y_fit, '-', label=label_fit, color = 'blue')
    plt.xlabel('Canal', fontsize = 10)
    plt.ylabel('Energía (keV)', fontsize = 10)
    plt.title('Relación entre canales de detección y energías tabuladas', fontsize = 12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    