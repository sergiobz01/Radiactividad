#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 2025

@author: sergiobarrioszamora

Script corregido para leer un fichero de dos columnas (x, y), realizar una regresión
lineal por mínimos cuadrados ortogonales (ODR) y mostrar resultados y gráfico.
Se corrige el manejo de errores en x (ex) para evitar divisiones por cero que generan NaN.
"""

import numpy as np
from scipy.odr import ODR, Model, RealData
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# =======================
# Parámetros editables
# =======================
ruta_fichero = ""
titulo_grafico = "Regresión lineal para la resolución"
etiqueta_x = "ln (E$_{\gamma}$)"
etiqueta_y = "ln (FWMHE/E$_{\gamma}$)"
# =======================

def read_data_2cols(filename):
    data = np.loadtxt(filename, delimiter=',')
    x, y = data.T
    return x, y

def linear_model(B, x):
    return B[0] * x + B[1]

def main():
    try:
        x, y = read_data_2cols(ruta_fichero)
    except Exception as e:
        print(f"Error al leer el fichero '{ruta_fichero}': {e}")
        return

    # Asumimos incertidumbre relativa en y
    ey = 0.01 * np.abs(y)
    # Si algun ey == 0, asignamos mínimo
    min_err_y = np.max(ey) * 1e-2
    ey[ey == 0] = min_err_y
    sy = ey

    # Configurar ODR con 2 parámetros [pendiente, ordenada]
    model = Model(linear_model)
    data  = RealData(x, y, sy=sy)
    odr   = ODR(data, model, beta0=[1.0, 0.0])
    out   = odr.run()
    m, b       = out.beta
    m_err, b_err = out.sd_beta
    chi2_red   = out.res_var


    r, p_val = pearsonr(x, y)
    r2 = r**2

    # Resultados
    print(f"Pendiente (m):    {m:.6f} ± {m_err:.6f}")
    print(f"Intercepto (b):   {b:.6f} ± {b_err:.6f}")
    print(f"χ² reducido:      {chi2_red:.6f}")
    print(f"Pearson: r = {r:.6f}, p = {p_val:.3g}")
    print(f"Determinación: R² = {r2:.6f}\n")

    # Gráfico de datos y ajuste
    plt.errorbar(x, y, yerr=sy, fmt='o', label='Datos', color = 'red', markersize = 3)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = m * x_fit + b
    plt.plot(x_fit, y_fit,
             label=f"Ajuste: y = ({m:.3f}±{m_err:.3f})·x + ({b:.3f}±{b_err:.3f})\nR² = {r2:.4f}", color = 'blue')
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.title(titulo_grafico)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()