#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:11:25 2025

@author: sergiobarrioszamora
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from statistics import stdev
from statistics import mean
from scipy.signal import find_peaks
from scipy.signal import peak_widths

ruta = ""


datos = np.loadtxt(ruta + '', skiprows=2, delimiter ='\t')
k_calibracion = 2.632 #Calculada previamente, cambiar a conveniencia.

energía   = datos[:, 0] #Primera columna del fichero de datos 
canales = datos[:, 1] #Segunda columna del fichero de datos

errores = np.sqrt(canales)

energía = (energía * k_calibracion)
# energíaNV = (energía * k_calibracion) + 0.302





def gauss(x, H, A, x0, sigma): #Función que define la gaussiana usando la librería de SciPy
    """
    H     : nivel de fondo (baseline)
    A     : amplitud del pico
    x0    : centro (media) de la gaussiana
    sigma : desviación estándar
    """
    # norm.pdf está normalizado a área = 1, así que escalamos por A*sqrt(2*pi)*sigma
    return H + A * norm.pdf(x, loc=x0, scale=sigma)

prominence_min = 25
distance_min = 30 # No tocar este parámetro, es una distancia pequeña pero que evita el doble conteo de un mismo pico.
#height_min = 50
peaks, properties   = find_peaks(canales,
                             prominence=prominence_min,
                             distance = distance_min)
                             #,height=height_min)  Por si acaso queremos utilizar el parámetro de altura

fit_curves = []    
num_picos = peaks.size # Variable para el conteo de picos

plt.figure(figsize=(8,5))
plt.errorbar(energía, canales, yerr=errores, fmt='.', color='red', label='Espectro medido', markersize = 2, capsize = 1, zorder = 1)
plt.plot(energía[peaks], canales[peaks], 'x', label='Picos', color='blue', zorder = 2, markersize = 10)
for x_fit, y_fit in fit_curves:
    plt.plot(x_fit, y_fit, '--', label='Ajuste gaussiano')
plt.xlabel('Energía (keV)', fontsize=14)
plt.ylabel('Cuentas', fontsize=14)
plt.title('Espectro del isótopo $^{}$ELEMENTO con picos detectados', fontsize=16)
plt.legend()
plt.tight_layout()
plt.show()

#Hasta aquí, se ha leído el archivo de datos y se han identificado computacionalmente los picos.

for i, idx in enumerate(peaks, start=1):
    half_width = 6 #Número de puntos antes y después del pico, ajustar como mejor convenga
    lo, hi = max(0, idx-half_width), min(len(energía), idx+half_width)
    x_win = energía[lo:hi]
    y_win = canales[lo:hi]
    
    #Se ajustan los parámetros de la gaussiana
    H0 = np.median([y_win[0], y_win[-1]])
    A0 = max(canales[idx] - H0, 0)    # amplitud aproximada, y siempre mayor que 0.1, si no, da problemas
    sigma0 = peak_widths(canales, [idx], rel_height=0.5)[0][0]/(2*np.sqrt(2*np.log(2)))  # FWHM→σ y evitamos que sea menor que 0.1
    x00 = energía[idx]
    p0 = [H0, A0, x00, sigma0]
    
    lower = [0,      0,    x00-10, sigma0/10]
    upper = [np.inf, np.inf, x00+10, sigma0*10]
    popt, properties = curve_fit(
            gauss, x_win, y_win, p0=p0,
            bounds=(lower, upper), method='trf', max_nfev=10000)
    
    # Cálculo de FWHM = 2.35 * sigma ajustada
    sigma_fit = popt[3]
    fwhm = 2.35 * sigma_fit
    print(f"Pico {i}: centro = {popt[2]:.3f} keV, sigma = {sigma_fit:.3f} keV, FWHM = {fwhm:.3f} keV")
        
    plt.figure(figsize=(8,5))
    errores_win = np.sqrt(y_win)
    plt.errorbar(x_win, y_win, yerr=errores_win, fmt='.', color='red', label='Datos espectro', capsize = 3)
    plt.plot(x_win, gauss(x_win, *popt), 'r-', label='Ajuste gaussiano', color = 'blue')
    plt.title(f'Pico {i}: Centrado en {popt[2]:.3f} keV', fontsize=16) 
    plt.xlabel('Energía (keV)', fontsize=14)
    plt.ylabel('Cuentas', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
