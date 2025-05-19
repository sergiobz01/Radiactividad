#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 2025
@author: sergiobarrioszamora
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import norm

def leer_spectro(ruta, skiprows=2, delimiter='\t'):
    datos = np.loadtxt(ruta, skiprows=skiprows, delimiter=delimiter)
    energia = datos[:,0]
    cuentas  = datos[:,1]
    errores  = np.sqrt(cuentas)
    return energia, cuentas, errores

def gauss(x, H, A, x0, sigma):
    return H + A * norm.pdf(x, loc=x0, scale=sigma)

def ajustar_pico(energia, cuentas, idx, half_width=25):
    lo, hi = max(0, idx-half_width), min(len(energia), idx+half_width)
    xw = energia[lo:hi]
    yw = cuentas[lo:hi]

    # Estimaciones iniciales
    H0 = np.median([yw[0], yw[-1]])
    A0 = max(yw.max() - H0, 1e-3)
    x00 = energia[idx]
    sigma0 = (energia[1] - energia[0]) * half_width / 2.355  # aproximación FWHM→σ

    p0 = [H0, A0, x00, sigma0]
    bounds = ([0, 0, x00-10, sigma0/10],
              [np.inf, np.inf, x00+10, sigma0*10])

    try:
        popt, _ = curve_fit(gauss, xw, yw, p0=p0, bounds=bounds, max_nfev=5000)
    except RuntimeError:
        popt = p0  # fallback si no converge

    return popt  # [H, A, x0, sigma]

def procesar_espectro(path, **find_peaks_kwargs):
    e, c, err = leer_spectro(path)
    peaks, _ = find_peaks(c, **find_peaks_kwargs)

    centros = []
    for idx in peaks:
        H, A, x0, sigma = ajustar_pico(e, c, idx)
        centros.append(x0)
    return e, c, err, np.array(centros)

def emparejar_picos(pos1, pos2, max_diff=20):
    #Empareja índices de dos arrays de posiciones según mínima distancia y umbral
    emparejados = []
    diffs = []
    for x1 in pos1:
        idx = np.argmin(np.abs(pos2 - x1))
        x2 = pos2[idx]
        if abs(x2 - x1) <= max_diff:
            emparejados.append((x1, x2))
            diffs.append(x2 - x1)
    return emparejados, np.array(diffs)

if __name__ == '__main__':
    # Rutas
    ruta = ""
    f1 = ruta + 'Na22.asc'
    f2 = ruta + 'Na22_NV.asc'

    # Parámetros de detección de picos
    pk_kwargs = dict(prominence=70, distance=30)

    # Procesamos ambos espectros
    e1, c1, err1, cent1 = procesar_espectro(f1, **pk_kwargs)
    e2, c2, err2, cent2 = procesar_espectro(f2, **pk_kwargs)
    
    emparejados, desplazamientos = emparejar_picos(cent1, cent2, max_diff=20)

    # Gráfica conjunta
    plt.figure(figsize=(8,5))
    # Espectro 1
    plt.errorbar(e1, c1, yerr=err1, fmt='o', color='blue',
                 label='Espectro 1', markersize=1, capsize=1)
    plt.plot(cent1, np.interp(cent1, e1, c1), 'x', color='blue', 
             label='Centroides 1', markersize = 8)

    # Espectro 2
    plt.errorbar(e2, c2, yerr=err2, fmt='o', color='red',
                 label='Espectro 2', markersize=1, capsize=1)
    plt.plot(cent2, np.interp(cent2, e2, c2), 'x', color='red', 
             label='Centroides 2', markersize = 8)

    plt.xlabel('Canal (energía)', fontsize = 14)
    plt.ylabel('Cuentas', fontsize = 14)
    plt.title('Comparación de dos espectros y sus centroides', fontsize = 16)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Impresión de desplazamientos emparejados
    print("Picos emparejados y desplazamientos (2 - 1):")
    for (x1, x2), d in zip(emparejados, desplazamientos):
        print(f"  {x1:.2f} → {x2:.2f} ; Δx = {d:.3f}")


