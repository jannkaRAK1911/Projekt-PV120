# Spojitý LTI systém chemických reakčných sietí – Varianta (a)
Tento repozitár obsahuje Python implementáciu spojitého lineárneho časovo invariantného (LTI) systému, ktorý modeluje jednoduchú chemickú reakčnú sieť.

## Reakčná sieť (Varianta a)
Systém pozostáva z troch premenných A, B a C s nasledujúcimi reakciami:
A→B
B→C
C→A
Prítok: _→A
Odtoky: A→_, B→_, C→_

Všetky reakčné koeficienty sú nastavené na 1.

## Funkcie
1. Simulácia v čase
  - Sleduje koncentrácie A,B a C v čase pre rôzne počiatočné podmienky.
  - Používa scipy.integrate.solve_ivp.
2. Analytické určenie ekvilibria
  - Vypočíta ekvilibrium systému analyticky pomocou sympy.
3. Analýza stability
  - Vypočíta Jacobiho maticu v ekvilibriu.
  - Určí vlastné čísla na analýzu stability.
4. 2D projekcie vektorového poľa
  - Vizualizuje dynamiku systému v 2D projekciách (AB, AC, BC) pomocou quiver grafov.

## Požiadavky

- `Python` 3.8+
- `numpy`
- `sympy`
- `scipy`
- `matplotlib`

## Autor
- Janka Rakytová
