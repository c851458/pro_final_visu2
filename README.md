# pro_final_visu2
## Warehouse Rush 3D

Un robot debe recoger un paquete de una estantería y llevarlo a la zona de entregas.
La visualización 3D usa three.js en el navegador; Python ejecuta el entorno y el Q-learning.

```bash
source .venv/bin/activate

./train3d.py                         # entrena con Q-learning y anima 1 de cada 1000 episodios
./train3d.py --demo-only --demo-episodes 5   # solo demostración con la tabla Q guardada
./play3d.py                          # control manual con el teclado
```

Cada script abre el navegador automáticamente (`--no-browser` para no hacerlo) en
`http://localhost:8765` (automático) o `http://localhost:8766` (manual).
Si el puerto está ocupado, se usa automáticamente el siguiente libre (la terminal muestra la dirección).
La página necesita internet para cargar three.js desde el CDN.

- Cámara: arrastrar para girar, rueda para zoom, clic derecho para desplazar; teclas 1-4 cambian de vista.
- Manual: flechas/WASD mover (relativo a la cámara), R recoger, E entregar, N nuevo episodio.

Las versiones anteriores siguen disponibles: `train.py` (terminal o pygame) y `play.py` (pygame).
