# Simulacro de rollback en V2-N2

Ejecutado el 2026-11-14 13:04 UTC sobre https://profinalvisu2.vercel.app con el flujo de Prefect `warehouse-rush-simulacro-rollback` (`bd53dde2-9868-4a36-8e09-ccd2ed7a95cb`). Evidencia completa: [`../ejecuciones/20261114-130457-simulacro-rollback-bd53dde2.json`](../ejecuciones/20261114-130457-simulacro-rollback-bd53dde2.json).

**Escenario:** un operador fuerza a producción un modelo que los criterios habían rechazado. Simulacro: modelo rechazado por los criterios (0.41 rescates por pedido, maximo 0.30) promovido con --forzar. La vigilancia sondea la URL pública, detecta la alerta y revierte al recomendado anterior sin intervención manual.

Sonda: 50 pedidos de la suite de evaluación de v2-n2, ejecutados por la API desplegada.

| Momento | Modelo servido | Estado | Éxito | OTIF | Rescates | Utilidad/pedido |
|---|---|---|---|---|---|---|
| 1. Antes | `v2-n2-imitacion-v2-n2-A-s0-9f293752` | ok | 100.0% | 96.0% | 0.00 | $45,893 |
| 3. Tras la promoción forzada | `v2-n2-ppo-v2-n2-A-s0-a1ae7239` | alerta | 98.0% | 90.0% | 0.50 | $40,444 |
| 5. Tras el rollback | `v2-n2-imitacion-v2-n2-A-s0-9f293752` | ok | 100.0% | 96.0% | 0.00 | $45,893 |

## Secuencia

1. **Estado inicial:** la URL sirve `v2-n2-imitacion-v2-n2-A-s0-9f293752` en v2-n2 y la sonda está en `ok` (commit `09b6a85`).
2. **Publicación:** se publica `v2-n2-ppo-v2-n2-A-s0-a1ae7239` en Hugging Face con su manifiesto.
3. **Promoción forzada:** `publicar_modelo --recomendar --forzar` → commit `4ac7230`. La URL lo sirve (commit `4ac7230`).
4. **Vigilancia** (`warehouse-rush-vigilancia`): la sonda marca `alerta`: rescates 0.50 por pedido, maximo 0.3. Rollback `v2-n2-ppo-v2-n2-A-s0-a1ae7239` → `v2-n2-imitacion-v2-n2-A-s0-9f293752` (commit `4343a74`); el modelo forzado queda como `revertido` en el catálogo y en el Model Registry.
5. **Verificación:** la URL vuelve a servir `v2-n2-imitacion-v2-n2-A-s0-9f293752` y la sonda está en `ok`.

Duración total: 5.2 min (incluye dos despliegues de Vercel).
