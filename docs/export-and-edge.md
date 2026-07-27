# Exportación, cuantización y edge

## ONNX

El exportador usa `torch.onnx.export` con `dynamo=True`, shapes dinámicas para batch y reporte de exportación. Cuando ONNX Runtime está disponible, compara la salida exportada con PyTorch.

## INT8

La cuantización dinámica se aplica a capas lineales y registra tamaño y latencia. No se asume que siempre mejore rendimiento; debe medirse en el hardware objetivo.

## ExecuTorch

El exportador opcional genera un archivo `.pte` utilizando `torch.export` y XNNPACK. Android, iOS y hardware embebido requieren toolchains y runtime propios.

## Benchmark

Los informes deben incluir FP32, compilado, ONNX, INT8 y edge cuando estén disponibles; tamaño, P50/P95/P99, throughput, memoria, error numérico y pérdida de calidad.
