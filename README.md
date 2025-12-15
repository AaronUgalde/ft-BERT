# Fine-Tuning de BERT para Análisis de Sentimiento (SST-2)

Este proyecto implementa un proceso de **fine-tuning** del modelo BERT pre-entrenado para la tarea de clasificación de sentimientos utilizando el dataset **Stanford Sentiment Treebank (SST-2)**. El objetivo es adaptar un modelo de lenguaje general a una tarea específica mediante transfer learning, logrando alta precisión en la distinción entre oraciones positivas y negativas.

## Características principales

- **Preprocesamiento completo**: Tokenización y preparación de datos para el modelo BERT.
- **Fine-tuning eficiente**: Utiliza la biblioteca Hugging Face Transformers para el entrenamiento optimizado.
- **Evaluación integral**: Métricas de evaluación (exactitud) durante y después del entrenamiento.
- **Inferencia en tiempo real**: Función para probar el modelo con nuevas oraciones.
- **Configuración flexible**: Hiperparámetros centralizados para fácil experimentación.

## Requisitos

### Dependencias principales
- Python 3.8+
- PyTorch 1.12+
- Transformers 4.30+
- Datasets 2.13+
- Evaluate 0.4+
- Accelerate 0.20+

### Instalación

```bash
# Instalar dependencias desde requirements.txt
pip install -r requirements.txt

# O instalar directamente
pip install transformers datasets evaluate accelerate
```

## Estructura del proyecto

```
bert-sst2-fine-tuning/
├── README.md
├── requirements.txt
├── bert_finetuning_sst2.ipynb          # Notebook principal
├── bert-sst2-demo/                     # Directorio de salida (generado)
│   ├── saved/                          # Modelo guardado
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer.json
│   └── checkpoint-*/                   # Checkpoints (si se guardan)
└── utils/                              # Utilidades adicionales (opcional)
```

## Uso

### 1. Ejecución del notebook

Abre el notebook Jupyter `bert_finetuning_sst2.ipynb` y ejecuta las celdas en orden:

```python
# En Jupyter o Google Colab
jupyter notebook bert_finetuning_sst2.ipynb
```

### 2. Configuración inicial

El notebook incluye variables configurables al inicio:

```python
MODEL_NAME = "bert-base-uncased"   # También puede ser "distilbert-base-uncased"
TASK = "sst2"                      # Dataset de GLUE
NUM_LABELS = 2                     # Clasificación binaria
MAX_LENGTH = 128                   # Longitud máxima de tokens
USE_SMALL_SUBSET = False           # True para debugging rápido
```

### 3. Entrenamiento

El entrenamiento se realiza mediante el `Trainer` de Hugging Face, con configuración optimizada:

```python
training_args = TrainingArguments(
    output_dir="./bert-sst2-demo",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,            # Aumentar para mejores resultados
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,                     # Aceleración con GPU
)
```

### 4. Evaluación

El modelo se evalúa automáticamente durante el entrenamiento y al finalizar:

```
Resultados de evaluación: {'eval_loss': 0.215, 'eval_accuracy': 0.925, 'epoch': 3.0}
```

### 5. Inferencia

Prueba el modelo con nuevas oraciones:

```python
examples = ["This movie was fantastic!", "The product quality is poor."]
predictions, probabilities = predict(examples)
```

Salida esperada:
```
Texto: This movie was fantastic!
Sentimiento: POSITIVO (Confianza: 98.5%)
----------------------------------------
Texto: The product quality is poor.
Sentimiento: NEGATIVO (Confianza: 94.2%)
```

## Metodología

### Flujo del proceso

1. **Carga de datos**: SST-2 desde el hub de Hugging Face
2. **Tokenización**: Convertir texto a tokens BERT con truncamiento/padding
3. **Modelado**: BERT-base con capa de clasificación adicional
4. **Entrenamiento**: Fine-tuning con optimizador AdamW
5. **Evaluación**: Métricas de exactitud en conjunto de validación
6. **Guardado**: Modelo y tokenizer para uso futuro
7. **Inferencia**: Pruebas con ejemplos personalizados

### Hiperparámetros recomendados

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Learning rate | 2e-5 | Estándar para fine-tuning BERT |
| Batch size | 8-32 | Depende de la memoria GPU |
| Epochs | 3-5 | Suficiente para convergencia |
| Max length | 128 | Balance entre rendimiento y velocidad |

## Resultados

Con la configuración por defecto (BERT-base, 3 épocas), se espera alcanzar:

- **Exactitud en validación**: 92-93%
- **Pérdida de validación**: 0.20-0.25

Para mejores resultados:
- Usar `bert-large-uncased` (mayor capacidad)
- Aumentar a 5 épocas
- Desactivar `USE_SMALL_SUBSET` (usar dataset completo)
- Ajustar hiperparámetros con búsqueda en cuadrícula

## Personalización

### Usar otro modelo

```python
MODEL_NAME = "distilbert-base-uncased"  # Más rápido, menos memoria
# O
MODEL_NAME = "roberta-base"             # Arquitectura diferente
```

### Cambiar dataset

```python
TASK = "imdb"  # Dataset de reseñas de películas
# Requerirá ajustar las columnas de texto y etiquetas
```

### Añadir más métricas

```python
# En la función compute_metrics
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
```

## Recursos adicionales

- [Documentación de Transformers](https://huggingface.co/docs/transformers)
- [Paper original de BERT](https://arxiv.org/abs/1810.04805)
- [Dataset SST-2 en Hugging Face](https://huggingface.co/datasets/glue/viewer/sst2)
- [Guía de fine-tuning](https://huggingface.co/docs/transformers/training)

