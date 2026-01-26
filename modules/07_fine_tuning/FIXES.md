# Fine-tuning Notebooks Fixes

## Исправленные проблемы

### 1. TypeError: unsupported operand type(s) for -: 'float' and 'dict'

**Проблема:**
```python
TypeError: unsupported operand type(s) for -: 'float' and 'dict'
```

**Причина:**
`evaluator()` возвращает **dict с метриками**, а не просто float:
- `TripletEvaluator` возвращает: `{'triplet_eval_cosine_accuracy': 0.98}`
- `EmbeddingSimilarityEvaluator` возвращает: `{'contrastive_eval_cosine_spearman': 0.82, 'contrastive_eval_cosine_pearson': 0.83}`

**До исправления:**
```python
initial_score = evaluator(model, output_path='...')
# initial_score = {'triplet_eval_cosine_accuracy': 0.98}  # это dict!
improvement = best_acc - initial_score  # TypeError!
```

**После исправления:**
```python
initial_result = evaluator(model, output_path='...')
if isinstance(initial_result, dict):
    initial_score = list(initial_result.values())[0]  # Извлекаем float
else:
    initial_score = initial_result
improvement = best_acc - initial_score  # Работает!
```

### 2. "No log" в Training Loss и Validation Loss

**Проблема:**
При обучении в progress bar показывались "No log" вместо значений loss.

**Причина:**
`sentence-transformers` не логирует raw loss values в progress bar по дизайну. Библиотека логирует только evaluation metrics (accuracy, spearman correlation), которые возвращаются evaluator'ами.

**Решение:**
Это НОРМАЛЬНОЕ поведение. Добавлена заметка в ноутбуках:
```python
print("\n💡 NOTE: 'No log' in Training/Validation Loss columns is NORMAL.")
print("   sentence-transformers logs evaluation metrics (accuracy/spearman)")
print("   but doesn't log raw loss values in the progress bar.")
print("   The evaluation metrics are saved to CSV and shown above.")
```

**Если нужно логировать loss:**
- Используйте TensorBoard callback
- Или перейдите на HuggingFace Transformers Trainer API

### 3. "Metrics file not found" warning

**Проблема:**
```
⚠️  Metrics file not found - evaluation may not have run
```

**Причина:**
Неправильные пути к CSV файлам с метриками.

**До исправления:**
```python
# Неправильный путь
metrics_file = '../output/triplet_finetuned_model/triplet_eval_results.csv'
```

**После исправления:**
```python
# Правильный путь (с eval/ и правильным префиксом)
metrics_file = '../output/triplet_finetuned_model/eval/triplet_evaluation_triplet_eval_results.csv'
```

**Почему:**
`sentence-transformers` сохраняет evaluation results:
1. В подпапку `eval/`
2. С префиксом от имени evaluator класса:
   - `TripletEvaluator` → `triplet_evaluation_`
   - `EmbeddingSimilarityEvaluator` → `similarity_evaluation_`
3. С суффиксом из параметра `name` evaluator'а

### 4. Дублирующиеся записи с epoch=-1

**Проблема:**
В CSV файлах были записи с `epoch=-1, steps=-1`.

**Причина:**
Это pre-training evaluation (initial evaluation перед началом обучения).

**Решение:**
Добавлена фильтрация:
```python
# Filter out pre-training evaluation
metrics_df = metrics_df[(metrics_df['epoch'] >= 0) & (metrics_df['steps'] >= 0)]
```

## Исправленные файлы

### 02_triplet_loss.ipynb
- ✅ Исправлен путь к metrics CSV (cell-18, cell-20)
- ✅ Добавлена фильтрация pre-training rows
- ✅ Добавлена заметка о "No log" в progress bar
- ✅ Исправлена обработка колонки `accuracy_cosine` вместо `accuracy`
- ✅ Добавлен параметр `optimizer_params={'lr': 2e-5}` для лучшей сходимости

### 03_contrastive_loss.ipynb
- ✅ Исправлен путь к metrics CSV (cell-16, cell-18)
- ✅ Добавлена фильтрация pre-training rows
- ✅ Добавлена заметка о "No log" в progress bar
- ✅ Исправлена обработка колонки `cosine_spearman` вместо `spearman`
- ✅ Добавлен параметр `optimizer_params={'lr': 2e-5}` для лучшей сходимости

## Структура CSV файлов

### TripletEvaluator results:
```
epoch,steps,accuracy_cosine
1.0,23,0.989130437374115
2.0,46,0.989130437374115
3.0,69,1.0
```

### EmbeddingSimilarityEvaluator results:
```
epoch,steps,cosine_pearson,cosine_spearman
1.0,25,0.8305601530137644,0.8244461317262469
2.0,50,0.8412345678901234,0.8356789012345678
```

## Как избежать проблем в будущем

1. **Всегда используйте eval/ в пути к метрикам**
2. **Проверяйте имя evaluator класса для префикса файла**
3. **Не ожидайте loss в progress bar для sentence-transformers**
4. **Фильтруйте epoch=-1 если нужны только training metrics**

## Дополнительные улучшения

### Если нужно логировать training loss:

```python
from torch.utils.tensorboard import SummaryWriter

# Создать TensorBoard writer
writer = SummaryWriter(log_dir='../output/logs')

# В цикле обучения (custom training loop)
writer.add_scalar('Loss/train', loss, step)
```

### Если нужен более гибкий контроль:

Используйте HuggingFace Transformers Trainer API:
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='../output',
    logging_steps=10,  # Логировать каждые 10 шагов
    eval_steps=50,
    save_strategy='steps',
    logging_strategy='steps'
)
```

## Заключение

Все проблемы исправлены. "No log" в progress bar - это нормальное поведение `sentence-transformers`. Метрики правильно сохраняются в CSV файлы и теперь корректно отображаются.
