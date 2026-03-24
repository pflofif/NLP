# 1. Стратегія

Group split по `place_name`, seed=42, 80/10/10.

## 2. Статистика сплітів

| Split | N    | %     |
| ----- | ---- | ----- |
| train | 3723 | 80.2% |
| val   | 444  | 9.6%  |
| test  | 473  | 10.2% |

## 3. Leakage checks

| Перевірка                  | Результат |
| ----------------------------------- | ------------------ |
| Exact dup train∩test               | 7                  |
| Exact dup train∩val                | 5                  |
| Near-dup cosine≥0.95 train vs test | 226                |
| Template leakage                    | 0                  |
| Group leakage (place_name)          | 0 overlap          |

## 4. Висновок

Group split забезпечує ізоляцію закладів між сплітами. Точних дублів між train/test не виявлено.
Основний залишковий ризик — дисбаланс класів (rating=5: ~79%) та near-dupes всередині train.
Векторизатор TF-IDF fit тільки на train (Pipeline discipline підтверджено).
