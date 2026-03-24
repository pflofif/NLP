# 1. Стратегія split

**Group split** по `place_name`, seed=42, пропорції 80/10/10.

Усі відгуки одного закладу потрапляють лише в один спліт. Це запобігає
vocabulary leakage: специфічні назви страв або локацій конкретного закладу
не потрапляють одночасно в train і test. Стратифікований split по рейтингу
не обраний, бо group leakage є більш критичним ризиком для цього завдання.
Time-based split не обраний, бо основна задача — класифікація рейтингу,
а не прогнозування дрейфу. Датаспан 2016–2026, reviews рівномірно по роках.

## 2. Статистика сплітів

| Split | N    | %     | Rating distribution                 |
| ----- | ---- | ----- | ----------------------------------- |
| train | 3723 | 80.2% | r1:11%, r2:3%, r3:3%, r4:4%, r5:79% |
| val   | 444  | 9.6%  | r1:12%, r2:3%, r3:5%, r4:5%, r5:76% |
| test  | 473  | 10.2% | r1:9%, r2:4%, r3:4%, r4:3%, r5:79%  |

Дисбаланс rating=5 (~79%) присутній у всіх сплітах рівномірно.

## 3. Leakage checks results

| Перевірка                           | Результат                 |
| -------------------------------------------- | ---------------------------------- |
| Exact duplicates train∩test                 | 7                                  |
| Exact duplicates train∩val                  | 5                                  |
| Exact duplicates val∩test                   | 2                                  |
| Near-duplicates (cosine≥0.95) train vs test | 226                                |
| Template / metadata leakage                  | 0                                  |
| Group leakage (place_name)                   | 0 train∩test, 0 train∩val        |
| Time leakage                                 | n/a (group split, не time-based) |

## 4. Ризики, що залишились

- **Дисбаланс класів**: rating=5 складає ~79% у всіх сплітах — потребує oversampling або weighted loss при навчанні
- **Near-duplicates всередині train**: ідентичні відгуки від одного автора для різних філій закладу (68 text dupes у corpus)
- **Time drift**: дані 2016–2026 в одному спліті — стиль і тематика відгуків могли змінитись
- **OOV для нових закладів**: group split означає, що val/test contains unseen places (це задумане, але знижує recall для рідкісних сутностей)
- **PII маски**: `<PHONE>/<EMAIL>/<URL>` присутні в тексті — модель може навчитись на їх наявності, а не на змісті

## 5. Що далі

- Застосувати class weighting або oversampling для rating≠5 при навчанні
- Для time-aware аналізу зробити додатковий time-based split як sensitivity check
- Перевірити якість near-duplicate removal всередині train перед навчанням ML-моделі
