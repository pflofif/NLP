# Lab 7 — Linear SVM + char-ngrams + imbalance

## 1. Підзадача класифікації
Binary relevance classification: для кожної пари (запит, документ) визначити, чи документ є релевантним до запиту.  
Корпус: відгуки про заклади харчування м. Львів (4 640 документів, 40 запитів, 43 релевантні пари).

## 2. Baseline із ЛР6
Взято winning baseline **B2** з ЛР6: TF-IDF word(1,2) + LogisticRegression, `class_weight="balanced"`, ознаки `noun_adj_text`.  
Val macro-F1 ≈ 0.9255 | Test macro-F1 = 1.0000 (малий тест-сет, 3 позитивних пари).

## 3. SVM-варіанти перевірені
| Variant | Features | class_weight | Val F1 | Test F1 |
|---------|----------|--------------|--------|---------|
| V2 | TF-IDF word(1,2), LinearSVC | None | 0.9255 | 1.0000 |
| V2b | TF-IDF word(1,2), LinearSVC | balanced | 0.9255 | 1.0000 |
| V3a | TF-IDF char_wb(3,5), LinearSVC | balanced | 0.9255 | 0.6137 |
| V3b | TF-IDF word(1,2) + char_wb(3,5), LinearSVC | balanced | 0.9255 | 0.7566 |

## 4. Дисбаланс класів
**Так, є суттєвий дисбаланс:** train 19.4:1, val 20:1, test 26.7:1.  
`class_weight="balanced"` не дав числового виграшу F1 на val (всі word-моделі = 0.9255), але char-only модель без балансування виявилась нестабільною (test F1=0.6137).

## 5. Обраний поріг
- **CHOSEN_THR = −0.7675** (підібраний за max F1 на validation)  
- При default порозі 0.0: Recall=0.75 (один позитив втрачається), F1=0.857  
- При порозі −0.7675: P=1.0, R=1.0, F1=1.0 — всі позитиви знайдені  
- **Логіка: recall-first** — FN (пропущений релевантний документ) дорожчий за FP у IR  
- Поріг підбирався виключно на val; test використовувався тільки для фінальної оцінки

## 6. Найкраща модель
**V2: LinearSVC word(1,2), no class_weight** — Val F1=0.9255, Test Acc=1.0000, Test F1=1.0000.  
Збігається з LogReg (V1) та SVM balanced (V2b) — всі три word-моделі однаково сильні на цих даних.  
Char-ngrams (V3a, V3b) дали гірший результат на test, що свідчить про чистоту тексту корпусу.

## 7. Що робити далі
- Hard negative mining: замінити random-негативи на тематично близькі для сильнішого навчання
- Збільшити анотований датасет (зараз 40 запитів — замало для надійних висновків)
- Спробувати BM25 як пошуковий baseline для порівняння з TF-IDF + класифікатором
- Розглянути dense retrieval (sentence-transformers) як наступний крок після класичних baseline
