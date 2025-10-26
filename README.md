## **PRD**

Problem statement: В RAG исходный пользовательский запрос часто плохо покрывает релевантные документы (терминология,
уровень детализации, синонимия), из-за чего падает recall и качество ответа.

Цель фичи: автоматически сгенерировать 3 оптимизированные варианта запроса (multi-query), которые при поиске в векторном
хранилище увеличивают recall@k на ≈+20%, не ухудшая precision@k>5%. Фича должна работать как онлайн-компонент перед step'ом
retrieval и возвращать варианты с их scores.

Ограничения: end-to-end p95 < 1s, cost < 5$/M input tokens

## **User stories**

- Как конечный пользователь ассистента, я хочу, чтобы ассистент автоматически расширил мой запрос и пробовал
  альтернативные формулировки, чтобы найти релевантные документы, даже если я использую узкую/неформальную лексику.
- Как MLE, я хочу простую CLI/REST-интеграцию, чтобы тестировать multi-query на наших датасетах и метриках.
- Как PM, я хочу видеть объяснение реализации и метрики качества, чтобы мониторить и A/B тестировать

## **Text-based wireframes**

### **Step 1: User Input**

- Prompt: "Enter your question:"
- Input: free-text string from user
- Example: "Key risks in climate reports?"
- Validation: non-empty string

### **Step 2: Query Generation (LLM API call)**

- Input: user question
- Action: call LLM API
- Output: 3 candidate queries
- Handle Errors/Timeouts/Retries
- Validate output format

### **Step 3: Embedding & Similarity Scoring**

- Input: 3 candidate queries
- Action:
    1. Generate embeddings using sentence-transformers
    2. Compute similarity with input query **as proxy score** (recall@k on test data in future)
    3. Assign a score to each query (0-1)
- Output: ```List[Tuple[str, float]]```

### **Step 4: Output Display**
  - Format results for CLI
  - Display top 3 queries + scores

### **Step 5: Logging / Analytics**
  - Log:
      - User question
      - Generated query variants with scores
      - Timestamp

## **Sub tasks**
### High-Priority
- CLI User Input Handling + LLM API
- Prompt Engineering
- Output Validation & Error Handling

### Middle-Priority
- Embeddings Generation
- Scoring
- CLI Output Display

### Low-Priority
- Logging / Analytics
- Mock Retrieval / Recall Testing



1. Выбор модели
   ```mistralai/mistral-7b-instruct``` — лёгкая и экономичная модель, отлично подходит для прототипирования и быстрых
   экспериментов.
   Подходит, если нужно:

- быстро проверять гипотезы без больших затрат
- запускать batch-тесты (например, 1000+ запросов)
- не требуется сложное reasoning.
- Плюсы: Низкая цена, хорошее следование простым инструкциям.
- Минусы: Иногда генерирует менее разнообразные варианты, чем флагманские модели.

```Grok``` (из ТЗ) — средне-тяжёлая модель, с хорошими возможностями reasoning и огромным контекстом, но чуть дороже по
стоимости и латентности.
Подходит, если:
- нужно использовать большой контекст (например, длинные документы),
- важно получать “умные” переформулировки (а не просто синонимы),
- нет строгих ограничений по бюджету.
- Плюсы: Сильна в генерации контекстно насыщенных запросов.
- Минусы: Иногда менее стабильна в структурированной генерации (например, при требовании строго 3 JSON-объектов).

Будем использовать StructedOutput

Спустя время эксплуатации latency упадет в силу кеширования openrouter

## Описание решения
> ❗Перед использованием обязательно создайте ```.env``` файл в корневой директории проекта с OpenRouter ```API_KEY = ...```❗
- ```app.py``` - FastAPI реализация генерации Multi Query, для каждого из вариантов считается ```score``` (Косинусное расстояние по отношению к основному запросу)
- ```cli.py``` - Полная CLI реализация, вплоть до mock-retrieval
- ```docs.py``` - Ingest + Indexing документов взятых из [IPCC](https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_Chapter16.pdf)
- ```models.py``` - Pydantic модели
- ```utils.py``` - Обращение к OpenRouter API, подсчет схожести
- ```requirements.txt``` - Зависимости

### Пример CLI использования
```User query: Key risks in climate reports?```

```Default query results (10): [...]```

```Optimized results(18): [...]```

Как правило, все выданные варианты релевантны, соответственно прирост recall@k:

**Recall improvement** = (18 − 10) / 10 = 0.8 → **+80%**

