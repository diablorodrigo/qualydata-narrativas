# qualydata-narrativas

Pipeline para el **Desafío QualyData** del Hackathon Argentina Space 2026 (Salta, 11 al 13 de noviembre de 2026), track Datos y Streaming.

Lee cientos de miles de comentarios y contesta dos preguntas:

1. **¿Qué narrativas circulan?** Las descubre solo (sin lista previa), marca el ruido, dice si cada comentario apoya o rechaza su narrativa y estima cuándo nació cada una.
2. **¿Qué parte está armada?** Detecta las cuentas que actúan coordinadas y las agrupa por operación.

Todo corre en una notebook, en CPU, sin APIs pagas: **unos 40 segundos para 200.000 comentarios** y costo de IA cero.

## Cómo correrlo (un solo comando)

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set final (el viernes 13, con los datos en `datos/final/`):

```bash
python correr.py --datos datos/final --salida entrega
```

Set de práctica y puntaje con el script oficial del jurado:

```bash
python correr.py --datos datos/practica --salida salida/practica
python ../desafio-qualydata/evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica
```

La primera vez baja el modelo de embeddings (unos 220 MB) a `.cache/`. Después funciona sin internet.

Salida en la carpeta elegida:

| Archivo | Contenido |
|---|---|
| `comentarios.csv` | `comentario_id, narrativa, postura` (narrativa `ninguna` si es ruido) |
| `narrativas.csv` | `narrativa, titulo, frases_clave, fecha_origen` |
| `cuentas.csv` | `cuenta_id, grupo` (vacío si la cuenta es genuina) |
| `costo.json` | tiempo, tokens y costo de IA, hardware y almacenamiento |
| `tablero.html` | tablero para la demo: qué circula, desde cuándo y quién lo empuja |

## Resultados en el set de práctica

Puntaje automático con `evaluacion/puntuar.py` (el mismo script del jurado):

| Componente | Línea base del repo | Este pipeline |
|---|---|---|
| Agrupamiento B-cubed F1 | 0,349 | **0,999** |
| Detección de ruido F1 | 0,665 | **0,988** |
| Postura macro-F1 | 0,491 | **0,999** |
| Origen de narrativas emergentes | 0,333 | **1,000** (las 3 a menos de 3 h) |
| **Narrativas (30)** | 12,68 | **29,92** |
| Cuentas coordinadas F1 | 0,116 | **0,955** (P 0,98, R 0,93) |
| Grupos por operación B-cubed F1 | 0,278 | **0,937** |
| **Coordinación (25)** | 4,12 | **23,75** |
| **Total automático (55)** | **16,8** | **53,67** |

Tiempo: 35 a 42 segundos de punta a punta en una máquina Linux de 8 CPU y 16 GB de RAM, sin GPU (ingesta 1 s, coordinación 9 s, narrativas 25 s).

Para no engañarnos con el set de práctica hicimos tres pruebas extra:

- **Temas que el modelo no vio:** sacamos todo el tema litio y nos quedamos con 10 días (93.500 comentarios). El pipeline encontró las 8 narrativas restantes sin tocar nada: 51,6 de 55.
- **Sin el clasificador entrenado:** si borramos `modelos/postura.joblib`, la postura baja de 0,999 a 0,87 y el total queda en 51,6. Ese clasificador se entrenó con la verdad de práctica, así que en el set final esperamos una postura entre esos dos valores.
- **Datos sucios:** archivo en Latin-1, fechas vacías o rotas, fechas sin zona, textos vacíos, respuestas a comentarios inexistentes e ids duplicados. Corre igual y saca 53,6.

## Cómo funciona

```
CSV/CSV.gz ──> DuckDB + Parquet ──┬──> Coordinación: sincronía vs. mundo nulo ──> grupos
                                  └──> Texto: muletillas ──> núcleos únicos ──> embeddings
                                                                 │
                                       grafo de conversación ────┴──> narrativas, ruido, postura, origen
```

### 1. Ingesta (`qn/ingesta.py`)
Lee todo como texto y recién después convierte tipos, así un valor roto no tira abajo la corrida. Detecta el encoding, parsea fechas en varios formatos (si falta la zona asume `-03:00`), imputa fechas faltantes con la mediana de la publicación, saca ids duplicados y anula respuestas a comentarios que no existen.

### 2. Texto (`qn/texto.py`)
La gente dice lo mismo con muletillas distintas: «che,», «la verdad que», «qué bronca», emojis. **No usamos una lista fija:** aprendemos las muletillas del propio set. Un prefijo o sufijo es muletilla si, al sacarlo, queda un texto que también aparece solo, y eso pasa con muchos textos distintos. Así funciona con cualquier tema. En práctica, 200.000 comentarios quedan en unos 10.000 «núcleos» únicos, y solo esos pasan por el modelo.

### 3. Narrativas (`qn/narrativas.py`)
- **Significado:** embeddings multilingües locales (`paraphrase-multilingual-MiniLM-L12-v2`, ONNX en CPU vía `fastembed`) de cada núcleo único.
- **Conversación:** el desafío dice que una respuesta que contradice a otra es de la misma narrativa con la postura opuesta. Con eso armamos un grafo entre núcleos (A responde discutiendo a B) y buscamos comunidades con Louvain. **Cada comunidad es una narrativa y la cantidad sale sola.** Los núcleos «comodín» que discuten con todos se detectan y se sacan.
- **Postura:** dentro de cada comunidad, las aristas son «posturas opuestas», así que partimos el grafo en dos lados. Para decidir qué lado afirma la narrativa votan varias señales: el clasificador liviano entrenado en práctica, el tamaño de cada lado, el largo de los textos, los me gusta, cuál de los dos responde más para refutar y qué lado empujan las cuentas coordinadas. Si hay una IA configurada (opcional, ver abajo), ella decide con una sola consulta por narrativa.
- **Variantes y typos:** los núcleos que no aparecen en ninguna discusión se asignan por similitud de embeddings al núcleo conocido más parecido (coseno ≥ 0,80).
- **Ruido:** las respuestas cortas de acompañamiento («Tal cual», «Coincido») se detectan solas y heredan la narrativa y la postura del comentario al que responden. Si aparecen sueltas, son ruido. También es ruido un texto que aparece muchas veces y nunca entra en una discusión, porque para una narrativa real eso es estadísticamente improbable. Y lo que no se parece a nada conocido.
- **Origen:** detectamos el arranque, es decir, el primer momento en que el volumen de la narrativa en 3 horas supera el 3% de su pico, y tomamos el primer comentario de las 3 horas previas. Así un comentario suelto mal asignado días antes no adelanta la fecha.
- **Título y frases clave:** la frase más repetida del lado que afirma, y c-TF-IDF para las frases clave.

### 4. Coordinación (`qn/coordinacion.py`)
- **Sincronía:** para cada par de cuentas contamos en cuántas publicaciones distintas comentaron con menos de 10 minutos de diferencia. Cada coincidencia pesa menos cuanto más concurrida estaba la publicación en ese momento, porque veinte comentarios en diez minutos es normal en un posteo viral y raro en uno que nadie mira. El total se normaliza por la actividad de cada cuenta.
- **Contra lo esperable:** generamos un **mundo nulo** barajando las cuentas dentro de cada publicación, con los mismos horarios y la misma actividad pero sin coordinación. El umbral es el percentil 99,95 de ese mundo nulo, así que se adapta solo a cada set.
- **Respuestas entre sí:** si dos cuentas se responden y además coinciden en horario, el vínculo se refuerza. Así aparecen las operaciones con textos variados que se responden entre ellas.
- **Operaciones:** comunidades Louvain sobre ese grafo, con densidad mínima. El tamaño mínimo de grupo también sale del mundo nulo: corremos lo mismo sobre los datos barajados y descartamos todo grupo que no supere al más grande que aparece por azar.

## Dónde guardamos los datos y por qué DuckDB

Los CSV se cargan en **DuckDB** (`trabajo/<set>/qualydata.duckdb`) y se exportan a **Parquet** comprimido con zstd (`comentarios.parquet`, `cuentas.parquet`, `publicaciones.parquet`). Los resultados (etiquetas, grupos y narrativas) quedan como tablas en la misma base.

- **Es analítico y columnar:** lo pesado del desafío son agregaciones y self-joins por ventana temporal (los pares de cuentas que coinciden en minutos). DuckDB los resuelve en segundos sobre millones de filas, en paralelo y con spill a disco si falta RAM.
- **Embebido:** no hay servidor que instalar ni mantener. El jurado lo reproduce con un `pip install`.
- **Parquet es el formato estándar de un data lake:** pesa unas 4 veces menos que el CSV sin comprimir y lo leen Spark, BigQuery, ClickHouse, Polars o pandas sin conversión.
- **SQL estándar:** las consultas se pasan tal cual a MotherDuck, ClickHouse o BigQuery cuando el volumen lo pide.

## Cómo escalaría a millones de comentarios por día

- **Ingesta incremental:** Parquet particionado por día (`fecha=AAAA-MM-DD/`) en un bucket. DuckDB lee solo las particiones nuevas.
- **Embeddings, que es lo más caro:** solo se calculan para núcleos nuevos y se guardan en caché. Como los textos se repiten mucho, el volumen real que pasa por el modelo es una fracción chica: en práctica, el 5%. Con ONNX en CPU son unos 400 núcleos por segundo por máquina, y se reparten entre workers sin estado.
- **Asignación online:** cada comentario nuevo se asigna en milisegundos por su núcleo, si ya existe, o por vecino más cercano en un índice vectorial. El grafo y las comunidades se recalculan cada hora sobre la ventana reciente.
- **Coordinación en streaming:** los pares de cuentas se cuentan por ventana de 10 minutos y publicación, agregados de forma incremental. El mundo nulo se recalcula una vez por día.
- **Costo:** sin APIs pagas el costo marginal es CPU. Con IA opcional es una consulta por narrativa nueva, no por comentario. Unas decenas de consultas por día sin importar el volumen.

## IA opcional

Por defecto no usa IA (`tokens_ia: 0`). Si querés que una IA decida qué lado afirma cada narrativa y le ponga título, definí estas variables de entorno, nunca en el repo:

```bash
export QN_IA_URL="https://api.x.ai/v1/chat/completions"   # cualquier endpoint compatible con OpenAI
export QN_IA_CLAVE="tu-clave"
export QN_IA_MODELO="nombre-del-modelo"
export QN_IA_USD_MILLON="0.5"                             # precio por millón de tokens, para costo.json
```

Hace una sola consulta por narrativa (unas 13 en total, cerca de 500 tokens cada una) y lo registra en `costo.json`. Si algo falla, el pipeline sigue sin IA.

## Estructura del repo

```
correr.py              comando único de punta a punta
entrenar_postura.py    entrena el clasificador auxiliar de postura con la verdad de práctica
qn/ingesta.py          carga robusta a DuckDB + Parquet
qn/texto.py            normalización y aprendizaje automático de muletillas
qn/narrativas.py       embeddings, grafo de conversación, narrativas, ruido, postura y origen
qn/coordinacion.py     sincronía, mundo nulo y grupos coordinados
qn/tablero.py          tablero HTML estático para la demo
qn/ia.py               IA opcional, una consulta por narrativa
modelos/postura.joblib clasificador auxiliar (TF-IDF + regresión logística, 120 KB)
entrega/               resultado del set final (se commitea el viernes)
PITCH.md               guion del pitch
```

`datos/`, `salida/`, `trabajo/` y `.cache/` no se suben al repo.

## Qué existía antes del evento

Todo el código de este repo se escribió durante el hackathon. Usamos solo librerías y modelos públicos y abiertos:

- [DuckDB](https://duckdb.org), [pandas](https://pandas.pydata.org), [PyArrow](https://arrow.apache.org), [NumPy](https://numpy.org)
- [scikit-learn](https://scikit-learn.org) para TF-IDF y la regresión logística
- [NetworkX](https://networkx.org) para los grafos y Louvain
- [fastembed](https://github.com/qdrant/fastembed) y [ONNX Runtime](https://onnxruntime.ai) para correr los embeddings en CPU
- El modelo [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (Apache 2.0)
- Del repo del desafío usamos los datos y `evaluacion/puntuar.py` para medirnos. No copiamos código de la línea base.

Usamos asistentes de IA para programar más rápido, como permiten las reglas.

## Licencia

MIT. Ver `LICENSE`.
