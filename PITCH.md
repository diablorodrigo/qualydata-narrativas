# Guion del pitch (3 a 5 minutos)

**Demo:** abrir `entrega/tablero.html` en el navegador. Es un solo archivo y funciona sin internet.

## 1. El problema (30 s)
«Cuando miles de personas discuten una obra, un hospital o un aumento, hay dos preguntas: qué se está diciendo y qué parte está armada. Leerlo a mano es imposible. Pasar cada comentario por una IA es lento y caro. Buscar palabras clave falla, porque la gente dice lo mismo de mil maneras.»

## 2. La idea (60 s)
«Usamos dos señales que se complementan.»
- **Qué significa cada texto:** primero sacamos las muletillas («che», «qué bronca», emojis). No usamos una lista: el sistema las aprende solo del set. Así 200.000 comentarios quedan en 10.000 frases únicas, y solo esas pasan por un modelo de embeddings chico que corre en la notebook.
- **Cómo discute la gente:** cuando alguien responde para refutar, está hablando de la misma narrativa desde la vereda de enfrente. Ese grafo de discusiones nos da las narrativas, la cantidad sale sola, y además la postura: cada narrativa se parte en dos lados.
- **Coordinación contra lo esperable:** no alcanza con una ráfaga. Medimos qué pares de cuentas coinciden en las mismas publicaciones con minutos de diferencia y lo comparamos con un mundo barajado, con los mismos horarios y sin coordinación. Solo queda lo que el azar no explica.

## 3. Demo del tablero (60 a 90 s)
- **Arriba:** volumen procesado, cantidad de narrativas, % de ruido, operaciones detectadas, tiempo y costo.
- **Narrativas:** título, frases clave, desde cuándo circula (las emergentes están marcadas), curva de volumen con el origen y el balance entre apoya y rechaza.
- **Quién la empuja:** la etiqueta del grupo coordinado al lado de cada narrativa.
- **Operaciones:** tamaño, antigüedad de las cuentas, horario (por ejemplo, ráfagas de madrugada) y los textos que repiten.
- Mostrar un caso: una narrativa emergente que nace en una ráfaga nocturna de cuentas creadas días antes.

## 4. Resultados (30 s)
- En el set de práctica: **54,9 de 55 puntos automáticos**, contra 16,8 de la línea base.
- Narrativas 29,9 de 30 y coordinación **25,0 de 25** (100% precisión y recall en las 3 operaciones, incluida la de actividad de cobertura).
- Lo probamos con temas que no vio (51,6), sin el clasificador entrenado (51,6) y con datos sucios (53,6).
- Set final: [completar el viernes con tiempo de corrida y lo que encontró].

## 5. Costo, escala y almacenamiento (40 s)
- **Unos 40 segundos para 200.000 comentarios** en CPU, sin GPU. Costo de IA cero: `costo.json` lo registra.
- **DuckDB sobre Parquet:** columnar, embebido y en SQL estándar. Resuelve en segundos los cruces por ventana de tiempo, y el Parquet sirve tal cual para un data lake.
- **Millones por día:** como los textos se repiten, solo las frases nuevas pasan por el modelo. La asignación es online, el grafo se recalcula por hora y la coordinación se agrega por ventana de 10 minutos. Si se suma IA, es una consulta por narrativa, no por comentario.

## 6. Cierre (15 s)
«Un comando, sin APIs pagas, reproducible y explicable: qué circula, desde cuándo y quién lo empuja.»

## Preguntas probables
- **¿Y si no hay respuestas en los datos?** Los embeddings igual agrupan por significado, y el grafo de respuestas es un refuerzo. Las respuestas son una señal que el propio desafío plantea.
- **¿Cómo decide qué lado apoya?** Votan varias señales: el clasificador liviano, el tamaño, el largo de los textos, los me gusta, cuál lado refuta y qué lado empujan las campañas. Con IA opcional, una consulta por narrativa.
- **¿No es sobreajuste?** Los umbrales salen de los datos: muletillas aprendidas, mundo nulo y tamaño mínimo de narrativa relativo al volumen. Lo validamos sacando un tema entero.
