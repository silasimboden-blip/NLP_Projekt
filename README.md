# NLP-Inference Service mit Microbatching, Prometheus und Grafana

CAS AI Operations — NLP Operations — Projekt.

Klassifiziert Kommentare einer fiktiven Online-Zeitung mit der Hugging-Face
Zero-Shot-Pipeline (`facebook/bart-large-mnli`) in eine von vier Kategorien.
Eingehende Requests werden in Mikrobatches gebündelt (Zeitfenster ODER
maximale Grösse) und gemeinsam klassifiziert.

## Quickstart

```bash
# 1. Stack starten (erster Lauf lädt ~1.6 GB Modell)
docker compose up -d

# 2. Healthcheck abwarten (kann beim ersten Start ein paar Minuten dauern)
until curl -fs http://localhost:8080/healthz; do sleep 2; done && echo

# 3. Smoke-Test
curl -X POST http://localhost:8080/classify \
  -H 'content-type: application/json' \
  -d '{"comment":"Sehr guter Artikel, danke!"}'

# 4. Last erzeugen, damit das Dashboard Daten zeigt
python scripts/loadgen.py --scenario mixed --duration 120

# 5. Dashboard öffnen
open http://localhost:3000   # admin / admin
```

## Endpoints

| URL | Beschreibung |
|---|---|
| `http://localhost:8080/classify` | `POST` mit `{"comment": "..."}`, gibt `label` + `score` zurück |
| `http://localhost:8080/metrics` | Prometheus-Metriken im Textformat |
| `http://localhost:8080/healthz` | Liveness-Check |
| `http://localhost:9090` | Prometheus UI |
| `http://localhost:3000` | Grafana (admin/admin), Dashboard "NLP Classifier Service" automatisch geladen |

## Lasttest fahren

```bash
# Size-Trigger demonstrieren: 16 gleichzeitige Requests in flight,
# Batches füllen sich auf MAX_BATCH_SIZE=8
python scripts/loadgen.py --scenario burst --duration 60

# Time-Trigger demonstrieren: ~3 req/s, Batches sind klein und schliessen nach 200ms
python scripts/loadgen.py --scenario trickle --duration 60

# Beide Trigger in einem Lauf: 10-Bursts, gefolgt von 1s Pause
python scripts/loadgen.py --scenario mixed --duration 120
```

Im Dashboard zeigt das Panel **Durchschnittliche Batch-Grösse** den
Unterschied am deutlichsten: bei `burst` nahe 8, bei `trickle` nahe 1.

`burst` ist **concurrency-limitiert**, nicht rate-limitiert: es hält konstant
`--concurrency` (Default 16) Requests gleichzeitig in flight statt mit fester
req/s zu feuern. Auf CPU braucht `facebook/bart-large-mnli` ~25–35 s pro Batch
von 8; eine feste hohe Rate (z. B. 25 req/s) würde die Queue sofort bis
`MAX_QUEUE_SIZE` fluten und fast alle Requests in den Timeout laufen lassen.
Mit der Concurrency-Begrenzung bleibt die Queue beschränkt, die Batches füllen
sich trotzdem auf 8 (Size-Trigger), und kein Request läuft in den Timeout.

## Konfiguration

Alle Einstellungen werden über Environment-Variablen in `docker-compose.yml`
gesteuert. Defaults entsprechen der Aufgabenstellung:

| Variable | Default | Bedeutung |
|---|---|---|
| `MODEL_ID` | `facebook/bart-large-mnli` | HF-Modell für Zero-Shot |
| `CANDIDATE_LABELS` | siehe unten | Komma-getrennte Liste |
| `BATCH_WINDOW_MS` | `200` | Maximale Wartezeit pro Batch |
| `MAX_BATCH_SIZE` | `8` | Maximale Anzahl Kommentare pro Batch |
| `MAX_QUEUE_SIZE` | `256` | Maximale Queue-Tiefe (Backpressure) |

Labels:
- `Zustimmung`
- `sachliche Kritik`
- `Empörung`
- `Beleidigung oder persönlicher Angriff`

### Hinweis zur Label-Verteilung

Beim Start werden alle vier Label-Counter mit dem Wert 0 initialisiert
(`COMMENTS_BY_LABEL.labels(...)` im `lifespan`). Dadurch erscheint jedes
Label im Panel "Kommentare nach Label" — auch eines, das das Modell nie
vorhersagt. Ohne diese Initialisierung gäbe es für ein nie vergebenes Label
keine Prometheus-Zeitreihe, und die Linie würde im Dashboard schlicht fehlen.

Konkret: `facebook/bart-large-mnli` ist englisch-trainiert und ordnet deutsche
Empörungs-Kommentare im Zero-Shot fast nie dem Label `Empörung` zu — die
Wahrscheinlichkeitsmasse geht an `sachliche Kritik` bzw. `Beleidigung`.
`Empörung` bleibt deshalb meist eine flache Null-Linie. Das ist eine bekannte
Zero-Shot-Limitierung des Modells, kein Fehler im Service oder Dashboard. Für
echte deutsche Klassifikationsqualität wäre ein mehrsprachiges MNLI-Modell
(z. B. `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) nötig.

## Erklärung der Metriken

| Metrik | Typ | Was sie aussagt — operationaler Nutzen |
|---|---|---|
| `comments_classified_total` | Counter | Gesamtdurchsatz. Mit `rate(...)` ergibt sich Kommentare/Sekunde. |
| `batches_processed_total` | Counter | Wie oft das Modell tatsächlich aufgerufen wurde. Kombiniert mit `comments_classified_total` ergibt sich die effektive Batch-Effizienz. |
| `batch_size` | Histogram | Verteilung der Batch-Grössen. Ein Mittelwert nahe `MAX_BATCH_SIZE` heisst: Size-Trigger dominiert (genug Last). Nahe 1: Time-Trigger dominiert (zu wenig Last für effektives Batching). |
| `batch_inference_duration_seconds` | Histogram | Wie lange das Modell pro Batch braucht. p95 → Latenz-SLO. Steigt mit Batch-Grösse, aber sublinear (das ist der Sinn von Batching). |
| `batch_wait_time_seconds` | Histogram | Wie lange ein Kommentar in der Queue wartet, bevor er klassifiziert wird. Sollte unter `BATCH_WINDOW_MS` bleiben — wenn nicht, ist die Queue überlastet. |
| `microbatch_queue_size` | Gauge | Aktuelle Queue-Tiefe. Wächst → Backpressure droht. Bei `MAX_QUEUE_SIZE` lehnt der Service neue Requests ab. |
| `comments_classified_by_label_total{label}` | Counter | Verteilung der Vorhersagen. Plötzlich viel `Empörung` oder `Beleidigung`? → Moderation aufmerksam machen. |

## Beschreibung der Microbatching-Umsetzung

Implementierung in `classifier-service/app/batcher.py`.

Ein einziger `asyncio`-Worker-Task läuft im Hintergrund und konsumiert eine
`asyncio.Queue` von Anfragen. Jede Anfrage legt einen `BatchItem` mit einem
zugehörigen `Future` in die Queue und wartet auf das Ergebnis.

Der Worker-Loop:

1. Blockiert auf das erste Item (`await queue.get()`).
2. Startet ein Zeitfenster (`deadline = now + BATCH_WINDOW_MS`).
3. Sammelt weitere Items per `asyncio.wait_for(queue.get(), timeout=remaining)`
   bis entweder **`MAX_BATCH_SIZE` erreicht** ist (Size-Trigger) oder
   **die Wartezeit abläuft** (Time-Trigger).
4. Ruft `classifier.classify_batch(...)` *einmal* mit allen gesammelten
   Kommentaren auf.
5. Verteilt die Ergebnisse über die zugehörigen Futures zurück an die
   FastAPI-Handler.

Fehlerpfad: Wirft das Modell eine Exception, wird sie auf *alle* Futures
des Batches gesetzt, so dass jeder Caller einen sauberen 500er erhält und
keine Anfrage stillschweigend hängt.

Bewusster Trade-off: Der HF-Pipeline-Aufruf ist synchron und blockiert
während der Inferenz den Event-Loop. Für eine Single-Process-Demo
akzeptabel; in einer Produktionsumgebung würde man den Aufruf in
`loop.run_in_executor(...)` einpacken.

## Screenshot

Siehe `screenshots/dashboard.png`. Neu erstellt nach einem
`scripts/loadgen.py --scenario mixed --duration 120` Lauf bei laufendem Stack.

## Tests

```bash
cd classifier-service
.venv/bin/pytest -v
```

14 Tests, kein HF-Download nötig — die Tests verwenden Stub-Klassifikatoren.

## Projektstruktur

```
.
├── classifier-service/      FastAPI-Service (app/, tests/, Dockerfile)
├── prometheus/              prometheus.yml
├── grafana/                 Provisioning + Dashboard JSON
├── scripts/                 loadgen.py, sample_comments.py
├── screenshots/             dashboard.png
└── docker-compose.yml
```
