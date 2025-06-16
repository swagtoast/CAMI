# CAMI: Classificatore Automatico di Metafore per l’Italiano

CAMI è un software basato su Python progettato per l'analisi, l'identificazione e l'interpretazione di metafore nella lingua italiana. Sfrutta modelli di linguaggio Transformer per eseguire compiti complessi di Natural Language Processing, offrendo strumenti sia per analisi quantitative (Distant Reading) sia per analisi qualitative e di dettaglio (Close Reading).

Questo progetto è stato sviluppato come un dialogo interattivo, evolvendo passo dopo passo per risolvere bug, migliorare le strategie di training e approfondire l'interpretabilità dei modelli.

## Funzionalità

-   **Classificazione Metaforica**: Distingue tra frasi letterali e metaforiche utilizzando un modello di classificazione di sequenze basato su UmBERTo.
-   **Estrazione di Entità (NER)**: Identifica e estrae autonomamente i componenti chiave della metafora, ovvero l'**Argomento** (ciò di cui si parla) e il **Veicolo** (ciò con cui lo si paragona), grazie a un modello NER robusto che gestisce le varianti morfologiche (es. singolari/plurali).
-   **Analisi Vettoriale (PCA)**: Rappresenta Argomento e Veicolo su uno spazio 2D per visualizzare la loro "distanza semantica", offrendo spunti su come il modello codifica il significato.
-   **Interpretabilità del Modello (Attention Heatmaps)**: Genera heatmap delle matrici di attenzione per mostrare quali parole il modello ritiene più correlate durante l'analisi, fornendo una finestra sul suo processo decisionale.
-   **Analisi di Corpus (Distant Reading)**: Include uno script (`cami_app.py`) per analizzare corpus di testo di grandi dimensioni (es. romanzi) e calcolare un **indice di figuralità**, che misura la densità metaforica del testo.

## Architettura del Modello

Il sistema si basa su due modelli Transformer fine-tuned su un dataset specifico di metafore italiane:

1.  **Classificatore di Sequenze (`models/cami_classifier_tuned`)**:
    * **Base**: `Musixmatch/umberto-commoncrawl-cased-v1`.
    * **Scopo**: Classificare un'intera frase come "Letterale" o "Metaforica".

2.  **Estrattore di Entità (`models/cami_ner_v5_final`)**:
    * **Base**: `dbmdz/bert-base-italian-xxl-cased`.
    * **Scopo**: Eseguire il "Close Reading", etichettando le singole parole (token) come `ARG` (Argomento) e `VEI` (Veicolo). Il training è reso robusto dall'uso di **spaCy** per la lemmatizzazione, permettendo di gestire plurali e forme flesse.

## Struttura del Progetto

-   `data/`: Contiene i dati di training e i testi per l'analisi.
    -   `metafore_dataset.csv`: Il dataset principale per l'addestramento.
    -   `nuovi_testi/`: Cartella dove inserire i file `.txt` da analizzare.
-   `requirements.txt`: Elenca tutte le dipendenze Python.
-   **Script di Addestramento**:
    -   `train_advanced.py`: Addestra il modello Classificatore.
    -   `train_ner_full.py`: Addestra il modello Estrattore (NER) robusto.
-   **Script di Esecuzione**:
    -   `demo_completa.py`: Esegue una demo completa end-to-end che classifica, estrae e visualizza i risultati per un set di frasi di test.
    -   `cami_app.py`: Esegue l'analisi di Distant Reading su interi file di testo.

## Installazione

Per eseguire il progetto in locale, segui questi passaggi nell'ordine corretto:

1.  **Clona la repository** (se non l'hai già fatto):
    ```bash
    git clone [https://github.com/tuo-username/CAMI.git](https://github.com/tuo-username/CAMI.git)
    cd CAMI
    ```

2.  **Crea e attiva un ambiente virtuale**:
    ```bash
    # Crea l'ambiente
    python3 -m venv venv
    # Attiva l'ambiente (su macOS/Linux)
    source venv/bin/activate
    # Oppure, su Windows:
    # venv\Scripts\activate
    ```

3.  **Installa le dipendenze Python**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Scarica il modello linguistico `spaCy` per l'italiano**:
    ```bash
    python -m spacy download it_core_news_lg
    ```

## Utilizzo

### 1. Addestrare i Modelli (Opzionale)

Se vuoi ri-addestrare i modelli da zero (ad esempio, dopo aver modificato il dataset), esegui questi due comandi:

```bash
# 1. Addestra il classificatore di frasi
python train_advanced.py

# 2. Addestra l'estrattore di Argomento/Veicolo
python train_ner_full.py