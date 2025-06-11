# CAMI: Classificatore Automatico di Metafore per l’Italiano


CAMI è un software basato su Python progettato per l'analisi, l'identificazione e l'interpretazione di metafore nella lingua italiana. Sfrutta modelli di linguaggio Transformer (UmBERTo) per eseguire compiti di Natural Language Processing complessi, offrendo strumenti sia per analisi su larga scala (Distant Reading) sia per analisi di dettaglio (Close Reading).

## Funzionalità

-   **Classificazione Metaforica**: Distingue tra frasi letterali e metaforiche utilizzando un modello di classificazione di sequenze fine-tuned.
-   **Distant Reading**: Analizza corpus di testo di grandi dimensioni (es. romanzi) per calcolare un **indice di figuralità**, che misura la densità metaforica del testo.
-   **Close Reading**:
    -   **Estrazione di Entità**: Identifica e estrae i componenti chiave della metafora, ovvero l'**Argomento** (ciò di cui si parla) e il **Veicolo** (ciò con cui lo si paragona).
    -   **Analisi Vettoriale**: Rappresenta Argomento e Veicolo su uno spazio 2D per visualizzare la loro "distanza semantica".
    -   **Interpretabilità del Modello**: Genera heatmap delle matrici di attenzione per mostrare quali parole il modello ritiene più correlate durante l'inferenza.
-   **Human-in-the-Loop (Progettato)**: L'architettura è pensata per integrare un ciclo di feedback umano, permettendo al modello di migliorare nel tempo tramite ri-addestramento su dati corretti dall'utente.

## Architettura del Modello

Il sistema si basa su due modelli principali, entrambi derivati da architetture Transformer e fine-tuned su un dataset specifico di metafore italiane:

1.  **Classificatore di Sequenze**: Un modello (`cami_classifier_tuned`) basato su **UmBERTo** che classifica un'intera frase.
2.  **Classificatore di Token (NER)**: Un modello (`cami_ner_v2`) basato su **`dbmdz/bert-base-italian-xxl-cased`** che etichetta ogni singola parola (token) della frase per estrarre Argomento e Veicolo.

Il progetto è sviluppato utilizzando PyTorch e la libreria `transformers` di Hugging Face.

## Struttura del Progetto

Ecco una descrizione dei file principali presenti in questa repository:

-   `data/`: Cartella destinata a contenere i dati.
    -   `metafore_dataset.csv`: Il dataset principale per l'addestramento.
    -   `nuovi_testi/`: La cartella dove inserire i file `.txt` da analizzare con lo strumento di Distant Reading.
-   `.gitignore`: Specifica quali file e cartelle ignorare (es. l'ambiente virtuale `venv/`, i modelli salvati in `models/`).
-   `requirements.txt`: Elenca tutte le librerie Python necessarie per far funzionare il progetto.

### Script di Addestramento

-   `train_classifier.py`: **[Fase 1]** Primo script per il training del modello di classificazione. Semplice e basilare.
-   `train_advanced.py`: **[Fase 1.1]** Versione avanzata dello script di training per il classificatore, con tuning degli iperparametri e calcolo di metriche avanzate (F1-score, etc.).
-   `train_ner.py`: **[Fase 3]** Script per addestrare il modello di Token Classification (NER) per l'estrazione di Argomento e Veicolo.

### Script dell'Applicazione

-   `cami_app.py`: **[Distant Reading]** Applicazione principale che carica il modello classificatore e analizza tutti i file `.txt` presenti nella cartella `data/nuovi_testi/`.
-   `cami_app_demo_con_visuals.py`: **[Close Reading]** Script di demo che mostra l'integrazione di tutti i modelli. Classifica una serie di frasi di test, estrae Argomento e Veicolo e genera le visualizzazioni (spazio vettoriale 2D e heatmap di attenzione).
-   `cami_demo.py` / `cami_app_demo_con_ner.py`: Versioni di sviluppo intermedie. La versione più aggiornata per la demo è `cami_app_demo_con_visuals.py`.

## Installazione

Per eseguire il progetto in locale, segui questi passaggi:

1.  **Clona la repository:**
    ```bash
    git clone https://github.com/tuo-username/CAMI.git
    cd CAMI
    ```

2.  **Crea un ambiente virtuale:**
    ```bash
    python3 -m venv venv
    ```

3.  **Attiva l'ambiente virtuale:**
    -   Su macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    -   Su Windows:
        ```bash
        venv\Scripts\activate
        ```

4.  **Installa le dipendenze:**
    ```bash
    pip install -r requirements.txt
    ```

## Utilizzo

### 1. Addestrare i Modelli

Se vuoi ri-addestrare i modelli da zero (ad esempio, dopo aver aggiornato il dataset), puoi usare gli script di training.

-   **Per addestrare il classificatore di frasi:**
    ```bash
    python train_advanced.py
    ```
-   **Per addestrare il modello di estrazione (NER):**
    ```bash
    python train_ner.py
    ```
    I modelli addestrati verranno salvati nella cartella `models/`.

### 2. Eseguire l'Analisi Distant Reading

1.  Aggiungi i tuoi file di testo (`.txt`) nella cartella `data/nuovi_testi/`.
2.  Esegui lo script `cami_app.py`:
    ```bash
    python cami_app.py
    ```
    Lo script analizzerà ogni file e stamperà un report dettagliato nel terminale.

### 3. Eseguire la Demo di Close Reading con Visualizzazioni

Per vedere le funzionalità di estrazione e visualizzazione in azione su frasi di esempio:
```bash
python cami_app_demo_con_visuals.py