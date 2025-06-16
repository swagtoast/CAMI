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
2.  **Classificatore di Token (NER)**: Un modello (`cami_ner_v4_robust`) basato su **`dbmdz/bert-base-italian-xxl-cased`** che etichetta ogni singola parola (token) della frase per estrarre Argomento e Veicolo, addestrato con un robusto sistema di lemmatizzazione.

Il progetto è sviluppato utilizzando PyTorch e le librerie `transformers` e `spaCy`.

## Struttura del Progetto

-   `data/`: Cartella destinata a contenere i dati.
    -   `metafore_dataset.csv`: Il dataset principale per l'addestramento.
    -   `nuovi_testi/`: La cartella dove inserire i file `.txt` da analizzare.
-   `requirements.txt`: Elenca tutte le librerie Python necessarie.
-   `train_ner_full.py`: Script per addestrare il modello NER robusto.
-   `train_advanced.py`: Script per addestrare il modello di classificazione.
-   `cami_app_full_autonomous.py`: Script di demo che mostra l'integrazione di tutti i modelli.

## Installazione

Per eseguire il progetto in locale, segui questi passaggi nell'ordine corretto:

1.  **Clona la repository:**
    ```bash
    git clone [https://github.com/tuo-username/CAMI.git](https://github.com/tuo-username/CAMI.git)
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

4.  **Installa le dipendenze Python:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Scarica il modello linguistico `spaCy` per l'italiano:**
    ```bash
    python -m spacy download it_core_news_lg
    ```

## Utilizzo

### 1. Addestrare i Modelli

Se vuoi ri-addestrare i modelli da zero:

-   **Per addestrare il classificatore di frasi:**
    ```bash
    python train_advanced.py
    ```
-   **Per addestrare il modello di estrazione (NER) robusto:**
    ```bash
    python train_ner_full.py
    ```

### 2. Eseguire la Demo Autonoma (Close Reading)

Per vedere il sistema completo in azione con analisi e visualizzazioni:
```bash
python cami_app_full_autonomous.py