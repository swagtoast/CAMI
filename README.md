# CAMI: Classificatore Automatico di Metafore per l’Italiano

Questo progetto implementa un sistema per l'identificazione automatica di metafore in testi italiani, utilizzando un modello Transformer pre-addestrato (UmBERTo) e tecniche di NLP.

## Struttura del Progetto

-   `data/`: Contiene il dataset (`CAMI_dataset_v2.csv`) e i testi per l'inferenza (`nuovi_testi/`).
-   `cami_model/`: Directory dove viene salvato il modello fine-tunato.
-   `data_utils.py`: Utility per il caricamento, la pulizia e la suddivisione dei dati.
-   `train_model.py`: Script per il fine-tuning del modello UMBERTO sul dataset CAMI.
-   `evaluate_model.py`: Script per valutare le performance del modello fine-tunato.
-   `predict.py`: Script per effettuare inferenze su nuovi testi e calcolare l'indice di figuralità.
-   `semantics.py`: Modulo per analisi semantiche (distanza semantica, stima concretezza) usando WordNet.

## Prerequisiti

Assicurati di avere Python 3.8+ installato. Le librerie necessarie possono essere installate tramite pip:

```bash
pip install pandas torch torchvision torchaudio transformers datasets scikit-learn nltk matplotlib plotly
```


## Setup in Google Colab

1.  **Carica i file del progetto:**
    *   Crea una cartella principale, ad esempio `CAMI_Project` nel tuo Google Drive e carica tutti i file `.py`.
    *   Oppure, carica i file direttamente nell'ambiente di runtime di Colab (saranno persi alla chiusura della sessione, a meno che non si monti Drive).

2.  **Prepara i dati:**
    *   Crea una sottocartella `data` all'interno di `CAMI_Project`.
    *   Carica `CAMI_dataset_v2.csv` in `CAMI_Project/data/`.
    *   Crea una sottocartella `nuovi_testi` in `CAMI_Project/data/`.
    *   Carica i file `.txt` contenenti i testi su cui fare inferenza in `CAMI_Project/data/nuovi_testi/`.

3.  **Monta Google Drive (consigliato per persistenza):**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    # Imposta il path di lavoro
    import os
    os.chdir('/content/drive/MyDrive/CAMI_Project') # Adatta questo path se necessario
    ```

4.  **Download risorse NLTK:**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4') # Open Multilingual WordNet
    ```

## Esecuzione del Progetto



1.  **Training del Modello:**
    *   Esegui `train_model.py`:
        ```python
        !python train_model.py
        ```
    *   Questo script caricherà i dati, fine-tunerà il modello UMBERTO e lo salverà in `cami_model/`.

2.  **Valutazione del Modello:**
    *   Esegui `evaluate_model.py`:
        ```python
        !python evaluate_model.py
        ```
    *   Questo script caricherà il modello fine-tunato e valuterà le sue performance sul test set.

3.  **Inferenza su Nuovi Testi:**
    *   Assicurati che i tuoi file `.txt` siano in `data/nuovi_testi/`.
    *   Esegui `predict.py`:
        ```python
        !python predict.py
        ```
    *   Questo script analizzerà i testi, predierà le metafore, calcolerà l'indice di figuralità e salverà i risultati in `figurality_results.csv`.
