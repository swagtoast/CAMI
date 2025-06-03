"""
semantics.py

Modulo con funzioni per misure semantiche basate su WordNet/OMW per l'italiano.

Funzioni:
- distanza_semantica(argomento, veicolo): calcola similarità massima (path_similarity o wup_similarity) fra synset italiani.
- stima_concretezza(parola): indica se la parola è concreta (True), astratta (False) o None se non disponibile.

Nota sui fallback:
- Se la parola non è trovata in WordNet italiano, si restituisce None.
- Potrebbe essere necessario un lookup in BabelNet o uso di un dizionario inglese come fallback, non implementato qui.
"""

import nltk
from nltk.corpus import wordnet as wn

# Scarica risorse necessarie (se non già presenti)
nltk.download('omw-1.4')
nltk.download('wordnet')

def distanza_semantica(argomento: str, veicolo: str):
    """
    Calcola la massima similarità semantica tra due parole italiane (argomento, veicolo) usando OMW.
    
    Ritorna un valore float tra 0 e 1, oppure None se una delle parole non è trovata.
    """
    synsets1 = wn.synsets(argomento, lang='ita')
    synsets2 = wn.synsets(veicolo, lang='ita')
    if not synsets1 or not synsets2:
        return None
    
    max_sim = 0.0
    for s1 in synsets1:
        for s2 in synsets2:
            # Prova path_similarity e wu_palmer_similarity
            sim = s1.path_similarity(s2)
            if sim is None:
                sim = s1.wup_similarity(s2)
            if sim is None:
                sim = 0.0
            if sim > max_sim:
                max_sim = sim
    return max_sim

def stima_concretezza(parola: str):
    """
    Stima se una parola italiana è concreta o astratta.
    Usa WordNet italiano: verifica se tra gli iperonimi compare 'entità fisica'.
    
    Ritorna:
        True se è concreta (presenza di 'entità fisica' tra gli iperonimi)
        False se non lo è (non trova catena con 'entità fisica')
        None se la parola non è trovata in WordNet italiano
    """
    synsets = wn.synsets(parola, lang='ita')
    if not synsets:
        return None
    
    # Prova a ottenere synset di 'entità fisica' in italiano
    phys_synsets = wn.synsets('entità fisica', lang='ita')
    if not phys_synsets:
        # Se non esiste, si restituisce None come fallback
        return None
    
    for syn in synsets:
        # Per ogni synset, prendi tutti i percorsi di iperonimi
        for path in syn.hypernym_paths():
            for hyper in path:
                # Confronta con i synset di 'entità fisica'
                if hyper in phys_synsets:
                    return True
    # Se nessuna corrispondenza
    return False