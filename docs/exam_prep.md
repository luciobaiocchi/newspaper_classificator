Manuale Operativo: Pre-elaborazione dei Dati e Valutazione di Modelli in Python

Introduzione

Questo manuale funge da guida pratica e operativa per le fasi cruciali del ciclo di vita del machine learning: la pre-elaborazione dei dati e la valutazione delle performance dei modelli. L'obiettivo è fornire una base solida e applicabile per preparare i dati per l'analisi e per misurare in modo oggettivo l'efficacia degli algoritmi. Tutte le tecniche e gli esempi presentati sono basati esclusivamente sui concetti e sui dati forniti nel contesto di riferimento, utilizzando le librerie fondamentali dell'ecosistema Python come Pandas, Numpy e Scikit-learn. Questo documento è progettato per consolidare una comprensione chiara e applicabile di questi concetti fondamentali.


--------------------------------------------------------------------------------


1. Fondamenti delle Librerie Python per la Data Science

Prima di affrontare le tecniche avanzate di pre-elaborazione e modellazione, è essenziale padroneggiare gli strumenti di base che ne costituiscono il fondamento. Le librerie Numpy e Pandas sono i pilastri per la manipolazione, la selezione e la gestione efficiente dei dati in Python, rendendole prerequisiti indispensabili per qualsiasi progetto di data science. I seguenti sottocapitoli illustreranno le operazioni più comuni e rilevanti di queste librerie, fornendo le competenze necessarie per gestire i dati in modo programmatico e scalabile.

1.1. Operazioni Essenziali con Numpy

1. Tecniche di Indicizzazione degli Array Numpy offre un sistema di indicizzazione potente e flessibile per la selezione di dati da array multidimensionali. Consideriamo il seguente array x:
  * Slicing e Inversione: È possibile selezionare sottoinsiemi dell'array e invertire l'ordine degli elementi. La sintassi x[1:, ::-1] seleziona tutte le righe a partire dall'indice 1 (la seconda riga) fino alla fine, e per ciascuna riga seleziona tutte le colonne in ordine inverso (::-1).
  * Indicizzazione Booleana e Slicing: Si può utilizzare una lista di booleani per selezionare specifiche righe. La sintassi x[[False, True, True], :0:-1] seleziona la seconda e la terza riga (corrispondenti a True), e per queste righe seleziona le colonne dall'ultima fino alla prima (esclusa, dato che lo stop è 0).
2. Operazioni Vettoriali Numpy eccelle nell'esecuzione di operazioni numeriche su interi array (operazioni vettorializzate), che sono significativamente più efficienti dei cicli for tradizionali in Python. È possibile eseguire calcoli elemento per elemento e aggregazioni lungo assi specifici. Ad esempio, date due immagini a e b, il calcolo della somma delle differenze al quadrato può essere eseguito come segue:
3. Rappresentazione dei Valori Mancanti In Numpy, lo standard per rappresentare valori numerici mancanti è np.nan ("Not a Number"). È fondamentale notare che il tipo di np.nan è float. Questo ha un'implicazione importante: un array Numpy che contiene anche un solo np.nan avrà come tipo di dato (dtype) float. Questa scelta progettuale permette di mantenere alte performance nei calcoli numerici, poiché le operazioni su array di float sono altamente ottimizzate, a differenza di quanto accadrebbe con un array di tipo Object che potrebbe contenere l'oggetto Python None.

1.2. Manipolazione di Dati con Pandas

1. Introduzione a DataFrame e Serie Pandas introduce due strutture dati fondamentali: la Series e il DataFrame. Una Series è un array monodimensionale, simile a una colonna di un foglio di calcolo, in cui ogni elemento è associato a un indice esplicito. Un DataFrame è una struttura dati tabellare bidimensionale, simile a una tabella di un database o a un intero foglio di calcolo, in cui ogni colonna può avere un tipo di dato diverso.
2. Operazioni di Input/Output (I/O) Pandas fornisce un insieme ricco di funzioni per leggere e scrivere dati in vari formati. Di seguito sono riassunte le operazioni più comuni.

Operazione	Funzione Pandas	Esempio e Note
Lettura da CSV	pd.read_csv()	pd.read_csv('file.csv', sep=',', na_values=['no info', 'x']). Permette di specificare il separatore di colonna e una lista di stringhe da interpretare come valori nulli (NaN).
Scrittura su CSV	df.to_csv()	df.to_csv('output.csv', index=False). L'argomento index=False è cruciale per evitare di scrivere l'indice del DataFrame come prima colonna nel file CSV.
Lettura da JSON	pd.read_json()	pd.read_json('file.json'). Carica dati da un file in formato JSON, interpretando la struttura ad oggetti come un DataFrame.

1. Confronto tra i Metodi di Selezione loc e iloc Una delle distinzioni più importanti in Pandas riguarda i metodi di accesso ai dati: loc e iloc.
  * loc: È un metodo di indicizzazione basato su etichette (label-based). Quando si utilizza uno slice (es. df.loc['a':'c']), vengono inclusi sia l'indice di inizio che quello di fine.
  * iloc: È un metodo di indicizzazione basato sulla posizione intera (integer-location based), seguendo la convenzione di Python. Quando si utilizza uno slice (es. df.iloc[0:2]), l'indice di fine è escluso.
2. Ad esempio, per selezionare righe con etichette 'a' e 'c' e colonne con etichette 'Price' e 'Liters', si usa loc:
3. Selezione tramite Maschere Booleane È possibile filtrare i dati in un DataFrame creando una maschera booleana, ovvero una Serie di valori True o False. La maschera viene poi utilizzata all'interno di loc per selezionare solo le righe in cui la condizione è True.
4. Operazioni di Manipolazione del DataFrame Pandas offre un'ampia gamma di metodi per modificare la struttura e il contenuto di un DataFrame.
  * Rimozione di colonne: Utilizza il metodo drop specificando l'asse delle colonne.
  * Rinominare colonne: Il metodo rename accetta un dizionario per mappare i vecchi nomi con i nuovi.
  * Raggruppamento dei dati: Il metodo groupby è fondamentale per l'analisi "split-apply-combine", che permette di raggruppare i dati in base ai valori di una o più colonne per poi applicare funzioni di aggregazione.
  * Creazione di tabelle pivot: Il metodo pivot_table consente di riorganizzare i dati in un formato simile a quello delle tabelle pivot dei fogli di calcolo.

Una volta caricate e selezionate le porzioni di dati di interesse, il passo successivo è quasi sempre quello di pulirle e prepararle, rendendole idonee per l'addestramento dei modelli di machine learning.


--------------------------------------------------------------------------------


2. Tecniche di Pre-elaborazione dei Dati

La pre-elaborazione dei dati è una fase di importanza strategica: la qualità di un modello di machine learning dipende criticamente dalla qualità dei dati forniti in input. Dati incompleti, inconsistenti o non scalati possono degradare significativamente le performance di qualsiasi algoritmo. Questa sezione fornisce istruzioni operative per affrontare due delle sfide più comuni nel data cleaning e nella preparazione delle feature: la gestione dei valori mancanti e la normalizzazione. Queste tecniche sono fondamentali per garantire che i dati siano in una forma ottimale per l'analisi e l'addestramento dei modelli.

2.1. Gestione dei Valori Mancanti in Pandas

1. Identificare i Valori Mancanti Pandas fornisce i metodi isnull() e notnull() per identificare la presenza o l'assenza di valori NaN. Questi metodi restituiscono un oggetto (Serie o DataFrame) della stessa dimensione di quello di input, contenente valori booleani.
2. Rimuovere i Valori Mancanti Il metodo dropna() è utilizzato per rimuovere i valori mancanti. Il suo comportamento varia a seconda dell'oggetto e dei parametri utilizzati.
  * Su una Serie: Rimuove semplicemente gli elementi che sono NaN.
  * Su un DataFrame: Per impostazione predefinita, dropna() rimuove qualsiasi riga che contenga almeno un valore NaN. Questo comportamento può essere modificato:
    * axis='columns' (o axis=1): rimuove le colonne che contengono almeno un NaN.
    * how='all': rimuove solo le righe (o colonne) in cui tutti i valori sono NaN.
3. Sostituire i Valori Mancanti Anziché eliminare dati, spesso è preferibile sostituire i valori mancanti, un processo noto come imputazione. Il metodo fillna() permette di riempire i NaN con un valore specifico, come 0, o un valore calcolato statisticamente, come la media o la mediana della colonna. Questo approccio previene la perdita di informazioni che si verificherebbe con dropna().

2.2. Normalizzazione e Trasformazione delle Feature

1. Analisi della Normalizzazione Z-Score La normalizzazione Z-score (o standardizzazione) è una tecnica comune che trasforma le feature in modo che abbiano una media di 0 e una deviazione standard di 1. Questo è particolarmente utile per algoritmi sensibili alla scala delle feature di input. La formula è:
2. x' = (x - mean) / std
3. È cruciale che la media (mean) e la deviazione standard (std) utilizzate per normalizzare sia il training set che il test set siano calcolate esclusivamente sul training set.
4. Consideriamo un punto di test t1=[5.5, 1, 4] e i seguenti valori calcolati sul training set: mean=[7.5, 10, 1] e std=[2.5, 10, 1]. Il calcolo della normalizzazione per t1 è:
  * freq': (5.5 - 7.5) / 2.5 = -2.0 / 2.5 = -0.8
  * rpm': (1 - 10) / 10 = -9.0 / 10 = -0.9
  * power': (4 - 1) / 1 = 3.0 / 1 = 3.0
5. Il punto di test normalizzato è quindi [-0.8, -0.9, 3.0].
6. Illustrazione delle Pipeline di Trasformazione con Scikit-learn Scikit-learn offre il concetto di Pipeline per concatenare più passaggi di pre-elaborazione e modellazione in un unico oggetto. Questo approccio semplifica il codice, previene errori comuni (come il data leakage dal test set) e rende il flusso di lavoro più riproducibile.
7. Il seguente codice costruisce una pipeline che esegue due trasformazioni in sequenza:
  * MinMaxScaler(): Scala ogni feature in un intervallo predefinito (di default [0, 1]).
  * FunctionTransformer(): Applica una funzione personalizzata ai dati. In questo caso, una funzione lambda 1/(x**2+1).
8. L'output del comando fit_transform è:
9. Per comprendere questo risultato, analizziamo il processo passo dopo passo:
  1. Step 1 (MinMaxScaler): MinMaxScaler scala ciascuna colonna di X nell'intervallo [0, 1].
    * La prima colonna [1, 3, 2] viene trasformata in [0, 1, 0.5].
    * La seconda colonna [2, 5, 10] viene trasformata in [0, 0.375, 1].
    * L'array intermedio dopo questo passo è [[0, 0], [1, 0.375], [0.5, 1]].
  2. Step 2 (FunctionTransformer): La funzione lambda 1/(x**2+1) viene applicata elemento per elemento all'array intermedio.
    * Ad esempio, 1/(0**2+1) diventa 1, 1/(1**2+1) diventa 0.5, e 1/(0.5**2+1) diventa 0.8.
    * Questo processo genera l'array di output finale mostrato sopra.

Una volta che i dati sono stati puliti, trasformati e scalati, il passo successivo è addestrare e, soprattutto, valutare diversi tipi di modelli di machine learning per risolvere il problema in esame.


--------------------------------------------------------------------------------


3. Valutazione delle Performance dei Modelli

L'addestramento di un modello di machine learning è solo una parte del processo; valutarne l'efficacia in modo rigoroso è altrettanto cruciale per determinarne l'utilità nel mondo reale e per confrontare diverse alternative. Questa sezione tratterà le metriche e i concetti teorici fondamentali per valutare i tre principali tipi di task di machine learning: classificazione, regressione e clustering. Tutti gli esempi si baseranno esclusivamente sui dati e sulle formule fornite nel contesto di riferimento. Una valutazione quantitativa e rigorosa è l'unico modo per scegliere il modello più performante e affidabile per un determinato compito.

3.1. Valutazione di Modelli di Classificazione

1. Illustrazione del Funzionamento del K-Nearest Neighbors (KNN) L'algoritmo KNN classifica un nuovo punto dati basandosi sulla classe di maggioranza dei suoi 'K' vicini più prossimi nel training set.
  * Caso con Feature Categoriche: Per i dati categorici, una metrica di distanza comune è il numero di feature con valori diversi. Dato il punto di test t1 = [big, heavy, fast] e un set di dati di addestramento, calcoliamo le distanze:
    * dist(t1, x1=[big, light, fast]) = 1
    * dist(t1, x2=[big, heavy, fast]) = 0
    * dist(t1, x3=[small, heavy, slow]) = 2
    * dist(t1, x4=[big, heavy, slow]) = 1
    * dist(t1, x5=[small, light, slow]) = 3
  * Con K=3, i tre vicini più prossimi sono x2 (distanza 0, Classe B), x1 (distanza 1, Classe A) e x4 (distanza 1, Classe A). Poiché ci sono due voti per la Classe A e uno per la Classe B, la classe predetta per t1 tramite votazione di maggioranza è A.
  * Caso con Feature Numeriche e Probabilità: Con dati numerici, si usano distanze come la L1 (Manhattan). È anche possibile calcolare le probabilità di appartenenza a una classe pesando i voti dei vicini con l'inverso della loro distanza. Data la formula P(Y=c|x) ∝ Σ (1/distanza) per i vicini appartenenti alla classe c. Consideriamo il punto di test A=(0,9). I suoi 3 vicini più prossimi sono: il cross a (2,8) con distanza L1 |0-2|+|9-8|=3, il cross a (2,1) con distanza L1 |0-2|+|9-1|=10, e il circle a (5,3) con distanza L1 |0-5|+|9-3|=11.
    * Somma pesi cross: 1/3 + 1/10 = 13/30
    * Somma pesi circle: 1/11
    * Normalizzando, le probabilità sono: P(cross|A) = (13/30) / (13/30 + 1/11) = 143/173 e P(circle|A) = (1/11) / (13/30 + 1/11) = 30/173.
  * Consideriamo il punto di test B=(2,0). I suoi 3 vicini più prossimi sono: un punto cross a distanza 1, un punto circle a distanza 4 e un punto cross a distanza 5.
    * Somma pesi cross: 1/1 + 1/5 = 6/5
    * Somma pesi circle: 1/4
    * Normalizzando: P(cross|B) = (6/5) / (6/5 + 1/4) = 24/29 e P(circle|B) = (1/4) / (6/5 + 1/4) = 5/29.
2. Analisi dei Criteri di Split negli Alberi Decisionali Gli alberi decisionali costruiscono una serie di regole di "split" per suddividere i dati. L'Indice di Gini è una metrica comune per valutare la qualità di uno split, misurando l'impurità di un nodo.
  * Indice di Gini di un Nodo: Gini(t) = 1 - Σ[p(j|t)]², dove p(j|t) è la frazione di campioni della classe j nel nodo t. Un Gini di 0 indica un nodo puro (tutti i campioni appartengono alla stessa classe).
  * Indice di Gini di uno Split: Gini_split = Σ(ni/n) * Gini(i), dove ni è il numero di campioni nel nodo figlio i e n è il totale dei campioni nel nodo padre.
3. Dato uno split x con tre figli a, b, c, calcoliamo il Gini:
  * Gini del nodo a: Il nodo a contiene 80 campioni di C1, 0 di C2 e 20 di C3 (totale 100). Gini(a) = 1 - [(80/100)² + (0/100)² + (20/100)²] = 1 - [0.8² + 0² + 0.2²] = 1 - (0.64 + 0.04) = 0.32
  * Gini dello split x: Calcoliamo prima il Gini per gli altri nodi (Gini(b) = 0.56, Gini(c) = 0). Il numero di campioni è na=100, nb=50, nc=50, per un totale n=200. Gini_split(x) = (100/200)*Gini(a) + (50/200)*Gini(b) + (50/200)*Gini(c) Gini_split(x) = 0.5 * 0.32 + 0.25 * 0.56 + 0.25 * 0 = 0.16 + 0.14 = 0.30
4. Interpretazione delle Metriche da una Matrice di Confusione La matrice di confusione è uno strumento essenziale per valutare un classificatore. Da essa derivano metriche chiave come Precision e Recall.
  * Precision(C): Frazione di predizioni corrette tra i campioni predetti con la classe C. TP / (TP + FP). Misura la precisione delle previsioni positive.
  * Recall(C): Frazione di predizioni corrette tra i campioni con classe reale C. TP / (TP + FN). Misura la capacità del modello di identificare tutti i campioni positivi.
5. Confrontiamo i modelli M1 e M2 specificamente per la classe 'b':
  * Modello M1:
    * Precision(b): TP=50 (predetti 'b' correttamente). FP=10+20+10=40 (predetti 'b' erroneamente). Precision(b) = 50 / (50 + 40) = 50 / 90 ≈ 0.556
    * Recall(b): TP=50 (reali 'b' predetti correttamente). FN=0+5+5=10 (reali 'b' predetti erroneamente). Recall(b) = 50 / (50 + 10) = 50 / 60 ≈ 0.833
  * Modello M2:
    * Precision(b): TP=40. FP=10+5+5=20. Precision(b) = 40 / (40 + 20) = 40 / 60 ≈ 0.667
    * Recall(b): TP=40. FN=5+5+10=20. Recall(b) = 40 / (40 + 20) = 40 / 60 ≈ 0.667
6. Confrontando i risultati, l'affermazione corretta è: M1 ha una recall più alta di M2 e una precision più bassa.

3.2. Valutazione di Modelli di Regressione

1. Analisi dei Residui della Regressione Lineare Il residuo di un punto è la differenza tra il valore osservato (y_i) e il valore predetto dal modello (ŷ_i): r_i = y_i - ŷ_i. L'analisi del grafico dei residui è fondamentale per diagnosticare la bontà di un modello di regressione. Per un modello lineare ben adattato, i residui dovrebbero essere distribuiti casualmente attorno allo zero, senza pattern o trend evidenti. Nel grafico "Data/model plot", i dati mostrano un trend crescente ma la varianza attorno alla linea del modello appare costante. Di conseguenza, il grafico dei residui corretto dovrebbe apparire come una nuvola di punti centrata su zero, senza alcuna forma riconoscibile (come una parabola o un imbuto), corrispondente alla figura C.
2. Distinzione tra Regressione Ridge e Lasso Ridge e Lasso sono due tecniche di regolarizzazione utilizzate per prevenire l'overfitting nei modelli lineari, aggiungendo una penalità alla funzione di costo.
  * Ridge (L2): Minimizza Σ(y_i - ŷ_i)² + α * Σ(w_j)². La penalità L2 (Σ(w_j)²) tende a ridurre la magnitudine di tutti i coefficienti, ma raramente li azzera.
  * Lasso (L1): Minimizza Σ(y_i - ŷ_i)² + α * Σ|w_j|. La penalità L1 (Σ|w_j|) può forzare alcuni coefficienti a diventare esattamente zero, eseguendo di fatto una selezione delle feature e producendo modelli più sparsi (interpretabili).
3. Analizzando i quattro grafici dei coefficienti:
  * Un α più alto implica una maggiore regolarizzazione (coefficienti più piccoli o più zeri).
  * I grafici C e D sono sparsi (molti coefficienti sono zero), quindi corrispondono a Lasso. Il grafico D è più sparso di C, quindi D è Lasso con α=10 e C è Lasso con α=0.1.
  * I grafici A e B non sono sparsi, quindi corrispondono a Ridge. I coefficienti in B sono generalmente più piccoli di quelli in A, indicando una regolarizzazione più forte. Quindi, B è Ridge con α=10 e A è Ridge con α=0.1.
4. Calcolo dell'Errore Quadratico Medio (MSE) L'MSE è una delle metriche più comuni per valutare un modello di regressione. Calcola la media degli errori al quadrato.
  * Formula: MSE = (1/n) * Σ(ŷ_i - y_i)²
5. Applichiamo il modello ŷ = x1 + x2 ai 9 punti dati forniti nelle figure.
  * Predizioni (ŷ): a=6, b=6, c=6, d=4, e=4, f=6, g=2, h=4, i=2
  * Valori reali (y): a=4, b=4, c=4, d=3, e=3, f=6, g=1, h=2, i=2
  * Errori al quadrato (ŷ_i - y_i)²: a=4, b=4, c=4, d=1, e=1, f=0, g=1, h=4, i=0
  * Somma degli errori al quadrato: 4+4+4+1+1+0+1+4+0 = 19
  * MSE: 19 / 9 ≈ 2.11

3.3. Valutazione dei Modelli di Clustering

1. Esecuzione del Clustering Gerarchico Agglomerativo Questo algoritmo costruisce una gerarchia di cluster in modo agglomerativo (bottom-up).
  * Con Single Linkage (MIN): La similarità tra due cluster è definita dalla massima similarità tra due punti qualsiasi dei due cluster. Usando la matrice di similarità fornita:
    1. Inizialmente: 5 cluster {a}, {b}, {c}, {d}, {e}.
    2. La similarità massima è sim(c,e) = 0.9. Si uniscono {c} e {e}. Cluster: {a}, {b}, {d}, {c,e}.
    3. Si ricalcolano le similarità tra i cluster rimanenti. La massima è sim(a,d) = 0.8. Si uniscono {a} e {d}.
      * Stato con K=3: {a,d}, {b}, {c,e}
    4. La similarità massima tra i cluster rimanenti è sim({a,d}, {c,e}) = max(sim(a,c), sim(a,e), sim(d,c), sim(d,e)) = 0.5. Si uniscono.
      * Stato con K=2: {a,c,d,e}, {b}
  * Con Complete Linkage (MAX): La distanza tra due cluster è definita dalla massima distanza tra due punti qualsiasi dei due cluster. Si uniscono i cluster con la minima distanza. Usando la matrice di distanza:
    * a | b | c | d | e
    * La distanza minima tra coppie di punti è dist(c,d) = 1. Si uniscono {c} e {d}. a | b | {c d} | e
    * Si ricalcolano le distanze tra i cluster. La minima è dist(a,b) = 4. Si uniscono {a} e {b}. {a b} | {c d} | e
    * La distanza minima successiva è dist({a,b}, e) = max(dist(a,e), dist(b,e)) = max(23,13) = 13. Si uniscono {a,b} e {e}. Si noti che la definizione è la massima distanza, quindi il calcolo corretto è dist({a,b}, e) = 23. In questo caso dist(e,{c,d}) = max(25,5) = 25. La distanza minima è 13 tra b ed e. Quindi uniamo {b} con {e}. Riesaminando il problema, la distanza minima tra le coppie rimanenti (a,b)=4, (a,e)=23, (b,e)=13, (a,{c,d})=max(24,2)=24, (b,{c,d})=max(17,22)=22, (e,{c,d})=max(25,5)=25. La minima è 4. Si uniscono {a} e {b}. {a b} | {c d} | e
    * Si ricalcolano le distanze: dist(e, {a,b})=max(23,13)=23, dist(e, {c,d})=max(25,5)=25, dist({a,b}, {c,d})=max(24,2,17,22)=24. La minima è 23. Si uniscono {a,b} e {e}. {a b e} | {c d}
    * Infine si uniscono i due cluster rimanenti. {a b c d e}
2. Calcolo dello Score di Silhouette Lo score di Silhouette misura quanto un punto sia simile al proprio cluster rispetto agli altri cluster. Varia da -1 a +1.
  * Formula: silh(i) = (b(i) - a(i)) / max(a(i), b(i))
    * a(i): dissimilarità media di i con tutti gli altri punti nello stesso cluster.
    * b(i): la più bassa dissimilarità media di i verso qualsiasi altro cluster.
3. Calcoliamo lo score per il punto i (coordinate (4,3)) del cluster dei quadrati, usando la distanza di Manhattan.
  * Calcolo di a(i) (intra-cluster): Distanza media da i agli altri 5 punti del cluster dei quadrati. a(i) = (dist(i,c) + dist(i,e) + dist(i,f) + dist(i,g) + dist(i,l)) / 5 = (3 + 1 + 2 + 3 + 1) / 5 = 10 / 5 = 2.0
  * Calcolo di b(i) (inter-cluster):
    * Distanza media dal cluster dei cerchi: (dist(i,a) + dist(i,b) + dist(i,d) + dist(i,h)) / 4 = (6 + 6 + 2 + 2) / 4 = 4.0
    * Distanza media dal cluster delle stelle: (dist(i,m) + dist(i,n)) / 2 = (3 + 4) / 2 = 3.5
    * b(i) è il minimo tra questi valori: min(4.0, 3.5) = 3.5
  * Calcolo dello score di Silhouette: silh(i) = (3.5 - 2.0) / max(2.0, 3.5) = 1.5 / 3.5 = 3/7 ≈ 0.428

Le metriche discusse in questa sezione forniscono un quadro quantitativo essenziale per confrontare le performance dei modelli e selezionare in modo informato la soluzione più appropriata per un dato problema.


--------------------------------------------------------------------------------


4. Cheat Sheet: Riferimento Rapido

Questo capitolo funge da riferimento rapido e compatto (cheat sheet) contenente una sintesi delle principali funzioni, comandi e formule trattate in questo manuale. È progettato per fornire un accesso immediato alle informazioni chiave durante le operazioni pratiche di data science, facilitando l'applicazione delle tecniche descritte.

4.1. Sintassi Numpy

* Creazione Array: np.array([...])
* Indicizzazione e Slicing: array[start:stop:step], array[mask], array[::-1]
* Attributi: array.shape
* Operazioni: array.sum(axis=...)

4.2. Metodi Pandas

Metodo	Descrizione
pd.read_csv()	Carica dati da un file CSV in un DataFrame.
df.to_csv()	Salva un DataFrame in un file CSV.
df.loc[]	Seleziona dati per etichetta (lo slicing è inclusivo).
df.iloc[]	Seleziona dati per posizione intera (lo slicing esclude l'indice di fine).
df.isnull()	Rileva valori mancanti (restituisce una maschera booleana).
df.dropna()	Rimuove righe/colonne con valori mancanti.
df.fillna(value)	Sostituisce i valori mancanti con value.
df.groupby()	Raggruppa il DataFrame per una o più colonne per analisi split-apply-combine.
df.pivot_table()	Crea una tabella pivot per riorganizzare e aggregare i dati.
df.rename()	Modifica le etichette degli assi.
df.drop()	Rimuove righe o colonne specificate.

4.3. Classi Scikit-learn

* make_pipeline(): Costruisce una Pipeline concatenando una sequenza di trasformatori e, opzionalmente, un estimatore finale.
* MinMaxScaler(): Scala le feature in un intervallo definito (es. [0, 1]), preservando la forma della distribuzione.
* FunctionTransformer(): Incapsula una funzione Python arbitraria per usarla come passo di una Pipeline Scikit-learn.

4.4. Formule Matematiche Chiave

Metrica/Formula	Definizione
Normalizzazione Z-Score	x' = (x - μ) / σ
Distanza di Manhattan	`d(p, q) = Σ
Indice di Gini (Nodo)	`Gini(t) = 1 - Σ [p(j
Precision	TP / (TP + FP)
Recall	TP / (TP + FN)
Mean Squared Error (MSE)	(1/n) * Σ(ŷ_i - y_i)²
Score di Silhouette	(b - a) / max(a, b)
