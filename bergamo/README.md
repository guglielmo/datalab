# Introduzione

Questa è un'analisi di alcuni [microdati ISTAT sulle spese delle famiglie](https://www.istat.it/it/archivio/180356),
commissionata dall'Università di Bergamo.

In particolare si vogliono estrarre:

* Percentuale di monogenitori in povertà assoluta sul totale dei monogenitori
* Percentuale di famiglie numerose (con 3 e più figli) in povertà assoluta sul totale dello stesso tipo di nucleo.
* Percentuale di famiglie senza figli (solo coppie) in povertà assoluta sullo stesso nucleo
* Percentuale di famiglie unipersonali in povertà assoluta sullo stesso tipo di famiglia.
* La percentuale tra le famiglie monogenitoriali con figli che vivono in una casa di proprietà 
* La percentuale tra le famiglie monogenitoriali con figli che vivono in una casa in affitto
* La percentuale tra le famiglie monogenitoriali con figli che non vanno in vacanza 
* La percentuale tra le famiglie monogenitoriali con figli che andava in vacanza ma non è riuscito ad andare in vacanza


# Preparazione dei dati.

I microdati arrivano con oltre 1200 colonne, è possibile semplificarli, considerando solo quelle significative per il calcolo delle variabili sui nuceli mono-genitoriali. In tal modo l'import con pandas è rapido.

Si considerano solamente i primi 6 set di variabili sui membri del nucleo familiare, dato che nel campione non sono presenti 
record con dati nei set dal 7.mo al 12.mo. 

Inoltre 


```
csvcut -c1-80,190,1221,1226,1227,1241  -t HBS_Microdati_2018.txt > semplificati_2018.csv
```