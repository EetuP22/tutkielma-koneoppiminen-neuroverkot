# tutkielma-koneoppiminen-neuroverkot
ohke-teknologioita kurssin seminaarivaiheen toteutus.

## Tekijä
- Nimi: Eetu Pärnänen
- Kurssi: Ohjelmistokehityksen teknologioita
- Seminaarityö: Koneoppimistutkielma keskittyen neuroverkkoihin


## Johdanto
Koneoppiminen (machine learning) on modernin tietojenkäsittelyn yksi keskeisimmistä osa-alueista, sitä hyyödynnetään useilla eri teknologian osa-alueilla kääntäjä sovelluksista itseajaviin autoihin. Ykinkertaistettuna koneoppimisella tarkoitetaan prosessia, jossa ohjelmistoa koulutetaan datalla tekemään ennustuksia tai luomaan kontenttia yms. Tätä ohjelmistoa kutsutaan malliksi (model).

Viime vuosikymmenen aikana erityisen merkittävään rooliin on noussut keinotekoiset neuroverkot. Nämä neuroverkot ovat mahdollistaneet erilaisia läpimurtoja laajalla skaalalla teknologian osa-alueita, kuten luonnollisen kielen käsittelyssä ja autonomisissa järjestelmissä. Neuroverkot ovat perhe 'malleja' jotka pyrkivät jäljittelemään ihmisaivojen tapaa käsitellä tietoa. 

Tämän seminaarityön tavoite on tarjota selkeä katsaus tai vähintään pintaraapaisu siihen, mitä koneoppimisella tarkoitetaan, mitä neuroverkot ovat ja miten ne toimivat, sekä miten niitä voidaan soveltaa. Tavoitteena ei ole akateemisesti laaja tuotos, vaan pienempi kokonaisuus, joka auttaa ymmärtämään perus käsitteet ja rakenteet.


## Koneoppimisen perusteet
Koneoppiminen on tekoälyn osa-alue, jossa järjestelmät/ohjelmistot oppivat tekemään päätöksiä ennusteita tai luomaan datan perusteella ilman, että kaikkia sääntöjä tarvitsee erikseen määritellä ohjelmoimalla. Ideana on, että algoritmi havaitsee datasta toistuvia rakenteita ja oppii mallintamaan niitä, sekä pystyisi tämän pohjalta tekemään yleistyksiä uusiin tilanteisiin.

Koneoppimisen perusajatus voidaan tiivistää karkeasta kolmeen vaiheeseen:

- Data: Algoritmi saa esimerkkejä, kuten erilaisia arvoja, kuvia tai tektstiä.
- Oppiminen: Malli muokkaa sisäisiä parametrejään, jotta sen tekemät ennustukset paremmin vastaisivat opetusdataan.
- Yleistys: Valmis malli pystyy toimimaan myös uudella ja opetusdatan ulkopuolisella aineistolla.

**Oppimisen eri muodot** 
Koneoppiminen voidaan jakaa kolmeen pääluokkaan:

- Ohjattu oppiminen (supervised learning), jonka tavoitteena on ennustaa jotain konkreettista, ja opetusdatasta tiedetään haluttu tulos. Esimerkiksi kuvien luokittelu (onko kuvassa koira vai kissa). 
- Ohjaamaton oppiminen (unsupervised learning), jonka tarkoituksena on löytää rakenteita datasta ilman valmiita vastauksia. Esimerkiksi, erilaisten asiakasryhmien löytäminen datasta. Algoritmi etsii siis samanlaisuuksia.
- Vahvistusoppiminen (reinforcement learning), jossa oppiminen tapahtuu jatkuvassa vuorovaikutuksessa mallin ja ympäristön välillä. Missä malli oppii hiljalleen saamalla 'palkkioita' ja 'rangaistuksia'. Esimerkiksi, pelibotit tai robotiikka.

**Mikä tekee koneoppimisesta tehokasta**
Koneoppimismallit ja erityisesti neuroverkot pystyvät oppimaan monimutkaisia funktioita, joita on äärimmäisen vaikea määritellä käsin. Mallit pystyvät esimerkiksi: Tunnistamaan satojen muuttujien suhteita, yleistämään sotkuisesta ja epätäydellisestä datasta yms. 

## Keinotekoiset neuroverkot
Keinotekoiset neuroverkot (Artificial neural networks, ANN) ovat koneoppimismalleja, jotka on suunniteltu imitoimaan sitä, miten ihmis-aivot käsittelevät tietoa. Siinä missä ihmis-aivot hydöyntävät neuroneita datan prosessoinnissa, keinotekoiset neuroverkot käyttävät keinotekoisia neuroneita datan analysointiin, säännönmukaisuuksien tunnistamiseen, sekä ennustamiseen. Ne kykenevät oppimaan monimutkaisia funktioita datasta ja niiden teho perustuu kerrokselliseen rakenteeseen. Neuroverkot toimivat perustana usealle modernille tekoälyn osa-alueelle, kuten kuvantunnistukselle, kielenmallinnukselle ja ohjausjärjestelmille.

### Neuronin rakenne
Kuten mainittu ANN:ät koostuvat keinotekoisista neuroneista, jotka konseptuaalisesti seuraavat biologisia neuroneita. Jokainen näistä neuroneista ottaa vastaan syötteitä, jotka voivat olla ulkoista dataa, tai muiden neuroneiden ulostuloja, ja jokainen neuroni tuottaa yhden ulostulon. Neuroverkon viimeisten ulostulo neuroneiden ulostulot on tarkoitus suorittaa mallille annettu tehtävä.

Yhden neuronin laskenta koostuu tyyppillisesti seuraavista vaiheista:

- Syötteistä: x1,x2...,xn
- Painot: Jokaisella syötteellä on paino, joka kertoo syötteen merkityksen.
- Painotettu summa.
- Aktivaatio
### Aktivaatiofunktio
Aktivaatiofunktio neuroverkossa on matemaattinen funktio jota sovelletaan neuronin ulostuloon. Se tuo ei-lineaarisuuden neuronin ulostuloon, ilman sitä verkko olisi vain lineaarinen malli, joka ei kykene oppimaan monimutkaisia piirteitä. 

Aktivaatiofunktioita on useita erilaisia, yleisimpiä näistä on:
- Sigmoid: Tuottaa arvon välillä 0-1 ja on hyödyllinen todennäköisyyksien mallintamisessa, mutta altis vanhenemiselle.
- ReLU(Rectified linear unit): Tuottaa arvon 0-∞, eli ulostulo on aina positiivinen arvo. ReLU on yksi yleisimmin käytetyistä aktivaatiofunktioista tehokkuuden vuoksi ja koska se mahdollistaa vastavirta-algoritmin tehostamisen.
- Tanh(hyperbolic activation function): On samankaltainen kuin sigmoid funktio, mutta palauttaa arvon -1 - 1. Sitä käytetään useiten piilokerroksissa, kun tarvitaan laajempaa ulostulo skaalaa.
- Softmax: Kääntää raa'an ulostulon todennäköisyyksiin, joita hyödynnetään moni luokkaisten tunnistus tehtävien parissa.


### Verkkoarkkitehtuuri
Neuroverkot voidaan rakentaa erilaisiksi arkkitehtuureiksi riippuen siitä, millaista dataa käsitellään ja mitä ongelmaa ratkaistaan. Näitä arkkitehtuureja ovat muun muassa:

- Feedforward neuroverkot: Nämä ovat yksinkertaisimpia neuroverkkoja, tässä verkossa data kulkee yhteen suuntaan syötekerroksesta ulostulokerrokseen, käyden yhden tai jokaisen piilokerroksen. Data ei tässä arkkitehtuurissa koskaan palaa aikaisemmalle kerrokselle, se ei myöskään hyödynnä vastavirta-algoritmia ja sen pää-käyttötarkoitus on alkeellisissa tunnistus ja regressio tehtävissä.

- Kovoluutioneuroverkot: On suunniteltu prosessoimaan etenkin dataa kuten kuvia. Se sisältää konvoluutiokerroksia jotka lisäävät filtterin ja oppivat havaitsemaan tärkeitä piirteitä datasta, kuten reunoja ja tekstuureja. Tämä tekee konvoluutioverkoista erityisen tehokkaita visuaalisissa tehtävissä.

- Toistuvat neuroverkot (RNN, recurrent neural networks): On suunniteltu jono ja aikajana datan, kuten tekstin käsittelyyn. Toisin kuin muissa arkkitehtuureissa toistuvilla neuroverkoilla on feedback looppeja jotka sallivat tiedonsiirron takaisin aikaisempiin kerroksiin antaen verkolle 'muistin'. Tämä toiminto auttaa toistuvia verkkoja tekemään ennustuksia perustuen kontekstiin, jonka aikaisempi data on tarjonnut. Tästä johtuen RNN:iä hyödynnetään erityisesti puheentunnistus ja kielimallinnus tehtävissä.

Nämä kategoriat ovat vain osa erilaisita neuroverkkoarkkitehtuureista, mutta ovat mahdollisesti eniten käytettyjä nykypäivänä. 
## Oppimisprosessi

### Kustannusfunktio

### Gradientti ja optimointi

### Backpropagation/vastavirta-algoritmi


## Testiosuus

### Toteutusympäristö

### Koodirunko

### Tulokset


## Pohdinta

## Yhteenveto

## Lähteet