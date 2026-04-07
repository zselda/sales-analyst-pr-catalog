# Mizan Satış Analisti Çoklu Ajan Yapısı - Kantitatif Analist (Quant Analyst) Hesaplama Kılavuzu

Bu belge, **Quant Analyst** modülünde yer alan tüm finansal hesaplamaların algoritmalarını ve Türk bankacılık sektöründeki karşılıklarını detaylandırmaktadır. Model, hesaplamaları doğrudan işletmenin sağladığı Mizan (Trial Balance) verisindeki Türkiye Tek Düzen Hesap Planı (TDHP) kodlarına dayandırır.

## 1. Kârlılık ve FAVÖK (Profitability & EBITDA Proxy)

Kantitatif analist kârlılık metriklerini hem dönem hareketleri hem de kapanış bakiyelerini göz önünde bulundurarak aşağıdaki gibi hesaplar:

*   **Brüt Kâr Marjı (Gross Margin)**
    *   *Formül:* `(Yurtiçi/Yurtdışı Satışlar [600] - Satılan Malın Maliyeti [620]) / Satışlar [600]`
    *   *Bankacılık Karşılığı:* Firmanın sadece mal alım satımından veya üretiminden elde ettiği ana operasyonel marjı gösterir.
*   **Faaliyet Kâr Marjı (Operating Margin)**
    *   *Formül:* `(Brüt Kâr - Faaliyet Giderleri [630 + 631 + 632]) / Satışlar [600]`
    *   *Bankacılık Karşılığı:* Firmanın pazarlama, genel yönetim ve araştırma geliştirme giderleri düşüldükten sonraki net faaliyet performansıdır. Bankalar kredi tahsisinde bu metrik üzerinde durur.
*   **FAVÖK / EBITDA Yaklaşımı (EBITDA Proxy)**
    *   *Formül:* `Faaliyet Kârı + Amortismanlar [257 ve 268 Net Dönem Hareketi]`
    *   *Bankacılık Karşılığı:* Firmanın nakit yaratma kapasitesi. İşletmenin banka kredisi ana para ve faiz ödemelerini (Debt Service Coverage) karşılayıp karşılayamayacağı doğrudan bu nakit akış metniği ile incelenir.

## 2. Likidite ve İşletme Sermayesi (Liquidity & Working Capital)

İşletmenin kısa vadeli yükümlülüklerini, elindeki likiditeyle ne kadar sürede ödeyebileceğini ölçer. Kredi risk (tahsis) ekipleri kısa vadeli (rotatif/BCH) kredi limitlerini belirlerken kullanır:

*   **Cari Oran (Current Ratio)**
    *   *Formül:* `Dönen Varlıklar [1xx] / Kısa Vadeli Yabancı Kaynaklar [3xx]`
    *   *Bankacılık Karşılığı:* İstenen minimum değer genelde 1.0 - 1.5 bandıdır. Firmanın mevcut varlıklarının kısa vadeli borçlarını kapatma kabiliyetini ölçer.
*   **Asit Test Oranı (Quick Ratio)**
    *   *Formül:* `(Dönen Varlıklar [1xx] - Stoklar [150, 151, 152, 153]) / Kısa Vadeli Yabancı Kaynaklar [3xx]`
    *   *Bankacılık Karşılığı:* Stoklar her zaman anında nakde çevrilemeyebileceğinden, stoksuz (en likit) varlıklarla borç kapama oranını gösterir.

## 3. Nakit Döngüsü ve İşletme Sermayesi Süreleri (Cash Conversion Cycle)

Bankacılıkta ticari kredi kullanım hacmini hesaplayan en kritik algoritmadır. Gün sayıları, müşterinin fatura vadelerini tanımlar:

*   **Alacak Tahsilat Süresi (Collection Period)**
    *   *Formül:* `(Ticari Alacaklar [120 + 121] / Satışlar [600]) * Dönem Gün Sayısı (örn; Yıllık 360)`
*   **Stok Bekleme Süresi (Inventory Period)**
    *   *Formül:* `(Stoklar [15x] / SMM [620]) * Dönem Gün Sayısı`
*   **Tedariçiye Ödeme Süresi (Payment Period)**
    *   *Formül:* `(Ticari Borçlar [320 + 321] / SMM [620]) * Dönem Gün Sayısı`
*   **Net Nakit Döngüsü (Cash Conversion Cycle)**
    *   *Formül:* `Tahsilat Süresi + Stok Süresi - Ödeme Süresi`
    *   *Bankacılık Karşılığı:* Bu döngünün uzun olması, firmanın nakit açığına düştüğünü; yani işletme sermayesi veya BCH (Borçlu Cari Hesap) gibi banka finansmanına olan net ihtiyacını gösterir. Satış stratejisti, döngüsü uzunolan firmalara finansman çözümü teklif eder.

## 4. Kaldıraç, Bağımlılık ve Risk Metrikleri (Leverage, Dependency & Risks)

*   **Borç / Özkaynak Oranı (Debt to Equity)**
    *   *Formül:* `(Kısa Vadeli + Uzun Vadeli Borçlar) / Özkaynaklar [500 + 570 vb.]`
    *   *Bankacılık Karşılığı:* İşletmenin faaliyetlerini kendi öz sermayesi yerine banka borçlanması ile fonlama derecesidir.
*   **Bankalara Borçlanma Oranı (Bank Debt Ratio)**
    *   *Formül:* `(KV Banka Kredileri [300] + UV Banka Kredileri [400] + Diğer Mali Borçlar Kredi Kartları vb. [309]) / Toplam Borçlar`
*   **Finansman Gider Oranı (Financial Expense Ratio)**
    *   *Formül:* `Finansman Giderleri [780] / Satışlar [600]`
    *   *Bankacılık Karşılığı:* Şirketin elde ettiği gelirlerin yüzde kaçını banka faizlerine ve kredilerin faiz servislerine harcadığını (kredi yükünü) hesaplar.
*   **Çek Risk Oranı (Check Risk Ratio) - YENİ**
    *   *Formül:* `Verilen Çekler [103] / Banka Mevduat Hacmi [102]`
    *   *Bankacılık Karşılığı:* "Duality Check" mekanizması. Şirketin dışarıya kestiği çeklerin, hesaplarındaki (102) paradan yüksek olması nakit sıkışıklığı ya da "Karşılıksız Çek Riskine" (Bouncing Check) işaret edebilir.
*   **Grup İçi Borçlanma / Para Kaçırma Riski (Insider Lending Ratio) - YENİ**
    *   *Formül:* `Ortaklardan Alacaklar [131] / Aktif Toplamı`
    *   *Bankacılık Karşılığı:* "Cash Trapping" check. Şirket varlıklarının %5'inden fazlasının patronun/ortakların şahsi harcamalarına ("Ortaklardan Alacaklar" kasasına) çekilip çekilmediğini kontrol eden önemli bir finansal sağlık detektörüdür.
*   **POS Komisyon Oranı (POS Commission Ratio)**
    *   *Formül:* `POS Komisyon Giderleri [780.01] / Satışlar [600]`
    *   *Bankacılık Karşılığı:* Tahsilatların ne kadarının sanal ya da fiziksel kredi kartı (POS) kesintilerine gittiğini bulur, nakit yönetim ürünleri satışı için fırsat yaratır.

## 5. Rakip Banka Dağılımı ve Yeniden Finansman (Competitor Bank Analysis)

Model sadece toplam 102/300 hesaplarına bakıp bırakmaz, *102.01, 102.02, 300.01 Akbank, 300.02 Garanti* vb. gibi alt kırılımlara inerek diğer bankalardaki kasa ve kredi kullanım hacimlerini listeler.
Bu veri sayesinde ajanlar ("Strategist" Ajanı):
- *"Firmanın x Bankasında kredisini görüyoruz, o bankadan borcunu biz yapılandırarak vadesini uzatabiliriz. (Refinancing Buyout)"*
- *"Firmanın cüzdan payında mevduatı (102) X bankasında ama nakit çıkışlarını bizden yapıyor, çapraz satış ile o mevduatı çekebiliriz"* gibi senaryolar tasarlar.
