# Ödev 2 — MLE ile Akıllı Şehir Planlaması

YZM212 Makine Öğrenmesi dersi 2. Laboratuvar Ödevi

---

## Problem Tanımı

Şehrin en yoğun caddesinden bir dakikada geçen araç sayısı Poisson dağılımına
uyuyor. Maximum Likelihood Estimation (MLE) ile en iyi λ parametresi tahmin
edilmekte; outlier etkisi analiz edilmektedir.

---

## Veri

PDF'de verilen sabit trafik verisi:

```
traffic_data = [12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15]
```

n = 14 gözlem, ortalama = 12.14

---

## Yöntem

- **Bölüm 1:** Poisson log-likelihood'inin analitik türetimi →  
  λ̂_MLE = (Σk_i) / n = veri ortalaması (kanıtlandı)
- **Bölüm 2:** `scipy.optimize.minimize` ile sayısal NLL minimizasyonu
- **Bölüm 3:** Poisson PMF + gerçek veri histogramı karşılaştırması
- **Bölüm 4:** 200 araçlık outlier eklenerek MLE hassasiyeti analizi

Kod: [src/mle_traffic.ipynb](src/mle_traffic.ipynb)

---

## Sonuçlar

| | Temiz Veri | Outlier Ekli (200) |
|---|:---:|:---:|
| λ_MLE (scipy) | 12.1429 | 24.6667 |
| λ_MLE (ortalama) | 12.1429 | 24.6667 |
| Sapma | — | +%103 |

---

## Yorum / Tartışma

**Model Uyumu:** Poisson PMF, 8–16 araç aralığında yoğunlaşan gerçek veriyle
iyi örtüşmektedir.

**Outlier Etkisi:** MLE = ortalama olduğundan tek bir `200`'lük hatalı gözlem
λ'yı %103 şişiriyor. Bu, belediyenin gereksiz yol genişletme kararı almasına
yol açabilir. Çözüm: IQR/Z-score ile veri temizliği veya robust tahmin.

---

## Grafikler

| Grafik | Açıklama |
|--------|----------|
| [pmf_histogram.png](report/pmf_histogram.png) | PMF vs gerçek veri histogramı |
| [outlier_analizi.png](report/outlier_analizi.png) | Outlier etkisi — λ kayması |

Tüm grafikler ve yorumlar: [report/Odev2_Rapor.pdf](report/Odev2_Rapor.pdf)
