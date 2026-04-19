# Ödev 1 — HMM ile İzole Kelime Tanıma

YZM212 Makine Öğrenmesi dersi 1. Laboratuvar Ödevi

---

## Problem Tanımı

Bu projede Gizli Markov Modeli (HMM) kullanılarak izole kelime tanıma sistemi tasarlanmıştır.
"EV" ve "OKUL" kelimeleri için ayrı HMM modelleri oluşturulmuş, gelen ses gözlem dizisi
hangi modelde daha yüksek Log-Likelihood veriyorsa o kelime olarak sınıflandırılmıştır.

---

## Veri

Her kelime için gözlem dizileri iki sembolden oluşur: `High` (0) ve `Low` (1).
Bu semboller sesin frekans karakteristiğini temsil eder.

**Test gözlem dizileri:**
- `[High, Low]` → EV beklenir
- `[Low, Low, Low, Low]` → OKUL beklenir
- `[High, High]` → EV beklenir
- `[Low, Low]` → OKUL beklenir

---

## Yöntem

Her kelime için `hmmlearn` kütüphanesiyle `CategoricalHMM` modeli tanımlanmıştır.
Modeller başlangıç, geçiş ve emisyon olasılıklarıyla elle yapılandırılmıştır.
Sınıflandırma için `model.score()` fonksiyonu ile Log-Likelihood hesaplanmış,
yüksek skoru veren model kazanan ilan edilmiştir.

Kod: [src/recognizer.py](src/recognizer.py)

---

## Sonuçlar

| Gözlem Dizisi | EV Skoru | OKUL Skoru | Tahmin |
|---------------|----------|------------|--------|
| [High, Low] | -0.9729 | -1.2730 | **EV** ✓ |
| [Low, Low, Low, Low] | -2.4702 | -1.4585 | **OKUL** ✓ |
| [High, High] | -1.1332 | -2.1203 | **EV** ✓ |
| [Low, Low] | -1.8202 | -0.8675 | **OKUL** ✓ |

4/4 test doğru sınıflandırıldı.

---

## Yorum / Tartışma

Model, High ağırlıklı dizileri EV, Low ağırlıklı dizileri OKUL olarak başarıyla
sınıflandırmaktadır. Bu sonuç, elle tasarlanan HMM parametrelerinin kelimeler
arasındaki frekans farklılığını yeterince yansıttığını göstermektedir.

Gerçek bir sistemde gürültü emisyon olasılıklarını bozacağından model performansı
düşebilir. Binlerce kelime içeren büyük ölçekli sistemlerde ise Viterbi + HMM yerine
Transformer tabanlı modeller (Whisper, wav2vec) tercih edilmektedir.
