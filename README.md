# Prediction Pipeline for TJK Race Analytics

## 1. Hızlı Başlangıç

### Proje Amacı

TJK yarış verilerini içeren CSV dosyalarını doğrulayan, birleştiren ve zenginleştiren bu prediction pipeline, yalnızca harici olarak sağlanan program ve opsiyonel idman kayıtlarını okuyarak kapsamlı özellik mühendisliği uygular. Ardından CPU dostu makine öğrenimi modelleri ile her yarışa dair kazanç/place olasılıklarını, beklenen bitiş sıralarını ve sürelerini tahmin edip standartlaştırılmış JSON çıktıları ile kısa insan-okur raporları üretir.

Proje bulut ortamlarında GPU gerektirmeden çalışacak şekilde tasarlanmıştır; yerel RTX 4060 (8 GB) kartı için FP16/TensorRT hazırlıkları kod tabanında bayraklar ve yorumlar halinde mevcuttur ancak varsayılan olarak kapalıdır.

### 5 Dakikada Çalıştır

1. Kurulum adımlarını tamamlayın (Python 3.x ortamı + bağımlılıklar).
2. `program.csv` ve gerekirse `workouts.csv` dosyalarını proje köküne yerleştirin.
3. Eğitim için aşağıdaki komutu çalıştırın:
   ```bash
   python -m cli.train --program program.csv --workouts workouts.csv --val-date "2025-09-20" --cpu-only
   ```
4. Tahmin ve rapor oluşturmak için şu komutu yürütün:
   ```bash
   python -m cli.predict --program program.csv --workouts workouts.csv --out predictions.json --report report.md --cpu-only
   ```

## 2. İçindekiler

- [Prediction Pipeline for TJK Race Analytics](#prediction-pipeline-for-tjk-race-analytics)
  - [1. Hızlı Başlangıç](#1-h%C4%B1zl%C4%B1-ba%C5%9Flang%C4%B1%C3%A7)
    - [Proje Amacı](#proje-amac%C4%B1)
    - [5 Dakikada Çalıştır](#5-dakikada-%C3%A7al%C4%B1%C5%9Ft%C4%B1r)
  - [3. Mimari ve Klasör Yapısı](#3-mimari-ve-klas%C3%B6r-yap%C4%B1s%C4%B1)
  - [4. Kurulum](#4-kurulum)
  - [5. Veri Girdileri (CSV Şeması ve Kurallar)](#5-veri-girdileri-csv-%C5%9Femasi-ve-kurallar)
  - [6. Temizleme ve Doğrulama Kuralları](#6-temizleme-ve-do%C4%9Frulama-kurallar%C4%B1)
  - [7. Özellik Mühendisliği](#7-%C3%B6zellik-m%C3%BChendisli%C4%9Fi)
  - [8. Modeller ve Ensemble](#8-modeller-ve-ensemble)
  - [9. Eğitim & Değerlendirme](#9-e%C4%9Fitim--de%C4%9Ferlendirme)
  - [10. Çıktılar](#10-%C3%A7%C4%B1kt%C4%B1lar)
  - [11. CLI Kullanımı](#11-cli-kullan%C4%B1m%C4%B1)
  - [12. Sorun Giderme](#12-sorun-giderme)
  - [13. Veri Gizliliği ve Log Politikası](#13-veri-gizlili%C4%9Fi-ve-log-politikas%C4%B1)
  - [14. Sürümleme & Değişiklik Kaydı](#14-s%C3%BCr%C3%BCmleme--de%C4%9Fi%C5%9Fiklik-kayd%C4%B1)
  - [15. Katkı Rehberi](#15-katk%C4%B1-rehberi)
  - [16. SSS](#16-sss)
  - [17. Sözlük](#17-s%C3%B6zl%C3%BCk)
  - [18. Lisans](#18-lisans)

## 3. Mimari ve Klasör Yapısı

```
src/
  dataio/{read_program.py, read_workouts.py, merge.py}
  features/{parsers.py, set_features.py, market_features.py, gate_context.py}
  models/{xgb.py, lgbm.py, catb.py, set_mlp.py, ensemble.py, calibrate.py}
  eval/{metrics.py, backtest.py}
  cli/{synth.py, train.py, predict.py, report.py}
artifacts/   # eğitim çıktı modelleri
```

- `src/dataio/read_program.py`: Program CSV dosyalarını okur, normalleştirir ve doğrulama yapar.
- `src/dataio/read_workouts.py`: Workout CSV’lerini işler, skor eşiklerine göre filtreler.
- `src/dataio/merge.py`: Program ve workout setlerini yarış ve at bazında birleştirir.
- `src/features/parsers.py`: Ham alanları tarih, saat, mesafe, dereceler gibi standart formatlara dönüştürür.
- `src/features/set_features.py`: Yarış içi field-wise istatistiklerini hesaplar.
- `src/features/market_features.py`: Overround, implied probability, market share ve MDI sinyallerini üretir.
- `src/features/gate_context.py`: Gate/pist bağlam anahtarlarını ve rank yüzdeliklerini çıkarır.
- `src/models/xgb.py`: XGBoost tabanlı tahmin modellerini CPU parametreleriyle hazırlar.
- `src/models/lgbm.py`: LightGBM konfigürasyonlarını ve eğitim yordamlarını içerir.
- `src/models/catb.py`: CatBoost modelleri için CPU uyumlu pipeline sağlar.
- `src/models/set_mlp.py`: Set tabanlı MLP yapısını PyTorch üzerinde CPU modunda tanımlar.
- `src/models/ensemble.py`: Bağlamsal gating kullanan meta-ensemble’ı uygular.
- `src/models/calibrate.py`: Temperature scaling ve isotonic kalibrasyon modüllerini barındırır.
- `src/eval/metrics.py`: AUC, Brier, LogLoss vb. metrik hesaplayıcılarını içerir.
- `src/eval/backtest.py`: Zaman bazlı walk-forward geri test döngülerini yönetir.
- `src/cli/synth.py`: Sentetik CSV üretim aracı (gerçek veri yoksa).
- `src/cli/train.py`: Eğitim ve kalibrasyon akışını çalıştırır.
- `src/cli/predict.py`: Tahmin, kalibrasyon uygulaması ve çıktı üretiminden sorumludur.
- `src/cli/report.py`: JSON tahminlerinden kısa insan-okur raporu üretir.
- `artifacts/`: CPU’da eğitilmiş modellerin ve kalibrasyon parametrelerinin depolandığı dizin.

## 4. Kurulum

1. **Sistem Gereksinimleri**
   - Python 3.9 veya üstü
   - 8–16 GB RAM
   - Depolama: ~2 GB boş alan
   - GPU gerekmez; internet bağlantısı olmadan çalışabilir.

2. **Sanal Ortam Önerileri**
   - `venv`:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
     ```
   - `conda`:
     ```bash
     conda create -n tjk-predict python=3.10
     conda activate tjk-predict
     ```

3. **Bağımlılık Kurulumu**
   - Proje kökünde:
     ```bash
     pip install -e .
     ```
     veya sadece gereksinimleri kurmak isterseniz:
     ```bash
     pip install -r requirements.txt
     ```

4. **Platform Notları**
   - **Windows**: PowerShell’de komutları çalıştırın, `python` yerine `py` kullanılabilir.
   - **macOS/Linux**: Terminal komutları aynıdır; `python3`/`pip3` kullanımı önerilir.
   - Hiçbir adımda internetten veri çekilmez; CSV dosyaları dışarıdan sağlanır.

## 5. Veri Girdileri (CSV Şeması ve Kurallar)

### Program CSV (Zorunlu)

| Sütun Adı | Açıklama |
| --- | --- |
| Program Başlık | Yarış programının resmi başlığı. |
| Tarih | `gg/aa/yyyy` formatında yarış tarihi. |
| Gün | Haftanın günü bilgisi. |
| Hipodrom Başlık | Hipodrom adı için vitrin başlık alanı. |
| Hipodrom | Standart hipodrom adı. |
| Yarış Günü Numarası | Gün içindeki program sırası. |
| Program Şehir ID | Şehir kodu. |
| Çim Durumu | Çim pistinin resmi durum bilgisi. |
| Kum Durumu | Kum pistinin resmi durum bilgisi. |
| Hava Bilgisi | Genel hava raporu özeti. |
| Hava Sıcaklığı | Santigrat sıcaklık değeri. |
| Hava Durumu | Detaylı hava durumu açıklaması. |
| Nem | Nem oranı (%). |
| Koşu ID | Resmi koşu kimliği. |
| Koşu Başlık | Koşu adı. |
| Koşu Numarası | Gün içindeki koşu sırası. |
| Koşu Saati | Resmi çıkış saati (`HH:MM`). |
| Özel Koşu Adı | Özel koşu kategorisi. |
| Koşu Sınıfı | Koşunun sınıf/handikap düzeyi. |
| Baz Sıklet | Koşu için temel sıklet değeri. |
| Mesafe | Yarış mesafesi (metre). |
| Pist Tipi | Koşulacak pist türü (Çim/Kum/Sentetik). |
| En İyi Derece (Tarihçe) | Tarihsel en iyi derece kayıtları. |
| Koşu Koşulları | Koşuya katılım şartları. |
| Muhtemel Linki | Resmi muhtemel tablosu bağlantısı. |
| Koşu PDF | PDF formu bağlantısı. |
| Pist Şeması Görseli | Pist planı görsel URL’si. |
| {Başlık} {Sıra} | Ek kolonlar (dinamik). |
| Program Sırası | Programdaki sıra numarası. |
| At İsmi | Yarışacak atın adı. |
| At Detay Linki | At için detay sayfası bağlantısı. |
| Donanım Kodları | Kullanılan ekipman kodları. |
| Yaş Bilgisi | Atın yaşı (metin). |
| Orijin (Baba - Anne) | Pedigri açıklaması. |
| Baba | Baba adı. |
| Baba Linki | Baba için referans link. |
| Anne | Anne adı. |
| Anne Linki | Anne için referans link. |
| Kısrak Babası | Kısrak baba adı. |
| Kısrak Babası Linki | Kısrak baba linki. |
| Sıklet | Atın sıkleti (kg). |
| Jokey | Jokey adı. |
| Jokey Linki | Jokey referans linki. |
| Sahip | Atın sahibi. |
| Sahip Linki | Sahip linki. |
| Antrenör | Antrenör adı. |
| Antrenör Linki | Antrenör linki. |
| Start No | Kapı numarası. |
| Handikap Puanı | Handikap puanı. |
| Son 6 Yarış | Son altı yarış sonuçları. |
| KGS | Kayıtlı galibiyet serisi vb. |
| s20 | Son 20 performans özeti. |
| En İyi Derece | Atın en iyi derecesi. |
| En İyi Derece Linki | Derece kaynağı linki. |
| En İyi Derece Açıklaması | Derecenin bağlamı. |
| Ganyan | Ganyan oranı. |
| AGF | Altılı ganyan favori yüzdesi. |
| AGF Açıklaması | AGF bağlamı. |
| İdman Linki | İdman sayfası linki. |
| W_Workout_Count | Mevcut workout sayısı. |
| W_Latest_Date | Son workout tarihi. |
| W_Latest_Hip | Son workout hipodromu. |
| W_800m_Best | 800m en iyi derece. |
| W_600m_Best | 600m en iyi derece. |
| W_Match_Score | Workout eşleştirme skoru. |
| W_Match_Method | Eşleştirme yöntemi. |
| W_Data_Quality | Workout verisi kalitesi. |

### Workout CSV (Opsiyonel)

| Sütun Adı | Açıklama |
| --- | --- |
| Tarih | Program tarihinin tekrarlandığı alan. |
| Hipodrom | Workout’un yapıldığı hipodrom. |
| Koşu ID | İlgili koşu kimliği. |
| At İsmi | Workout yapan at adı. |
| At ID | Atın benzersiz kimliği (varsa). |
| Workout Tarih | İdman tarihi. |
| W_Hip | Workout hipodromu. |
| W_Pist | Workout pist türü. |
| W_Type | Çalışma tipi. |
| W_Status | Workout durumu. |
| W_Jokey | Workout sırasında binen jokey. |
| w1200_s | 1200m derecesi (saniye). |
| w1000_s | 1000m derecesi. |
| w800_s | 800m derecesi. |
| w600_s | 600m derecesi. |
| w400_s | 400m derecesi. |
| w200_s | 200m derecesi. |
| matched_name | Eşleşen at adı (normalize). |
| matched_url | Eşleşen kaynak linki. |
| match_method | Eşleştirme yöntemi açıklaması. |
| match_score | Eşleştirme skor yüzdesi. |

### Join Anahtarları

- **Yarış**: `(TarihISO, Hipodrom, Koşu ID)`; Koşu ID yoksa `(TarihISO, Hipodrom, Koşu Numarası)`.
- **At**: Yarış anahtarı + `(At İsmi)` ve Start No mevcutsa eklenir.
- **Workout**: `(Tarih, Hipodrom, Koşu ID, At İsmi)` + `match_score >= 85` veya tam isim eşleşmesi.

Örnek: `2025-09-20`, `Ankara`, `R5` koşusu için Start No `3` olan `Yıldırım` atı; workout verisinde aynı tarih/hipodrom/koşu/isim ve `match_score=92` ise merge edilir.

## 6. Temizleme ve Doğrulama Kuralları

Tüm tarih ve saat işlemleri **Europe/Istanbul** zaman dilimini esas alır; CSV’deki `gg/aa/yyyy` tarihleri pipeline içinde opsiyonel bir post-process aşamasında ISO `yyyy-mm-dd` formatına dönüştürülür.

| Kural | Uygulama | Hata Durumu |
| --- | --- | --- |
| Tarih dönüşümü | `gg/aa/yyyy` → ISO `yyyy-mm-dd` | Parse başarısızsa yarış atlanır, `errors` listesine ham değer yazılır. |
| Koşu Saati | `HH:MM` formatı beklenir | Uymayan değerle yarış atlanır, hata kaydı yapılır. |
| Mesafe | Nokta/virgül temizlenir, metreye çevrilir | [800, 3400] dışındaysa yarış atlanır. |
| Pist Tipi | Çim/Kum/Sentetik normalleştirilir | Tanınmazsa null; pist durumu belirlenemezse `null`. |
| En İyi Derece | Süre formatları saniyeye çevrilir | Parse yoksa `null`. |
| AGF | `%28` → 0.28 | Sayısal dönüşüm yoksa `null`. |
| Ganyan | Locale uyumlu float | `>250` ise `null`; `implied_prob = 1/ganyan` yalnız >0 için. |
| Yaş Bilgisi | `3y` → 3 | Parse yoksa `null`. |
| Start No | İnt’e çevrilir | Eksikse `x1, x2...` geçici numara. |
| Donanım Kodları | `has_KG` bayrağı üretilir | `KG` yoksa 0. |

**Errors JSON Örneği**
```json
{"row": 123, "reason": "invalid_date", "value": "32/13/2025"}
```

## 7. Özellik Mühendisliği

- **Field-wise İstatistikler**: `yas, siklet, handikap_puanı, kgs, s20, en_iyi_derece_s, agf_01, implied_prob, start_no` alanları için yarış içi ortalama, std, median, min, max, `rel_z`, `rank_pct`, `delta_med` hesaplanır.
- **Overround & Pazar Sinyali**: Her yarışta `Q = Σ (1/ganyan)`; `p_market = (1/ganyan) / Q`. Ganyan yoksa `null`.
- **MDI**: `agf_01 = AGF/100`; `mdi = sigmoid(10*(agf_01 - p_market))`. `p_market` yoksa `null`.
- **Odds Drift**: Birden çok snapshot varsa `dp60/dp30/dp15/dp5`, `dagf15/dagf30` farkları hesaplanır; yoksa tüm drift sütunları `null` ve raporda “drift: n/a”.
- **Gate/Pist Etkileşimi**: `gate_rank_pct` (kapı yüzdelik sırası), `mesafe_bucket` (`<=1400`, `1400-2000`, `>2000`), `gate_context_key = pist_tipi × pist_durumu × mesafe_bucket × hipodrom`.
- **Genealogy Tokens**: Baba, Anne, Kısrak Babası adları ASCII’ye çevrilip küçük harfe indirilir (`victory_gallop` gibi) ve liste halinde saklanır.
- **Race Context**: `{ field_size, mesafe, pist_tipi, pist_durumu, hipodrom, kosu_sinifi, hava_durumu, median_handikap, median_en_iyi_derece_s, median_agf_01 }`.

Örnek çıktı satırı:

| horse_uid | implied_prob | p_market | mdi | gate_rank_pct | race_context.field_size |
| --- | --- | --- | --- | --- | --- |
| `2025-09-20_Ankara_R5-3-yildirim` | 0.18 | 0.16 | 0.55 | 0.42 | 12 |

## 8. Modeller ve Ensemble

| Model | Varsayılan Hiperparametreler |
| --- | --- |
| XGBoost | `tree_method=hist`, `max_depth=6`, `eta=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `n_estimators=600`. |
| LightGBM | `num_leaves=31`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `learning_rate=0.05`, `n_estimators=800`. |
| CatBoost | `depth=8`, `l2_leaf_reg=3`, `iterations=2000`, `od_type=Iter`. |
| Set-MLP | PyTorch tabanlı, `d_model=128`, `2-3 blok`, `dropout=0.1`, `AdamW lr=3e-4`. |

**Bağlamsal Gated Meta-Learner**
- Girdi: `race_context` vektörü.
- Çıktı: `w_k = softmax(g(context))` ağırlıkları; nihai skor `Σ w_k · p_k`.
- Eğitim: Base modeller validation tahminleri → gate ağı eğitilir.

**Kalibrasyon**
- Temperature scaling ve isotonic regresyon uygulanır; validation’da daha iyi olan yöntem seçilir ve parametreleri JSON’a kaydedilir.

**RTX 4060 Notları**
- PyTorch başlıkları için ONNX→TensorRT FP16 dönüşüm komutları hazır (ör. `--enable-trt` bayrağı), ancak varsayılan olarak `--cpu-only` ile kapalı. AMP (Automatic Mixed Precision) bayrakları `train.py` ve `predict.py` içinde yorum satırlarıyla işaretlidir.

## 9. Eğitim & Değerlendirme

- **Zaman Bazlı Split**: Eğitimde geçmiş tarihler, validasyonda gelecekteki tarihler kullanılır. `--val-date` parametresi ile sınır belirlenir.
- **Walk-Forward Backtest**: `eval/backtest.py` ardışık dönemler için modeli yeniden eğiterek performansı ölçer.
- **Metrikler**: AUC, PR-AUC, Brier Score, LogLoss, NDCG@K, RMSE (race_time), ECE (kalibrasyon).
- **Top-K Lift & Edge**: `edge = win_prob - implied_prob`; yüksek edge değerleri pozitif beklenti sinyali kabul edilir.

## 10. Çıktılar

### JSON Örneği

```json
{
  "race_id": "2025-09-20_Ankara_R5",
  "meta": {
    "hipodrom": "Ankara",
    "tarih": "2025-09-20",
    "kosu_saati": "15:30",
    "kosu_sinifi": "Handikap 16",
    "mesafe": 1600,
    "pist_tipi": "cim",
    "calibration": {"method": "temperature", "param": 1.12}
  },
  "predictions": [
    {
      "horse_id": "2025-09-20_Ankara_R5-3-yildirim",
      "at_ismi": "Yıldırım",
      "start_no": 3,
      "win_prob": 0.24,
      "place_prob": 0.52,
      "expected_finish": 2.1,
      "race_time": 93.4,
      "uncertainty": {"win_std": 0.03, "place_std": 0.04},
      "ganyan": 4.2,
      "implied_prob": 0.238,
      "edge": 0.002,
      "mdi": 0.61,
      "drift_dp15": null,
      "extras": {"has_KG": 1, "gate_rank_pct": 0.42, "gate_context_key": "cim-good-1400-2000-Ankara"}
    }
  ],
  "metrics": {"val_auc": 0.78, "brier": 0.19, "logloss": 0.52, "ndcg@3": 0.63},
  "errors": []
}
```

### Kısa Rapor Örneği

```
Ankara | 2025-09-20 15:30 | Handikap 16 • 1600m • cim
Top 5:
#3 Yıldırım      — Win 24% | Place 52% | ExpFin 2.1 | Ganyan 4.2 | Edge 0.2% | MDI 0.61 | Drift15 n/a
#6 Şimşek        — Win 18% | Place 45% | ExpFin 2.8 | Ganyan 5.8 | Edge -0.5% | MDI 0.47 | Drift15 n/a
#1 GüçlüKız      — Win 15% | Place 39% | ExpFin 3.3 | Ganyan 6.5 | Edge 1.5% | MDI 0.52 | Drift15 n/a
#4 RüzgarınOğlu  — Win 12% | Place 34% | ExpFin 3.7 | Ganyan 8.0 | Edge -0.7% | MDI 0.44 | Drift15 n/a
#2 AkKanat       — Win 10% | Place 28% | ExpFin 4.2 | Ganyan 10.5| Edge -1.0% | MDI 0.39 | Drift15 n/a
Notlar: calibration=temperature(param=1.12), N=12, field-wise norm aktif, overround düzeltmesi aktif, gate_ctx=cim-good-1400-2000-Ankara
```

## 11. CLI Kullanımı

### Sentetik Veri Üretimi
```bash
python -m cli.synth --out program.csv --workouts workouts.csv --n-races 6 --city "Ankara" --date "27/09/2025"
```
- Sadece program CSV isteniyorsa `--workouts` parametresini atlayabilirsiniz.

### Eğitim
```bash
python -m cli.train --program program.csv --workouts workouts.csv --val-date "2025-09-20" --cpu-only
```
- Workout verisi yoksa `--workouts` parametresini kullanmayın.
- Sentetik veri ile test: `--program synthetic_program.csv` vb.

### Tahmin + Rapor
```bash
python -m cli.predict --program today.csv --workouts today_w.csv --out out.json --report out.md --cpu-only
```
- Sadece Program CSV ile tahmin: `python -m cli.predict --program today.csv --out out.json --report out.md --cpu-only`.
- `--cpu-only` bayrağı RTX optimizasyonlarını kapalı tutar; FP16/TensorRT için `--enable-trt` benzeri bayraklar kodda ancak varsayılan `False`.

### Parametre Referansı

| Komut | Parametre | Varsayılan | Açıklama |
| --- | --- | --- | --- |
| `cli.synth` | `--n-races` | 6 | Üretilecek yarış sayısı. |
| `cli.synth` | `--city` | "İstanbul" | Program şehir adı. |
| `cli.train` | `--val-date` | None | Validation sınır tarihi (ISO). |
| `cli.train`/`cli.predict` | `--cpu-only` | `True` | CPU fallback zorlaması. |
| `cli.predict` | `--out` | `predictions.json` | JSON çıktı dosyası. |
| `cli.predict` | `--report` | `report.md` | Rapor dosyası. |

## 12. Sorun Giderme

| Hata | Belirti | Çözüm |
| --- | --- | --- |
| Eksik sütun | CLI uyarı verir | CSV başlıklarının talep edilen şemayla eşleştiğini doğrulayın. |
| Tarih parse hatası | `invalid_date` hatası | Tarih formatını `gg/aa/yyyy` yapın. |
| Mesafe aralığı dışı | `distance_out_of_range` | Mesafeyi 800–3400 m aralığına getirin. |
| Drift verisi yok | Raporda `drift: n/a` | Ek snapshot sağlanana kadar normaldir. |
| AGF/Ganyan boş | İlgili alanlar null | CSV’de bu sütunları doldurun; yoksa pipeline çalışmaya devam eder. |

`--verbose` veya `--dry-run` bayrakları CLI komutlarına eklenebilir (varsa) ve log seviyesini artırarak hata ayıklamayı kolaylaştırır.

## 13. Veri Gizliliği ve Log Politikası

- Loglar sadece `race_uid`, `horse_uid`, `At İsmi`, `Start No` gibi yarış bazlı kimlikler içerir.
- Kişisel veri toplanmaz.
- Reprodüksiyon için rastgele tohum değerleri sabittir; eğitim/validation split parametreleri JSON konfigürasyon dosyalarına kaydedilir.

## 14. Sürümleme & Değişiklik Kaydı

- Pipeline şema v1.0.0 ile uyumludur.
- `CHANGELOG.md` için örnek giriş:
  ```markdown
  ## [1.1.0] - 2025-09-30
  ### Added
  - Yeni gate_context varyantı.
  ### Fixed
  - AGF parse hatası.
  ```
- Sürüm artırırken semantik versiyonlama (MAJOR.MINOR.PATCH) kullanın; şema değişikliği MAJOR artışı gerektirir.

## 15. Katkı Rehberi

- Issue açarken başlık, beklenen davranış ve örnek CSV parçalarını paylaşın.
- PR gönderirken açıklama, yapılan değişiklikler, test sonuçları ve ilgili CLI komutlarını ekleyin.
- Kod stili: PEP8 uyumlu Python, tip ipuçları tercih edilir.
- Test için örnek komutlar:
  ```bash
  python -m cli.synth --out test_program.csv --n-races 2
  python -m cli.train --program test_program.csv --val-date "2025-09-20" --cpu-only
  python -m cli.predict --program test_program.csv --out test.json --report test.md --cpu-only
  ```

## 16. SSS

- **Scraper nerede?** Scraper ayrı projedir; bu pipeline yalnızca harici CSV dosyalarıyla çalışır.
- **Tarih formatı değişirse?** Pipeline içinde post-process dönüştürücüler vardır; scraper’a müdahale etmeyin.
- **GPU kullanabilir miyim?** Gerekli değil; RTX optimizasyon kodu hazır fakat varsayılan olarak kapalı. `--cpu-only` bayrağıyla CPU modunda kalın.

## 17. Sözlük

- **AGF**: Altılı Ganyan Favori yüzdesi.
- **Overround**: Bahis oranlarının toplamı üzerinden pazar marjı.
- **p_market**: İmplied probability’lerin normalize edilmiş hali.
- **MDI**: AGF ile pazar olasılığı arasındaki farklılığın sigmoid skorlaması.
- **Gate context**: Start kapısı, pist tipi/durumu ve hipodrom kombinasyonuyla oluşturulan bağlam anahtarı.
- **Field-wise normalization**: Aynı yarıştaki atlar için normalize edilmiş istatistikler.

## 18. Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında sunulmaktadır.
