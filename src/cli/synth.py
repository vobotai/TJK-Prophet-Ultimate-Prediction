from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

PROGRAM_COLUMNS = [
    "Program Başlık",
    "Tarih",
    "Gün",
    "Hipodrom Başlık",
    "Hipodrom",
    "Yarış Günü Numarası",
    "Program Şehir ID",
    "Çim Durumu",
    "Kum Durumu",
    "Hava Bilgisi",
    "Hava Sıcaklığı",
    "Hava Durumu",
    "Nem",
    "Koşu ID",
    "Koşu Başlık",
    "Koşu Numarası",
    "Koşu Saati",
    "Özel Koşu Adı",
    "Koşu Sınıfı",
    "Baz Sıklet",
    "Mesafe",
    "Pist Tipi",
    "En İyi Derece (Tarihçe)",
    "Koşu Koşulları",
    "Muhtemel Linki",
    "Koşu PDF",
    "Pist Şeması Görseli",
    "Program Sırası",
    "At İsmi",
    "At Detay Linki",
    "Donanım Kodları",
    "Yaş Bilgisi",
    "Orijin (Baba - Anne)",
    "Baba",
    "Baba Linki",
    "Anne",
    "Anne Linki",
    "Kısrak Babası",
    "Kısrak Babası Linki",
    "Sıklet",
    "Jokey",
    "Jokey Linki",
    "Sahip",
    "Sahip Linki",
    "Antrenör",
    "Antrenör Linki",
    "Start No",
    "Handikap Puanı",
    "Son 6 Yarış",
    "KGS",
    "s20",
    "En İyi Derece",
    "En İyi Derece Linki",
    "En İyi Derece Açıklaması",
    "Ganyan",
    "AGF",
    "AGF Açıklaması",
    "İdman Linki",
    "W_Workout_Count",
    "W_Latest_Date",
    "W_Latest_Hip",
    "W_800m_Best",
    "W_600m_Best",
    "W_Match_Score",
    "W_Match_Method",
    "W_Data_Quality",
    "Result_Win",
    "Result_Place",
    "Finish_Position",
    "Race_Time"
]

WORKOUT_COLUMNS = [
    "Tarih",
    "Hipodrom",
    "Koşu ID",
    "At İsmi",
    "At ID",
    "Workout Tarih",
    "W_Hip",
    "W_Pist",
    "W_Type",
    "W_Status",
    "W_Jokey",
    "w1200_s",
    "w1000_s",
    "w800_s",
    "w600_s",
    "w400_s",
    "w200_s",
    "matched_name",
    "matched_url",
    "match_method",
    "match_score",
]


def generate_program(n_races: int, base_date: datetime, city: str) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)
    records: List[Dict[str, object]] = []
    for r in range(n_races):
        race_date = base_date + timedelta(days=r)
        tarih_str = race_date.strftime("%d/%m/%Y")
        hipodrom = city
        pist_tipi = random.choice(["Çim", "Kum", "Sentetik"])
        pist_durum = random.choice(["İyi", "Islak", "Kuru"])
        distance = random.choice([1200, 1400, 1600, 1800, 2000])
        field_size = random.randint(7, 12)
        base_time = 1.35 + (distance - 1200) / 1000 * 0.1
        winners = list(range(1, field_size + 1))
        random.shuffle(winners)
        race_id = f"{r+1:03d}"
        for start_no in range(1, field_size + 1):
            horse_name = f"Horse {r+1}-{start_no}"
            ganyan = round(float(np.random.lognormal(mean=0.2 + start_no * 0.05, sigma=0.3)), 2)
            agf = max(0.01, min(0.95, float(np.random.normal(loc=0.5 / start_no, scale=0.05)))) * 100
            workout_count = random.randint(0, 5)
            latest_date = race_date - timedelta(days=workout_count + random.randint(1, 5)) if workout_count else race_date - timedelta(days=random.randint(7, 15))
            workout_quality = random.choice(["matched", "unmatched"])
            finish_pos = winners[start_no - 1]
            record = {col: "" for col in PROGRAM_COLUMNS}
            record.update(
                {
                    "Program Başlık": f"Program {r+1}",
                    "Tarih": tarih_str,
                    "Gün": race_date.strftime("%A"),
                    "Hipodrom Başlık": hipodrom,
                    "Hipodrom": hipodrom,
                    "Yarış Günü Numarası": r + 1,
                    "Program Şehir ID": random.randint(1, 20),
                    "Çim Durumu": pist_durum if pist_tipi == "Çim" else "",
                    "Kum Durumu": pist_durum if pist_tipi != "Çim" else "",
                    "Hava Bilgisi": "",
                    "Hava Sıcaklığı": round(random.uniform(10, 25), 1),
                    "Hava Durumu": random.choice(["Güneşli", "Yağmurlu", "Bulutlu"]),
                    "Nem": round(random.uniform(30, 80), 1),
                    "Koşu ID": race_id,
                    "Koşu Başlık": f"{distance}m {pist_tipi}",
                    "Koşu Numarası": r + 1,
                    "Koşu Saati": f"{12 + r:02d}:{30 + start_no:02d}",
                    "Koşu Sınıfı": random.choice(["KV6", "Şartlı 3", "Handikap 15"]),
                    "Baz Sıklet": round(random.uniform(52, 60), 1),
                    "Mesafe": f"{distance} m",
                    "Pist Tipi": pist_tipi,
                    "En İyi Derece (Tarihçe)": "1:34.50",
                    "Program Sırası": start_no,
                    "At İsmi": horse_name,
                    "Donanım Kodları": random.choice(["KG", "Dilbağı", ""]),
                    "Yaş Bilgisi": random.randint(3, 6),
                    "Orijin (Baba - Anne)": "Baba- Anne",
                    "Baba": "Victory Gallop",
                    "Anne": "Brave Lady",
                    "Kısrak Babası": "Bold Pilot",
                    "Sıklet": round(random.uniform(50, 60), 1),
                    "Jokey": f"Jokey {start_no}",
                    "Sahip": f"Owner {start_no}",
                    "Antrenör": f"Trainer {start_no}",
                    "Start No": start_no,
                    "Handikap Puanı": round(random.uniform(70, 95), 1),
                    "KGS": round(random.uniform(40, 60), 1),
                    "s20": round(random.uniform(80, 110), 1),
                    "En İyi Derece": "1.34.50",
                    "Ganyan": ganyan,
                    "AGF": agf,
                    "W_Workout_Count": workout_count,
                    "W_Latest_Date": latest_date.strftime("%d/%m/%Y"),
                    "W_Latest_Hip": hipodrom,
                    "W_800m_Best": round(random.uniform(47, 52) - start_no * 0.1, 2) if workout_count else "",
                    "W_600m_Best": round(random.uniform(35, 38) - start_no * 0.05, 2) if workout_count else "",
                    "W_Match_Score": random.randint(85, 99) if workout_count else "",
                    "W_Match_Method": random.choice(["exact", "fuzzy"]),
                    "W_Data_Quality": workout_quality,
                    "Result_Win": 1 if finish_pos == 1 else 0,
                    "Result_Place": 1 if finish_pos <= 3 else 0,
                    "Finish_Position": finish_pos,
                    "Race_Time": round((base_time + start_no * 0.02) * 60, 2),
                }
            )
            records.append(record)
    df = pd.DataFrame(records)
    return df[PROGRAM_COLUMNS]


def generate_workouts(program_df: pd.DataFrame) -> pd.DataFrame:
    records: List[List] = []
    for _, row in program_df.iterrows():
        if not row["W_Workout_Count"]:
            continue
        for w in range(int(row["W_Workout_Count"])):
            w_date = datetime.strptime(row["W_Latest_Date"], "%d/%m/%Y") - timedelta(days=w * 2)
            match_score = max(85, min(100, float(row["W_Match_Score"] or 88) - w))
            records.append([
                row["Tarih"],
                row["Hipodrom"],
                row["Koşu ID"],
                row["At İsmi"],
                f"{row['Koşu ID']}-{row['At İsmi']}",
                w_date.strftime("%d/%m/%Y"),
                row["Hipodrom"],
                row["Pist Tipi"],
                "Sprint",
                "Tamamlandı",
                row["Jokey"],
                round(float(row["W_800m_Best"] or 50) + w * 0.5, 2),
                round(float(row["W_800m_Best"] or 50) + w * 0.8, 2),
                round(float(row["W_800m_Best"] or 50) + w * 0.3, 2),
                round(float(row["W_600m_Best"] or 36) + w * 0.2, 2),
                round(24 + w * 0.2, 2),
                round(12 + w * 0.1, 2),
                row["At İsmi"],
                "",
                "exact",
                match_score,
            ])
    if not records:
        return pd.DataFrame(columns=WORKOUT_COLUMNS)
    return pd.DataFrame(records, columns=WORKOUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workouts", type=Path, default=None)
    parser.add_argument("--n-races", type=int, default=6)
    parser.add_argument("--city", type=str, default="Ankara")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%d/%m/%Y"))
    args = parser.parse_args()

    base_date = datetime.strptime(args.date, "%d/%m/%Y")
    program_df = generate_program(args.n_races, base_date, args.city)
    program_df.to_csv(args.out, index=False)

    if args.workouts:
        workouts_df = generate_workouts(program_df)
        workouts_df.to_csv(args.workouts, index=False)


if __name__ == "__main__":
    main()
