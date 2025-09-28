from __future__ import annotations

from typing import Any, Dict, List


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def generate_report(races: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for race in races:
        meta = race["meta"]
        header = f"{meta['hipodrom']} | {meta['tarih']} {meta['kosu_saati']} | {meta['kosu_sinifi']} • {meta['mesafe']}m • {meta['pist_tipi']}"
        lines.append(header)
        lines.append("Top 5:")
        top_preds = race["predictions"][:5]
        for pred in top_preds:
            start_no = pred.get("start_no") if pred.get("start_no") is not None else "x"
            ganyan = pred.get("ganyan")
            ganyan_str = f"{ganyan:.2f}" if ganyan is not None else "n/a"
            edge = pred.get("edge")
            mdi = pred.get("mdi")
            drift = pred.get("drift_dp15")
            drift_str = f"{drift:.3f}" if drift is not None else "n/a"
            edge_val = edge * 100 if edge is not None else 0.0
            mdi_str = f"{mdi:.3f}" if mdi is not None else "n/a"
            line = (
                f"#{start_no} {pred['at_ismi']} — Win {_fmt_percent(pred['win_prob'])} | "
                f"Place {_fmt_percent(pred['place_prob'])} | ExpFin {pred['expected_finish']:.2f} | "
                f"Ganyan {ganyan_str} | Edge {edge_val:.1f}% | "
                f"MDI {mdi_str} | Drift15 {drift_str}"
            )
            lines.append(line)
        gate_ctx = top_preds[0]["extras"].get("gate_context_key") if top_preds else None
        calibration = meta.get("calibration", {})
        param = calibration.get("param")
        param_str = f"{param:.3f}" if isinstance(param, (float, int)) else "null"
        drift_note = "drift: n/a" if all(pred.get("drift_dp15") is None for pred in race["predictions"]) else "drift: active"
        lines.append(
            f"Notlar: calibration={calibration.get('method')}(param={param_str}), N={len(race['predictions'])}, "
            f"field-wise norm aktif, overround düzeltmesi aktif, gate_ctx={gate_ctx}, {drift_note}"
        )
        lines.append("")
    return "\n".join(lines).strip()
