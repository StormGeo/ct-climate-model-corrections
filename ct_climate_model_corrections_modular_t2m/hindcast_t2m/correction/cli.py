import argparse
from pathlib import Path
from typing import Optional, List


from .config import CorrectionConfig
from .pipeline import ForecastCorrectionPipeline


TEMP_ALLOWED = {
    "2m_air_temperature",
    "2m_air_temperature_min",
    "2m_air_temperature_max",
}


def detect_temp_var_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts if p]
    if "2m_air_temperature_min" in parts:
        return "2m_air_temperature_min"
    if "2m_air_temperature_max" in parts:
        return "2m_air_temperature_max"
    if "2m_air_temperature" in parts:
        return "2m_air_temperature"
    raise ValueError(
        f"Não consegui detectar variável de temperatura pelo caminho: {path}\n"
        "Esperado uma pasta com: 2m_air_temperature_min | 2m_air_temperature_max | 2m_air_temperature"
    )


def find_clim_obs_file_auto(clim_path: Path, var_name: str) -> Path:
    p = clim_path.expanduser().resolve()

    if p.is_file():
        return p

    if not p.is_dir():
        raise FileNotFoundError(f"--clim-file não existe (arquivo ou pasta): {p}")

    cands = sorted(list(p.rglob("*.nc")))
    if not cands:
        raise FileNotFoundError(f"Nenhum .nc encontrado em --clim-file: {p}")

    vlow = var_name.lower()
    by_var = [f for f in cands if vlow in str(f).lower()]

    if not by_var:
        if var_name.endswith("_min"):
            keys = ["tmin", "temperature_min", "air_temperature_min", "min"]
        elif var_name.endswith("_max"):
            keys = ["tmax", "temperature_max", "air_temperature_max", "max"]
        else:
            keys = ["2m_air_temperature", "t2m", "mean", "temperature", "air_temperature"]
        by_var = [f for f in cands if any(k in str(f).lower() for k in keys)]

    if not by_var:
        raise FileNotFoundError(
            f"Não encontrei climatologia para var='{var_name}' dentro de: {p}\n"
            "Dica: garanta que o nome do arquivo/pasta contenha '2m_air_temperature[_min|_max]' ou alias (t2m/tmin/tmax).\n"
            "Ou passe diretamente o arquivo .nc em --clim-file."
        )

    by_daily = [f for f in by_var if "daily" in f.name.lower()]
    pool = by_daily if by_daily else by_var
    chosen = sorted(pool, key=lambda f: f.stat().st_mtime, reverse=True)[0]
    return chosen


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Daily temperature correction: lead-wise additive bias (obs_month_mean - hind_mean) + "
            "regrid + absolute clipping around daily observed climatology.\n"
            "--clim-file pode ser um arquivo .nc ou uma pasta (auto-detecta o arquivo correto pela variável)."
        )
    )
    p.add_argument("--forecast-root", required=True, help="Root com DOY folders contendo forecast .nc (ou DOY folder)")
    p.add_argument("--hindcast-root", required=True, help="Root com DOY folders contendo hindcast processado .nc (ou YEAR/DOY)")
    p.add_argument(
        "--clim-file",
        required=True,
        help="Climatologia observada DIÁRIA: arquivo .nc (time>=365) OU pasta contendo .nc (auto-detect).",
    )
    p.add_argument("--out-root", required=True, help="Base output directory")
    p.add_argument(
        "--var-name",
        default=None,
        help=(
            "Nome da variável (2m_air_temperature[_min|_max]). "
            "Se omitido, detecta automaticamente pelo caminho de --forecast-root."
        ),
    )
    p.add_argument("--subfolder", default=None, help="Subpasta opcional dentro de cada DOY para buscar forecast files")
    p.add_argument("--no-skip", action="store_true", help="Não pular outputs já existentes")
    p.add_argument("--save-regrid-weights", dest="save_regrid_weights", action="store_true", default=True,
                   help="Salvar os pesos do regrid para reutilização futura.")
    p.add_argument("--no-save-regrid-weights", dest="save_regrid_weights", action="store_false",
                   help="Não salvar os pesos do regrid.")
    p.add_argument("--regrid-cache-subdir", default="cache",
                   help="Subpasta dentro de --out-root para salvar pesos do regrid.")

    p.add_argument("--max-days-output", type=int, default=280, help="Hard limit output to N daily steps (default 280)")
    p.add_argument("--extend-months", type=int, default=1, help="Allow N extra months beyond hindcast leads (default 1)")

    p.add_argument("--no-clip", action="store_true", help="Disable absolute clipping around daily climatology")
    p.add_argument("--clip-delta-c", type=float, default=1.5, help="Absolute delta in °C (default 1.5)")
    return p.parse_args()


def main():
    args = parse_args()

    forecast_root = Path(args.forecast_root)
    hindcast_root = Path(args.hindcast_root)

    var_name = str(args.var_name).strip() if args.var_name else None
    if not var_name:
        # prefer forecast-root, fallback hindcast-root
        try:
            var_name = detect_temp_var_from_path(forecast_root)
        except Exception:
            var_name = detect_temp_var_from_path(hindcast_root)

    if var_name not in TEMP_ALLOWED:
        raise ValueError(f"var-name inválido: {var_name}. Esperado: {sorted(TEMP_ALLOWED)}")

    clim_file = find_clim_obs_file_auto(Path(args.clim_file), var_name)

    cfg = CorrectionConfig(
        forecast_root=forecast_root,
        hindcast_root=hindcast_root,
        clim_file=clim_file,
        out_root=Path(args.out_root),
        var_name=var_name,
        subfolder=(str(args.subfolder) if args.subfolder else None),
        skip_existing=(not bool(args.no_skip)),
        max_days_output=int(args.max_days_output),
        extend_months=int(args.extend_months),
        clip_with_abs_limit=(not bool(args.no_clip)),
        clip_delta_c=float(args.clip_delta_c),
        save_regrid_weights=bool(args.save_regrid_weights),
        regrid_cache_subdir=str(args.regrid_cache_subdir),
    )

    ForecastCorrectionPipeline(cfg).run()


if __name__ == "__main__":
    main()
