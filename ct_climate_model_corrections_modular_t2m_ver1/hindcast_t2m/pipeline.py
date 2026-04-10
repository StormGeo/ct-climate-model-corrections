# hindcast_t2m/pipeline.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import HindcastConfig
from .detector import detect_processor
from .download import download_grib
from .io import open_grib
from .regrid import regrid_dataset
from .utils import out_folder_for_month


def _norm_centre(value: str) -> str:
    # Normaliza aliases comuns: meteo_france -> meteo-france, etc.
    return str(value).lower().strip().replace("_", "-")


class HindcastPipeline:
    def __init__(
        self,
        cfg: HindcastConfig,
        out_grib_root: Path,
        out_nc_root: Path,
        enable_regrid: bool,
        ref_grid: Optional[Path],
    ):
        self.cfg = cfg
        self.out_grib_root = out_grib_root.expanduser().resolve()
        self.out_nc_root = out_nc_root.expanduser().resolve()
        self.enable_regrid = enable_regrid
        self.ref_grid = ref_grid.expanduser().resolve() if ref_grid else None

        self.out_grib_root.mkdir(parents=True, exist_ok=True)
        self.out_nc_root.mkdir(parents=True, exist_ok=True)

    def build_grib_outfile(self, init_month_str: str, folder_path: Path) -> Path:
        centre = self.cfg.originating_centre
        system = self.cfg.system
        # informational filename only
        stat = (self.cfg.product_type[0] if self.cfg.product_type else "monthly_mean")
        return (
            folder_path
            / f"{centre}_sys{system}_t2m_hindcast_{stat}_init{init_month_str}_1993-2016_lead1-6_global.grib"
        )

    def build_operational_nc_filename(self, var_name: str, init_stamp: str) -> str:
        return f"hindcast_{self.cfg.model_prefix}_{var_name}_{init_stamp}.nc"

    def build_nc_outfile_operational(self, nc_dir: Path, out_year: int, month_str: str) -> Path:
        init_stamp = f"{out_year:04d}{int(month_str):02d}0100"
        return nc_dir / self.build_operational_nc_filename(str(self.cfg.out_var_name), init_stamp)

    def run_month(self, month_str: str, out_year: int) -> Path:
        """
        month_str: dois dígitos da inicialização que queremos processar (ex.: "01" para init=jan).

        Alguns centros/sistemas precisam solicitar o GRIB do mês anterior para alinhar o "lead 1"
        com o mês-alvo (month_str). A saída NC continua sendo organizada pelo mês-alvo.
        """

        def prev_month_str(y: int, m_str: str) -> tuple[int, str]:
            m = int(m_str)
            if m == 1:
                return (y - 1, "12")
            return (y, f"{m - 1:02d}")

        subdir = out_folder_for_month(month_str)

        # Normalizados para comparação
        centre_norm = _norm_centre(self.cfg.originating_centre)
        system = str(self.cfg.system)

        # Mês/ano que serão realmente solicitados no CDS
        request_year = out_year
        request_month = month_str

        # Modelos que precisam pedir o mês anterior para alinhar o "lead 1" com o mês alvo
        needs_prev_month = False

        # Météo-France sys9
        if centre_norm in ("lfpw", "meteofrance", "meteo-france") and system == "9":
            needs_prev_month = True

        # UKMO sys603
        elif centre_norm in ("egrr", "ukmo", "uk-met-office", "ukmetoffice", "metoffice") and system == "603":
            needs_prev_month = True

        # JMA sys3 (lead 1 vem vazio)
        elif centre_norm in ("rjtd", "jma") and system == "3":
            request_year, request_month = prev_month_str(out_year, month_str)

        # DWD sys2
        elif centre_norm in ("edzw", "dwd") and system == "2":
            request_year, request_month = prev_month_str(out_year, month_str)

        # CMCC sys35
        elif centre_norm in ("cmcc", "cnmc") and system == "35":
            needs_prev_month = True

        # ECCC sys4
        elif centre_norm in ("eccc", "cwao") and system == "4":
            needs_prev_month = True

        # NCEP sys2
        elif centre_norm in ("kwbc", "ncep") and system == "2":
            request_year, request_month = prev_month_str(out_year, month_str)

        if needs_prev_month:
            request_year, request_month = prev_month_str(out_year, month_str)

        # Organiza GRIB por request_year (o ano real do arquivo solicitado)
        grib_dir = self.out_grib_root
        # Organiza NC exatamente no diretório informado
        nc_dir = self.out_nc_root
        grib_dir.mkdir(parents=True, exist_ok=True)
        nc_dir.mkdir(parents=True, exist_ok=True)

        # Nome do GRIB baseado no mês que vai ser solicitado (request_month)
        grib_file = self.build_grib_outfile(request_month, grib_dir)
        # Nome do NC baseado no mês-alvo (month_str)
        nc_file = self.build_nc_outfile_operational(nc_dir, out_year, month_str)

        # DEBUG (pode deixar; é leve e ajuda a validar)
        print("DEBUG raw centre:", self.cfg.originating_centre)
        print("DEBUG normalized centre:", centre_norm)
        print("DEBUG system:", system, "month_str(target):", month_str, "out_year:", out_year)
        print("DEBUG needs_prev_month:", needs_prev_month)
        print("DEBUG request_year/request_month:", request_year, request_month)
        print("DEBUG grib_file:", grib_file)

        # Download (se não existir)
        if grib_file.exists() and grib_file.stat().st_size > 0:
            print("\n=== DOWNLOAD SKIP ===")
            print(f"[INFO] GRIB já existe, pulando download: {grib_file}")
        else:
            download_grib(self.cfg, request_month, grib_file)

        print("\n=== PROCESSING ===")
        print(f"Input GRIB: {grib_file}")
        ds = open_grib(grib_file, self.cfg)

        processor = detect_processor(ds, self.cfg)
        print(f"[INFO] processor selecionado: {processor.name}")

        if hasattr(processor, "set_target_month"):
            processor.set_target_month(int(month_str))

        out = processor.process_grib(ds, self.cfg)

        if self.enable_regrid:
            if self.ref_grid is None:
                raise ValueError("With --regrid you must provide --ref-grid /path/to/grid.nc")
            if not self.ref_grid.exists():
                raise FileNotFoundError(f"Reference grid file does not exist: {self.ref_grid}")

            weights_file = None
            if self.cfg.save_regrid_weights:
                weights_dir = self.out_nc_root / self.cfg.regrid_cache_subdir
                weights_dir.mkdir(parents=True, exist_ok=True)
                weights_file = weights_dir / f"weights_{self.cfg.regrid_method}_periodic{int(self.cfg.regrid_periodic)}.nc"

            print(f"[INFO] Regridding enabled: method={self.cfg.regrid_method} periodic={self.cfg.regrid_periodic}")
            out = regrid_dataset(out, self.ref_grid, self.cfg, weights_file)

        print("\n=== SAVING ===")
        print(f"Output NC: {nc_file}")
        nc_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(nc_file)
        print("[OK] Done")
        return nc_file