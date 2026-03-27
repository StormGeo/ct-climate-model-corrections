from __future__ import annotations
from pathlib import Path
from typing import Optional

import xarray as xr

from .config import HindcastConfig
from .download import download_grib
from .detector import detect_processor
from .io import open_grib
from .regrid import regrid_dataset
from .utils import out_folder_for_month

class HindcastPipeline:
    def __init__(self, cfg: HindcastConfig, out_grib_root: Path, out_nc_root: Path, enable_regrid: bool, ref_grid: Optional[Path]):
        self.cfg = cfg
        self.out_grib_root = out_grib_root.expanduser().resolve()
        self.out_nc_root = out_nc_root.expanduser().resolve()
        self.enable_regrid = enable_regrid
        self.ref_grid = ref_grid.expanduser().resolve() if ref_grid else None
        self.out_grib_root.mkdir(parents=True, exist_ok=True)
        self.out_nc_root.mkdir(parents=True, exist_ok=True)

    def build_grib_outfile(self, month_str: str, folder_path: Path) -> Path:
        centre = self.cfg.originating_centre
        system = self.cfg.system
        return folder_path / f"{centre}_sys{system}_tp_hindcast_monthly_mean_init{month_str}_1993-2016_lead1-6_global.grib"

    def build_operational_nc_filename(self, var_name: str, init_stamp: str) -> str:
        return f"{self.cfg.model_prefix}_{var_name}_AVG_{init_stamp}.nc"

    def build_nc_outfile_operational(self, nc_dir: Path, out_year: int, month_str: str) -> Path:
        init_stamp = f"{out_year:04d}{int(month_str):02d}0100"
        return nc_dir / self.build_operational_nc_filename("total_precipitation", init_stamp)

    def run_month(self, month_str: str, out_year: int) -> Path:
        """
        month_str: dois dígitos da inicialização que queremos processar (ex.: "01" para init=jan).
        Para Météo-France sys9, solicitamos o GRIB do mês ANTERIOR, para alinhar meses durante a correção.
        """
        from calendar import monthrange
    
        def prev_month_str(y: int, m_str: str) -> tuple[int, str]:
            m = int(m_str)
            if m == 1:
                return (y - 1, f"{12:02d}")
            return (y, f"{(m - 1):02d}")
    
        subdir = out_folder_for_month(month_str)
    
        # Decide qual mês pedir ao CDS / procurar no disco
        request_year = out_year
        request_month = month_str

        centre = str(self.cfg.originating_centre).lower()
        system = str(self.cfg.system)
        prefix = str(getattr(self.cfg, "model_prefix", "")).lower()

        request_year = out_year
        request_month = month_str

        # Modelos que precisam pedir o mês anterior para alinhar o "lead 1" com o mês alvo
        needs_prev_month = False

        # Météo-France sys9
        if centre in ("lfpw", "meteofrance", "meteo-france") and system == "9":
            needs_prev_month = True

        # UKMO sys603 (se você já usa esse alinhamento)
        elif centre == "egrr" and system == "603":
            needs_prev_month = True

        # JMA sys3 (o seu caso: lead 1 vem vazio)
        elif centre in ("rjtd", "jma") and str(system) == "3":
            request_year, request_month = prev_month_str(out_year, month_str)

        # DWD sys2 (edzw/dwd): produto começa no lead real=2 -> pede mês anterior
        elif centre in ("edzw", "dwd") and system == "2":
            request_year, request_month = prev_month_str(out_year, month_str)

        # (opcional) se você quer amarrar por prefixo também:
        # elif centre == "rjtd" and system == "3" and prefix.startswith("jma_subseas_glo"):
        #     needs_prev_month = True

        # CMCC sys35 (edzw/cmcc): produto começa no lead real=2 -> pede mês anterior
        elif centre in ("cmcc", "cnmc") and str(system) == "35":
            needs_prev_month = True

        # ECCC sys4 (eccc/cwao): produto começa no lead real=2 -> pede mês anterior
        elif centre in ("eccc", "cwao") and str(system) == "4":
            needs_prev_month = True

        # NCEP sys2 (kwbc/ncep): produto começa no lead real=2 -> pede mês anterior
        elif centre in ("kwbc", "ncep") and str(system) == "2":
            request_year, request_month = prev_month_str(out_year, month_str)

        if needs_prev_month:
            request_year, request_month = prev_month_str(out_year, month_str)

    
        # Use request_year when organizando GRIBs por ano (recomendado)
        grib_dir = self.out_grib_root
        nc_dir = self.out_nc_root
        grib_dir.mkdir(parents=True, exist_ok=True)
        nc_dir.mkdir(parents=True, exist_ok=True)
    
        # Construir nome do arquivo GRIB a partir do mês que vamos solicitar
        grib_file = self.build_grib_outfile(request_month, grib_dir)
        nc_file = self.build_nc_outfile_operational(nc_dir, out_year, month_str)
    
        # Skip download se arquivo existir e tamanho > 0
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
                weights_file = weights_dir / f"weights_{self.cfg.regrid_method}_periodic{int(self.cfg.regrid_periodic)}.nc"
    
            print(f"[INFO] Regridding enabled: method={self.cfg.regrid_method} periodic={self.cfg.regrid_periodic}")
            out = regrid_dataset(out, self.ref_grid, self.cfg, weights_file)
    
        print("\n=== SAVING ===")
        print(f"Output NC: {nc_file}")
        nc_file.parent.mkdir(parents=True, exist_ok=True)
        out.to_netcdf(nc_file)
        print("[OK] Done")
        return nc_file

