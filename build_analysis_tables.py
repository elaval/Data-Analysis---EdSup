#!/usr/bin/env python3
"""
build_analysis_tables.py
========================
Genera 3 tablas analíticas por IE y las guarda en datos_analisis/:

  dim_estudiante.parquet    — perfil escolar/socioeconómico de cada estudiante
  trayectoria_es.parquet    — resumen trayectoria en educación superior
  cohorte_ingreso.parquet   — ingresantes por carrera y año en la IE focal

Uso:
    python build_analysis_tables.py 70          # Universidad de Chile
    python build_analysis_tables.py 86          # PUC
    python build_analysis_tables.py 70 86 23    # varias IEs a la vez
"""

import argparse
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(
    "/Users/ernestolaval/Documents/Prototipos/Data Analysis - EdSup/Datos por IES"
)

# cod_ense que corresponden a educación de adultos en Chile
ADULT_ENSE_CODES = frozenset(
    {360, 361, 363, 460, 463, 560, 563, 660, 663, 760, 860, 960}
)

FORMA_INGRESO_MAP = {
    "1":  "Ingreso Directo",
    "2":  "Continuidad Plan Común",
    "3":  "Cambio Interno",
    "4":  "Cambio Externo",
    "5":  "RAP",
    "6":  "Extranjeros",
    "7":  "PACE",
    "8":  "Inclusión",
    "9":  "Características Especiales",
    "10": "Otras",
}

# ─── helpers ──────────────────────────────────────────────────────────────────

def to_int(s: pd.Series) -> pd.Series:
    """Convierte serie de cualquier tipo a nullable Int64."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def birth_year(s: pd.Series) -> pd.Series:
    """
    Extrae el año de nacimiento de formatos mixtos:
      - 6 dígitos  YYYYMM   → divide por 100   (formato en Matricula_Ed_Superior)
      - 8 dígitos  YYYYMMDD → divide por 10000  (formato en Egresados_EM / SEP)
    """
    num = pd.to_numeric(s, errors="coerce").astype("Int64")
    result = pd.array([pd.NA] * len(num), dtype="Int64")
    mask_8 = num.notna() & (num >= 10_000_000)
    mask_6 = num.notna() & (num >= 100_000) & (num < 10_000_000)
    result[mask_8] = (num[mask_8] // 10_000).astype("Int64")
    result[mask_6] = (num[mask_6] // 100).astype("Int64")
    return pd.Series(result, index=s.index)


def normalize_forma_ingreso(s: pd.Series) -> pd.Series:
    """Normaliza los códigos de forma_ingreso a etiquetas legibles."""
    code = s.astype(str).str.extract(r"^(\d+)", expand=False).str.strip()
    return code.map(FORMA_INGRESO_MAP).fillna(s.str.strip())


def load_mruns(ie_dir: Path, cod_inst: str) -> set:
    df = pd.read_parquet(ie_dir / f"Estudiantes_{cod_inst}.parquet")
    return set(to_int(df["mrun"]).dropna())


# ─── 1. dim_estudiante ────────────────────────────────────────────────────────

def build_dim_estudiante(
    ie_dir: Path,
    cod_inst: str,
    mruns: set,
    mat_es: pd.DataFrame,
) -> pd.DataFrame:
    """
    Una fila por mrun. Combina:
      - NEM (jóvenes + adultos): agno_egreso_em, rbd, cod_depe, NEM, percentil
      - Egresados_EM: año nacimiento, género, datos del establecimiento al egreso
      - Alumnos_SEP: marca prioritario/preferente al egreso y en cualquier año
      - puntajes_admision 2021–2025: puntajes PAES/PSU
      - Matricula_Ed_Superior: fallback año nacimiento / género
    """
    print("  [dim_estudiante] Leyendo fuentes...")

    # ── NEM ─────────────────────────────────────────────────────────────────
    nem_j = pd.read_parquet(ie_dir / f"NEM_y_Percentil_Jovenes_{cod_inst}.parquet")
    nem_j["es_adultos"] = False
    nem_a = pd.read_parquet(ie_dir / f"NEM_y_Percentil_Adultos_{cod_inst}.parquet")
    nem_a["es_adultos"] = True
    nem_raw = pd.concat([nem_j, nem_a], ignore_index=True)

    nem_raw["mrun"]           = to_int(nem_raw["mrun"])
    nem_raw["agno_egreso_em"] = to_int(nem_raw["agno_egreso"])
    nem_raw["rbd_egreso"]     = to_int(nem_raw["rbd"])
    nem_raw["cod_depe_egreso"] = nem_raw["cod_depe"].astype(str).str.strip()
    nem_raw["nem"]            = to_float(nem_raw["nem"])
    nem_raw["percentil_egreso"] = to_int(nem_raw["percentil"])
    nem_raw["top_10"]  = (
        nem_raw["puesto_10"].astype(str).str.upper().str.strip() == "SI"
    ).astype("Int64")
    nem_raw["decil_1_3"] = (nem_raw["percentil_egreso"] <= 30).astype("Int64")

    nem = (
        nem_raw[["mrun", "agno_egreso_em", "rbd_egreso", "cod_depe_egreso",
                  "nem", "percentil_egreso", "top_10", "decil_1_3", "es_adultos"]]
        .sort_values("agno_egreso_em", ascending=False)
        .drop_duplicates("mrun")
    )

    # ── Egresados EM ────────────────────────────────────────────────────────
    egr_raw = pd.read_parquet(ie_dir / f"Egresados_EM_{cod_inst}.parquet")
    egr_raw["mrun"]         = to_int(egr_raw["mrun"])
    egr_raw["agno_egr"]     = to_int(egr_raw["agno"])
    egr_raw["rbd_egr"]      = to_int(egr_raw["rbd"])
    egr_raw["cod_ense_egr"] = to_int(egr_raw["cod_ense"])
    egr_raw["marca_egreso"] = to_int(egr_raw["marca_egreso"])
    egr_raw["rural_rbd"]    = to_int(egr_raw["rural_rbd"])

    egr_grad = egr_raw[egr_raw["marca_egreso"] == 1].copy()
    if egr_grad.empty:
        egr_grad = egr_raw.copy()
    egr_grad = egr_grad.sort_values("agno_egr", ascending=False).drop_duplicates("mrun")

    # Egresados_EM no tiene fec_nac_alu ni gen_alu; esos vienen de Matricula (fallback)
    keep_egr = ["mrun", "cod_ense_egr", "rural_rbd",
                "cod_depe",    # dependencia del establecimiento
                "cod_com_rbd", "nom_com_rbd",
                "cod_reg_rbd", "nom_reg_rbd_a"]
    egr = egr_grad[[c for c in keep_egr if c in egr_grad.columns]]

    # ── Lookup RBD → nombre establecimiento (desde SEP) ─────────────────────
    sep_raw = pd.read_parquet(ie_dir / f"Alumnos_SEP_{cod_inst}.parquet")
    sep_raw["mrun"]      = to_int(sep_raw["mrun"])
    sep_raw["agno_sep"]  = to_int(sep_raw["agno"])
    sep_raw["rbd_sep"]   = to_int(sep_raw["rbd"])
    sep_raw["prioritario"] = to_int(sep_raw["prioritario_alu"])
    sep_raw["preferente"]  = to_int(sep_raw["preferente_alu"])

    rbd_nom_lookup = (
        sep_raw[["rbd_sep", "nom_rbd"]]
        .dropna(subset=["nom_rbd"])
        .drop_duplicates("rbd_sep")
        .set_index("rbd_sep")["nom_rbd"]
        .to_dict()
    )

    # ── SEP: alguna vez ─────────────────────────────────────────────────────
    sep_any = (
        sep_raw.groupby("mrun")
        .agg(
            prioritario_alguna_vez=("prioritario", lambda x: int((x == 1).any())),
            preferente_alguna_vez =("preferente",  lambda x: int((x == 1).any())),
        )
        .reset_index()
    )

    # ── SEP: al año de egreso EM ────────────────────────────────────────────
    # Cruzamos SEP con el año de egreso de NEM y tomamos el registro SEP más
    # reciente que sea <= agno_egreso_em (el de ese año o el inmediatamente previo)
    sep_egr_merge = (
        sep_raw[["mrun", "agno_sep", "prioritario", "preferente"]]
        .merge(nem[["mrun", "agno_egreso_em"]], on="mrun", how="inner")
    )
    sep_egr_merge = sep_egr_merge[sep_egr_merge["agno_sep"] <= sep_egr_merge["agno_egreso_em"]]
    sep_egr = (
        sep_egr_merge.sort_values("agno_sep", ascending=False)
        .drop_duplicates("mrun")
        [["mrun", "prioritario", "preferente"]]
        .rename(columns={"prioritario": "prioritario_egreso",
                         "preferente":  "preferente_egreso"})
    )

    # ── Puntajes admisión 2021–2025 ─────────────────────────────────────────
    adm_frames = []
    for yr in range(2021, 2026):
        f = ie_dir / f"puntajes_admision_{yr}.parquet"
        if f.exists():
            adm_frames.append(pd.read_parquet(f))

    if adm_frames:
        adm_raw = pd.concat(adm_frames, ignore_index=True)
        adm_raw["mrun"]              = to_int(adm_raw["mrun"])
        adm_raw["ptje_nem_adm"]      = to_float(adm_raw["ptje_nem"])
        adm_raw["ptje_ranking_adm"]  = to_float(adm_raw["ptje_ranking"])
        adm_raw["promedio_notas_adm"] = to_float(adm_raw["promedio_notas"])
        for raw_col, new_col in [
            ("clec_max",  "clec_max_adm"),
            ("mate1_max", "mate1_max_adm"),
            ("hcsoc_max", "hcsoc_max_adm"),
            ("cien_max",  "cien_max_adm"),
        ]:
            if raw_col in adm_raw.columns:
                adm_raw[new_col] = to_float(adm_raw[raw_col])

        adm_cols = (
            ["mrun", "ptje_nem_adm", "ptje_ranking_adm", "promedio_notas_adm",
             "clec_max_adm", "mate1_max_adm", "hcsoc_max_adm", "cien_max_adm",
             "anyo_proceso", "nombre_unidad_educ"]
        )
        adm_cols = [c for c in adm_cols if c in adm_raw.columns]
        adm = (
            adm_raw[adm_cols]
            .sort_values("ptje_ranking_adm", ascending=False)
            .drop_duplicates("mrun")
            .rename(columns={"anyo_proceso": "anyo_proceso_adm",
                             "nombre_unidad_educ": "nom_rbd_adm"})
        )
    else:
        adm = pd.DataFrame(columns=["mrun"])

    # ── Fallback año nacimiento y género desde Matricula ────────────────────
    mat_fb = (
        mat_es[["mrun", "fec_nac_alu", "gen_alu"]]
        .drop_duplicates("mrun")
        .copy()
    )
    mat_fb["anio_nac_mat"]  = birth_year(mat_fb["fec_nac_alu"])
    mat_fb["gen_alu_mat"]   = to_int(mat_fb["gen_alu"])
    mat_fb = mat_fb[["mrun", "anio_nac_mat", "gen_alu_mat"]]

    # ── Ensamblar ───────────────────────────────────────────────────────────
    print("  [dim_estudiante] Ensamblando...")
    base = pd.DataFrame({"mrun": list(mruns)}, dtype="Int64")

    dim = (
        base
        .merge(nem,       on="mrun", how="left")
        .merge(egr,       on="mrun", how="left")
        .merge(sep_egr,   on="mrun", how="left")
        .merge(sep_any,   on="mrun", how="left")
        .merge(adm,       on="mrun", how="left")
        .merge(mat_fb,    on="mrun", how="left")
    )

    # Año nacimiento y género vienen exclusivamente de Matricula_Ed_Superior
    dim["anio_nac"] = dim["anio_nac_mat"]
    dim["gen_alu"]  = dim["gen_alu_mat"]

    # Consolidar cod_depe_egreso: Egresados_EM > NEM
    if "cod_depe" in dim.columns:
        cod_depe_egr = dim["cod_depe"].astype(str).str.strip().replace("nan", pd.NA)
        dim["cod_depe_egreso"] = cod_depe_egr.combine_first(dim["cod_depe_egreso"])

    # Consolidar rbd_egreso: NEM es la fuente principal (ya está en dim)
    # Nombre del establecimiento: SEP lookup > tabla admisión
    dim["nom_rbd_egreso"] = dim["rbd_egreso"].map(rbd_nom_lookup)
    if "nom_rbd_adm" in dim.columns:
        dim["nom_rbd_egreso"] = dim["nom_rbd_egreso"].fillna(dim["nom_rbd_adm"])

    # es_adultos: flag de NEM + verificación por cod_ense_egr
    dim["es_adultos"] = dim["es_adultos"].fillna(False)
    if "cod_ense_egr" in dim.columns:
        dim["es_adultos"] = dim["es_adultos"] | dim["cod_ense_egr"].isin(ADULT_ENSE_CODES)
    dim["es_adultos"] = dim["es_adultos"].astype("Int64")

    # Asegurar que las marcas SEP sean Int64 (0 / 1 / NA)
    for col in ["prioritario_egreso", "preferente_egreso",
                "prioritario_alguna_vez", "preferente_alguna_vez"]:
        if col not in dim.columns:
            dim[col] = pd.array([pd.NA] * len(dim), dtype="Int64")
        else:
            dim[col] = to_int(dim[col])

    # Eliminar columnas intermedias
    drop_cols = ["anio_nac_mat", "gen_alu_mat",
                 "cod_depe", "cod_ense_egr", "nom_rbd_adm"]
    dim = dim.drop(columns=[c for c in drop_cols if c in dim.columns])

    # Orden final de columnas
    ordered = [
        "mrun", "anio_nac", "gen_alu",
        # Origen escolar
        "agno_egreso_em", "rbd_egreso", "nom_rbd_egreso", "cod_depe_egreso",
        "cod_com_rbd", "nom_com_rbd", "cod_reg_rbd", "nom_reg_rbd_a", "rural_rbd",
        "es_adultos",
        # Rendimiento EM
        "nem", "percentil_egreso", "top_10", "decil_1_3",
        # SEP al egreso
        "prioritario_egreso", "preferente_egreso",
        # SEP cualquier año escolar
        "prioritario_alguna_vez", "preferente_alguna_vez",
        # Admisión
        "ptje_nem_adm", "ptje_ranking_adm", "promedio_notas_adm",
        "clec_max_adm", "mate1_max_adm", "hcsoc_max_adm", "cien_max_adm",
        "anyo_proceso_adm",
    ]
    present  = [c for c in ordered if c in dim.columns]
    leftover = [c for c in dim.columns if c not in ordered]
    return dim[present + leftover]


# ─── 2. trayectoria_es ────────────────────────────────────────────────────────

def build_trayectoria_es(
    ie_dir: Path,
    cod_inst: str,
    mruns: set,
    mat_es: pd.DataFrame,
) -> pd.DataFrame:
    """
    Una fila por mrun. Resume la trayectoria completa en educación superior
    usando Matricula_Ed_Superior (contiene todas las IEs donde estuvo el alumno)
    y Titulados_ES.
    """
    print("  [trayectoria_es] Calculando métricas...")

    focal_ie = int(cod_inst)

    mat = mat_es.copy()
    mat["mrun"]         = to_int(mat["mrun"])
    mat["cod_inst_num"] = to_int(mat["cod_inst"])
    mat["anio_ing"]     = to_int(mat["anio_ing_carr_ori"])
    mat["cod_carr_num"] = to_int(mat["cod_carrera"])

    # Filtrar años espurios antes de calcular mínimos
    mat = mat[mat["anio_ing"].between(1990, 2030)]

    # ── Métricas globales (todas las instituciones) ──────────────────────────
    agg_global = (
        mat.groupby("mrun")
        .agg(
            anio_primer_ingreso_es=("anio_ing",     "min"),
            n_instituciones_total  =("cod_inst_num", "nunique"),
        )
        .reset_index()
    )

    # Número de (institución, carrera) distintas
    n_carreras = (
        mat[["mrun", "cod_inst_num", "cod_carr_num"]]
        .dropna(subset=["cod_carr_num"])
        .drop_duplicates()
        .groupby("mrun")
        .size()
        .reset_index(name="n_carreras_total")
    )

    # ── Métricas en la IE focal ──────────────────────────────────────────────
    mat_focal = mat[mat["cod_inst_num"] == focal_ie]

    agg_focal = (
        mat_focal.groupby("mrun")
        .agg(
            anio_primer_ingreso_ie=("anio_ing",     "min"),
            n_carreras_en_ie       =("cod_carr_num", "nunique"),
        )
        .reset_index()
    )

    # ── Titulados ────────────────────────────────────────────────────────────
    tit_raw = pd.read_parquet(ie_dir / f"Titulados_ES_{cod_inst}.parquet")
    tit_raw["mrun"]         = to_int(tit_raw["mrun"])
    tit_raw["cod_inst_tit"] = to_int(tit_raw["cod_inst"])
    tit_raw["anio_tit"]     = to_int(tit_raw["cat_periodo"])

    # Titulados en la IE focal
    tit_focal = (
        tit_raw[tit_raw["cod_inst_tit"] == focal_ie]
        .sort_values("anio_tit")
        .groupby("mrun")["anio_tit"].min()
        .reset_index(name="anio_titulacion_ie")
    )
    tit_focal["titulado_en_ie"] = 1

    # Primer año de titulación en OTRA institución
    tit_otra_min = (
        tit_raw[tit_raw["cod_inst_tit"] != focal_ie]
        .groupby("mrun")["anio_tit"].min()
        .reset_index(name="anio_tit_otra_min")
    )

    # ── Ensamblar ────────────────────────────────────────────────────────────
    base = pd.DataFrame({"mrun": list(mruns)}, dtype="Int64")
    tray = (
        base
        .merge(agg_global,  on="mrun", how="left")
        .merge(n_carreras,  on="mrun", how="left")
        .merge(agg_focal,   on="mrun", how="left")
        .merge(tit_focal,   on="mrun", how="left")
        .merge(tit_otra_min, on="mrun", how="left")
    )

    tray["titulado_en_ie"] = to_int(tray["titulado_en_ie"]).fillna(0)

    # ¿Tenía matrícula en alguna IE antes de llegar a la focal?
    tray["tenia_matricula_previa_ie"] = (
        tray["anio_primer_ingreso_es"] < tray["anio_primer_ingreso_ie"]
    ).astype("Int64")

    # Años transcurridos entre primera ES y primer ingreso a la IE focal
    tray["n_anios_previos_es"] = (
        (tray["anio_primer_ingreso_ie"] - tray["anio_primer_ingreso_es"])
        .clip(lower=0)
        .astype("Int64")
    )

    # ¿Se había titulado en otra IE antes de ingresar a la focal?
    tray["titulado_previo_ie"] = (
        tray["anio_tit_otra_min"] < tray["anio_primer_ingreso_ie"]
    ).astype("Int64")

    drop_cols = ["anio_tit_otra_min"]
    tray = tray.drop(columns=[c for c in drop_cols if c in tray.columns])

    ordered = [
        "mrun",
        "anio_primer_ingreso_es", "anio_primer_ingreso_ie",
        "n_instituciones_total", "n_carreras_total", "n_carreras_en_ie",
        "tenia_matricula_previa_ie", "n_anios_previos_es",
        "titulado_en_ie", "anio_titulacion_ie", "titulado_previo_ie",
    ]
    present = [c for c in ordered if c in tray.columns]
    return tray[present]


# ─── 3. cohorte_ingreso ───────────────────────────────────────────────────────

def build_cohorte_ingreso(
    ie_dir: Path,
    cod_inst: str,
    mat_es: pd.DataFrame,
    dim_est: pd.DataFrame,
    tray_es: pd.DataFrame,
    gap_threshold: int = 2,
) -> pd.DataFrame:
    """
    Una fila por episodio de inscripción de un estudiante en una carrera de la IE focal.
    Un 'episodio' es una secuencia continua de años matriculado (sin ausencias > gap_threshold).
    Si un estudiante abandona y regresa después de gap_threshold años, se genera un segundo
    episodio (es_reingreso=1).

    Año de referencia: primer_anio_obs = primer año OBSERVADO en los datos para ese episodio.
    No se usa anio_ing_carr_ori como criterio primario porque ese campo tiene inconsistencias
    documentadas (valores cambian entre años del registro, swaps ori/act, centinelas).

    Nuevos campos de calidad/contexto:
      primer_anio_obs       — año base del episodio (ground-truth desde los datos)
      ultimo_anio_ep        — último año de matrícula en el episodio
      n_anios_en_ep         — años de matrícula observados en el episodio
      n_episodio            — 1 = primer episodio, 2 = reingreso, etc.
      es_reingreso          — 1 si n_episodio > 1
      anio_ori_declarado    — valor de anio_ing_carr_ori en los datos (referencia histórica)
      anio_ori_confiable    — 1 si anio_ori es consistente en todos los registros de esa carrera
      pre_cobertura_datos   — 1 si anio_ori sugiere ingreso anterior al inicio del dataset
      es_continuidad_plan_comun — 1 si forma_ingreso indica continuidad desde plan común
    """
    print("  [cohorte_ingreso] Construyendo cohortes de ingreso...")

    focal_ie = int(cod_inst)

    # ── Preparar datos de la IE focal ────────────────────────────────────────
    mat = mat_es.copy()
    mat["mrun"]         = to_int(mat["mrun"])
    mat["cod_inst_num"] = to_int(mat["cod_inst"])
    mat["year_int"]     = to_int(mat["year"])
    mat["cod_carr_num"] = to_int(mat["cod_carrera"])
    mat["anio_ori_raw"] = to_int(mat["anio_ing_carr_ori"])

    if "forma_ingreso" in mat.columns:
        mat["forma_ingreso_norm"] = normalize_forma_ingreso(
            mat["forma_ingreso"].fillna("").astype(str)
        )

    mat_focal = mat[
        (mat["cod_inst_num"] == focal_ie) &
        mat["year_int"].between(1990, 2030)
    ].copy()

    # Año mínimo con datos: usado para flag pre_cobertura_datos
    min_anio_dataset = int(mat_focal["year_int"].min())

    # ── Detección de episodios vía gaps de años ───────────────────────────────
    # Un episodio es una secuencia continua sin ausencias > gap_threshold años.
    # Trabajamos con una fila por (mrun, cod_carrera, year) para el análisis de gaps;
    # los registros descriptivos (forma_ingreso, sede, etc.) se recuperan después.
    key = ["mrun", "cod_carr_num"]
    mat_years = (
        mat_focal.dropna(subset=["cod_carr_num"])
        [key + ["year_int"]]
        .drop_duplicates()
        .sort_values(key + ["year_int"])
    )

    # Detectar nuevo episodio: primer año del grupo O gap > threshold respecto al año anterior
    mat_years["prev_year"] = mat_years.groupby(key)["year_int"].shift(1)
    mat_years["gap"]       = mat_years["year_int"] - mat_years["prev_year"]
    mat_years["new_ep"]    = mat_years["prev_year"].isna() | (mat_years["gap"] > gap_threshold)
    mat_years["n_episodio"] = (
        mat_years.groupby(key)["new_ep"].cumsum().astype(int)
    )

    # Metadatos por episodio
    episodes = (
        mat_years.groupby(key + ["n_episodio"])
        .agg(
            primer_anio_obs=("year_int", "min"),
            ultimo_anio_ep =("year_int", "max"),
            n_anios_en_ep  =("year_int", "count"),
        )
        .reset_index()
    )
    episodes["es_reingreso"] = (episodes["n_episodio"] > 1).astype("Int64")

    # ── Campos descriptivos: tomar del primer año de cada episodio ────────────
    # Join episodes → mat_focal por (mrun, cod_carr_num, year_int == primer_anio_obs)
    ep_lookup = episodes[key + ["n_episodio", "primer_anio_obs"]].rename(
        columns={"primer_anio_obs": "_year_target"}
    )
    desc_rows = (
        mat_focal
        .merge(ep_lookup, on=key, how="inner")
        .query("year_int == _year_target")
        .drop(columns=["_year_target"])
        .drop_duplicates(key + ["n_episodio"])
    )

    # ── Fiabilidad de anio_ori_declarado ─────────────────────────────────────
    # Para cada (mrun, cod_carrera): recoger todos los valores válidos de anio_ori_raw.
    # Si todos coinciden → confiable; si no → tomar el mínimo, marcar no confiable.
    ori_stats = (
        mat_focal[mat_focal["anio_ori_raw"].between(1990, 2030)]
        .groupby(key)["anio_ori_raw"]
        .agg(
            anio_ori_declarado=("min"),
            _n_valores_ori    =("nunique"),
        )
        .reset_index()
    )
    ori_stats["anio_ori_confiable"] = (ori_stats["_n_valores_ori"] == 1).astype("Int64")
    ori_stats = ori_stats.drop(columns=["_n_valores_ori"])

    # ── Flag continuidad plan común ───────────────────────────────────────────
    # ── Señal explícita de continuidad por forma_ingreso ─────────────────────
    if "forma_ingreso_norm" in mat_focal.columns:
        continuidad_explicita = (
            mat_focal[mat_focal["forma_ingreso_norm"] == "Continuidad Plan Común"]
            [key].drop_duplicates()
            .assign(_cont_explicita=1)
        )
    else:
        continuidad_explicita = pd.DataFrame(columns=key + ["_cont_explicita"])

    # ── Señal estructural: identificar carreras de Plan Común / Bachillerato ──
    # nivel_carrera_1 puede cambiar para la misma carrera entre años (ej. PC de Ing. pasa a
    # "Profesional Con Licenciatura" en años posteriores aunque siga siendo Plan Común).
    # Se complementa con nomb_carrera para cubrir esos casos.
    BACH_PATTERN = r"Bachillerato|Ciclo Inicial|Plan Común"
    BACH_NOMB_PATTERN = r"PLAN COMUN|PLAN COMÚN|BACHILLERATO|COLLEGE"
    bach_mask = (
        mat_focal["nivel_carrera_1"].str.contains(BACH_PATTERN, case=False, na=False)
        | mat_focal["nomb_carrera"].str.contains(BACH_NOMB_PATTERN, case=False, na=False)
    )

    # Último año observado de PC/Bach por mrun en esta IE
    ultimo_anio_bach = (
        mat_focal[bach_mask]
        .groupby("mrun")["year_int"].max()
        .reset_index(name="ultimo_anio_bach")
    )

    # ── Ensamblar cohorte ─────────────────────────────────────────────────────
    mat_cols = [
        "mrun", "cod_carr_num", "n_episodio",
        "nomb_carrera", "nivel_global", "nivel_carrera_1", "nivel_carrera_2",
        "tipo_plan_carr", "forma_ingreso_norm",
        "cod_sede", "nomb_sede", "comuna_sede", "region_sede",
        "jornada", "modalidad",
        "area_conocimiento", "area_carrera_generica",
        "valor_arancel", "valor_matricula",
        "sem_ing_carr_ori",
    ]
    mat_cols = [c for c in mat_cols if c in desc_rows.columns]
    cohorte = desc_rows[mat_cols].copy()

    cohorte = (
        cohorte
        .merge(episodes[key + ["n_episodio", "primer_anio_obs",
                               "ultimo_anio_ep", "n_anios_en_ep", "es_reingreso"]],
               on=key + ["n_episodio"], how="left")
        .merge(ori_stats,           on=key, how="left")
        .merge(continuidad_explicita, on=key, how="left")
        .merge(ultimo_anio_bach,    on="mrun", how="left")
    )

    # ── Flags de relación con Plan Común / Bachillerato ───────────────────────
    # es_bach_plan_comun: esta fila ES una carrera de PC/Bach
    nivel_match = (
        cohorte["nivel_carrera_1"].str.contains(BACH_PATTERN, case=False, na=False)
        if "nivel_carrera_1" in cohorte.columns
        else pd.Series(False, index=cohorte.index)
    )
    nomb_match = (
        cohorte["nomb_carrera"].str.contains(BACH_NOMB_PATTERN, case=False, na=False)
        if "nomb_carrera" in cohorte.columns
        else pd.Series(False, index=cohorte.index)
    )
    cohorte["es_bach_plan_comun"] = (nivel_match | nomb_match).astype("Int64")

    # Para filas que NO son PC: calcular gap entre salida de PC y entrada a esta carrera
    # gap_desde_bach = primer_anio_obs (esta carrera) - ultimo_anio_bach (del mrun)
    cohorte["gap_desde_bach"] = pd.array([pd.NA] * len(cohorte), dtype="Int64")
    no_bach_mask = cohorte["es_bach_plan_comun"] == 0
    cohorte.loc[no_bach_mask, "gap_desde_bach"] = (
        cohorte.loc[no_bach_mask, "primer_anio_obs"]
        - cohorte.loc[no_bach_mask, "ultimo_anio_bach"]
    ).astype("Int64")

    # vine_de_bach: el mrun tuvo PC en esta IE ANTES de esta carrera (gap >= 0)
    cohorte["vine_de_bach"] = (
        no_bach_mask
        & cohorte["ultimo_anio_bach"].notna()
        & (cohorte["gap_desde_bach"] >= 0)
    ).astype("Int64")
    # Para las filas que son PC, vine_de_bach no aplica → NA
    cohorte.loc[~no_bach_mask, "vine_de_bach"] = pd.NA

    # es_continuidad_plan_comun: señal EXPLÍCITA (forma_ingreso=2)
    #   OR INFERIDA (vine_de_bach=1 AND gap_desde_bach <= 2)
    # La señal inferida cubre los casos donde forma_ingreso es nulo pero el patrón
    # temporal es claro (pasó al año siguiente o con ≤2 años de diferencia).
    cont_explicita  = to_int(cohorte.get("_cont_explicita")).fillna(0) == 1
    cont_inferida   = (cohorte["vine_de_bach"] == 1) & (cohorte["gap_desde_bach"] <= 2)
    cohorte["es_continuidad_plan_comun"] = (
        (cont_explicita | cont_inferida) & no_bach_mask
    ).astype("Int64")
    # Para las filas de PC en sí, marcar como NA (no es continuidad de nada)
    cohorte.loc[~no_bach_mask, "es_continuidad_plan_comun"] = pd.NA

    cohorte = cohorte.drop(columns=["_cont_explicita", "ultimo_anio_bach"],
                           errors="ignore")

    # pre_cobertura_datos: anio_ori sugiere entrada antes de que arranquen los datos
    # Solo se marca cuando el primer año observado coincide con el mínimo del dataset
    # (no podemos ver el ingreso real) Y el anio_ori declarado es anterior a ese mínimo.
    cohorte["pre_cobertura_datos"] = (
        cohorte["anio_ori_declarado"].notna()
        & (cohorte["anio_ori_declarado"] < min_anio_dataset)
        & (cohorte["primer_anio_obs"] == min_anio_dataset)
    ).astype("Int64")

    # ── Campos de dim_estudiante ─────────────────────────────────────────────
    dim_cols = [
        "mrun", "anio_nac", "gen_alu", "agno_egreso_em",
        "cod_depe_egreso", "cod_com_rbd", "nom_com_rbd", "cod_reg_rbd",
        "rural_rbd", "es_adultos",
        "nem", "percentil_egreso", "top_10", "decil_1_3",
        "prioritario_egreso", "preferente_egreso",
        "prioritario_alguna_vez", "preferente_alguna_vez",
        "ptje_nem_adm", "ptje_ranking_adm",
    ]
    dim_join = dim_est[[c for c in dim_cols if c in dim_est.columns]]

    # ── Campos de trayectoria_es ─────────────────────────────────────────────
    tray_cols = [
        "mrun",
        "anio_primer_ingreso_es", "anio_primer_ingreso_ie",
        "n_instituciones_total", "n_carreras_total",
        "tenia_matricula_previa_ie", "n_anios_previos_es",
        "titulado_previo_ie",
    ]
    tray_join = tray_es[[c for c in tray_cols if c in tray_es.columns]]

    cohorte = (
        cohorte
        .merge(dim_join,  on="mrun", how="left")
        .merge(tray_join, on="mrun", how="left")
    )

    # ── Campos derivados ────────────────────────────────────────────────────
    cohorte["edad_al_ingreso"] = (
        cohorte["primer_anio_obs"] - cohorte["anio_nac"]
    ).astype("Int64")

    # Brecha entre egreso EM y primer año observado en esta carrera
    # ≤1 = ingreso directo | >1 = ingreso tardío | negativo o nulo = sin dato EM
    cohorte["brecha_post_em"]  = (cohorte["primer_anio_obs"] - cohorte["agno_egreso_em"]).astype("Int64")
    cohorte["ingreso_directo"] = (cohorte["brecha_post_em"] <= 1).astype("Int64")
    cohorte["ingreso_tardio"]  = (cohorte["brecha_post_em"] >  1).astype("Int64")

    # ¿Es el primer episodio del estudiante en esta IE (no es reingreso a otra carrera)?
    # Calculado directamente desde primer_anio_obs, sin depender del campo declarado.
    min_obs_ie = (
        cohorte.groupby("mrun")["primer_anio_obs"].min()
        .reset_index(name="_min_obs_ie")
    )
    cohorte = cohorte.merge(min_obs_ie, on="mrun", how="left")
    cohorte["es_primer_ingreso_ie"] = (
        (cohorte["primer_anio_obs"] == cohorte["_min_obs_ie"]) &
        (cohorte["n_episodio"] == 1)
    ).astype("Int64")
    cohorte = cohorte.drop(columns=["_min_obs_ie"])

    # ¿Tenía matrícula en ES antes de este episodio?
    cohorte["tenia_es_previa_este_ep"] = (
        cohorte["primer_anio_obs"] > cohorte["anio_primer_ingreso_es"]
    ).astype("Int64")

    # ── Ordenar columnas ─────────────────────────────────────────────────────
    front_cols = [
        "mrun", "cod_carr_num", "n_episodio",
        # Año de cohorte (ground-truth)
        "primer_anio_obs", "ultimo_anio_ep", "n_anios_en_ep",
        # Flags de calidad e interpretación del ingreso
        "es_reingreso", "es_continuidad_plan_comun", "pre_cobertura_datos",
        "anio_ori_declarado", "anio_ori_confiable",
    ]
    front_cols = [c for c in front_cols if c in cohorte.columns]
    rest_cols  = [c for c in cohorte.columns if c not in front_cols]
    return cohorte[front_cols + rest_cols]


# ─── 4. dim_carrera ───────────────────────────────────────────────────────────

def build_dim_carrera(
    cod_inst: str,
    mat_es: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tabla maestra de carreras de la IE focal.
    Grano: (cod_inst, cod_carrera, cod_sede) — una fila por oferta de carrera.

    Todos los atributos descriptivos se toman del año más reciente disponible,
    para capturar el estado actual de cada carrera (nombre vigente, duración,
    acreditación, clasificación CINE, etc.).

    Clave de join con cohorte_ingreso: cod_carr_num + cod_sede
    (cod_inst es constante dentro de la misma IE, así que filtrando el parquet
    de la IE ya no es necesario incluirlo en el ON).
    """
    print("  [dim_carrera] Construyendo tabla maestra de carreras...")

    focal_ie = int(cod_inst)

    mat = mat_es.copy()
    mat["cod_inst_num"] = to_int(mat["cod_inst"])
    mat["cod_carr_num"] = to_int(mat["cod_carrera"])
    # cod_sede se mantiene como string para que el join con cohorte_ingreso funcione sin cast
    mat["cod_sede_str"] = mat["cod_sede"].astype(str).str.strip()
    mat["anio_reg"]     = to_int(mat["year"])   # año del registro de matrícula

    # Filtrar a la IE focal y a años válidos
    mat = mat[
        (mat["cod_inst_num"] == focal_ie) &
        mat["anio_reg"].between(1990, 2030)
    ]

    # ── Atributos de carrera que queremos consolidar ─────────────────────────
    # Campos estables (texto, clasificaciones):
    #   tomar el valor más reciente (sort descendente + drop_duplicates)
    # Campos numéricos (duración):
    #   convertir a float, tomar más reciente no-nulo

    for col in ["dur_estudio_carr", "dur_total_carr", "dur_proceso_tit",
                "valor_arancel", "valor_matricula",
                "costo_proceso_titulacion", "costo_obtencion_titulo_diploma"]:
        if col in mat.columns:
            mat[col + "_num"] = to_float(mat[col])

    # Ordenar por año descendente para que "first" = más reciente
    mat_sorted = mat.sort_values("anio_reg", ascending=False)

    # Para la mayoría de atributos: tomar el valor del año más reciente
    # que ese (cod_inst, cod_carrera, cod_sede) tuvo registro.
    key = ["cod_inst_num", "cod_carr_num", "cod_sede_str"]

    attr_cols = [
        # Identidad
        "nomb_carrera", "nomb_sede", "nomb_inst",
        # Nivel y clasificación nacional
        "nivel_global", "nivel_carrera_1", "nivel_carrera_2",
        "area_carrera_generica", "area_conocimiento",
        # Clasificación internacional CINE
        "cine_f_13_area", "cine_f_13_subarea",
        "cine_f_97_area", "cine_f_97_subarea",
        # Formato
        "jornada", "modalidad", "tipo_plan_carr",
        # Geografía de la sede
        "comuna_sede", "provincia_sede", "region_sede",
        # Tipo de institución
        "tipo_inst_1", "tipo_inst_2", "tipo_inst_3",
        # Acreditación (varía año a año, tomamos la más reciente)
        "acreditada_carr", "acreditada_inst",
        "acre_inst_anio", "acre_inst_desde_hasta",
        # Vigencia y código DEMRE
        "vigencia_carrera", "codigo_demre", "codigo_unico",
        # Requisito de ingreso
        "requisito_ingreso",
    ]
    attr_cols = [c for c in attr_cols if c in mat_sorted.columns]

    # Primera pasada: atributos categóricos/texto (valor más reciente)
    dim = (
        mat_sorted[key + attr_cols]
        .drop_duplicates(subset=key, keep="first")
        .copy()
    )

    # Segunda pasada: duración y costos (valor numérico más reciente no-nulo)
    num_pairs = [
        ("dur_estudio_carr_num", "dur_estudio_carr"),
        ("dur_total_carr_num",   "dur_total_carr"),
        ("dur_proceso_tit_num",  "dur_proceso_tit"),
    ]
    for num_col, orig_col in num_pairs:
        if num_col in mat_sorted.columns:
            recent_num = (
                mat_sorted.dropna(subset=[num_col])
                [key + [num_col]]
                .drop_duplicates(subset=key, keep="first")
                .rename(columns={num_col: orig_col + "_n"})
            )
            dim = dim.merge(recent_num, on=key, how="left")

    # Rango de años en que la carrera estuvo activa
    year_range = (
        mat.groupby(key)["anio_reg"]
        .agg(anio_primer_registro="min", anio_ultimo_registro="max")
        .reset_index()
    )
    dim = dim.merge(year_range, on=key, how="left")

    # Renombrar columnas clave para consistencia con cohorte_ingreso
    # cod_sede_str es string → coincide con el tipo de cod_sede en cohorte_ingreso
    dim = dim.rename(columns={
        "cod_inst_num": "cod_inst",
        "cod_sede_str": "cod_sede",
    })

    # Orden final
    ordered = [
        "cod_inst", "cod_carr_num", "cod_sede",
        "nomb_inst", "nomb_carrera", "nomb_sede",
        "nivel_global", "nivel_carrera_1", "nivel_carrera_2",
        "area_carrera_generica", "area_conocimiento",
        "cine_f_13_area", "cine_f_13_subarea",
        "cine_f_97_area", "cine_f_97_subarea",
        "jornada", "modalidad", "tipo_plan_carr",
        "dur_estudio_carr_n", "dur_total_carr_n", "dur_proceso_tit_n",
        "comuna_sede", "provincia_sede", "region_sede",
        "tipo_inst_1", "tipo_inst_2", "tipo_inst_3",
        "acreditada_carr", "acreditada_inst",
        "acre_inst_anio", "acre_inst_desde_hasta",
        "vigencia_carrera", "codigo_demre", "codigo_unico",
        "requisito_ingreso",
        "anio_primer_registro", "anio_ultimo_registro",
    ]
    present  = [c for c in ordered if c in dim.columns]
    leftover = [c for c in dim.columns if c not in ordered]
    return dim[present + leftover]


# ─── main ─────────────────────────────────────────────────────────────────────

def process_ie(cod_inst: int) -> None:
    ie_dir    = BASE_DIR / str(cod_inst)
    out_dir   = ie_dir / "datos_analisis"
    out_dir.mkdir(exist_ok=True)
    cod_str   = str(cod_inst)

    print(f"\n{'='*60}")
    print(f"  IE {cod_inst}  →  {ie_dir.name}")
    print(f"{'='*60}")

    mruns = load_mruns(ie_dir, cod_str)
    print(f"  Población: {len(mruns):,} estudiantes únicos")

    # Cargar Matricula_Ed_Superior una sola vez (tabla más grande)
    print("\n  Cargando Matricula_Ed_Superior...")
    mat_es = pd.read_parquet(
        ie_dir / f"Matricula_Ed_Superior_{cod_str}.parquet"
    )
    # Normalizar mrun una vez
    mat_es["mrun"] = to_int(mat_es["mrun"])
    print(f"  Filas: {len(mat_es):,}")

    print("\n── dim_estudiante ──────────────────────────────────────────")
    dim_est = build_dim_estudiante(ie_dir, cod_str, mruns, mat_es)
    out = out_dir / "dim_estudiante.parquet"
    dim_est.to_parquet(out, index=False)
    print(f"  ✓ {out.name}  ({len(dim_est):,} filas × {dim_est.shape[1]} cols)")
    print(f"    cobertura NEM:          {dim_est['nem'].notna().sum():,} / {len(dim_est):,}")
    print(f"    con agno_egreso_em:     {dim_est['agno_egreso_em'].notna().sum():,}")
    print(f"    prioritario (egreso):   {(dim_est['prioritario_egreso']==1).sum():,}")
    print(f"    prioritario (algún año):{(dim_est['prioritario_alguna_vez']==1).sum():,}")
    print(f"    con puntajes admisión:  {dim_est['ptje_ranking_adm'].notna().sum():,}")

    print("\n── trayectoria_es ──────────────────────────────────────────")
    tray_es = build_trayectoria_es(ie_dir, cod_str, mruns, mat_es)
    out = out_dir / "trayectoria_es.parquet"
    tray_es.to_parquet(out, index=False)
    print(f"  ✓ {out.name}  ({len(tray_es):,} filas × {tray_es.shape[1]} cols)")
    print(f"    con trayectoria previa: {(tray_es['tenia_matricula_previa_ie']==1).sum():,}")
    print(f"    titulados en IE:        {(tray_es['titulado_en_ie']==1).sum():,}")

    print("\n── cohorte_ingreso ─────────────────────────────────────────")
    cohorte = build_cohorte_ingreso(ie_dir, cod_str, mat_es, dim_est, tray_es)
    out = out_dir / "cohorte_ingreso.parquet"
    cohorte.to_parquet(out, index=False)
    print(f"  ✓ {out.name}  ({len(cohorte):,} filas × {cohorte.shape[1]} cols)")
    if "primer_anio_obs" in cohorte.columns:
        años = cohorte["primer_anio_obs"].dropna().astype(int)
        print(f"    rango cohortes (obs):   {años.min()}–{años.max()}")
    print(f"    episodios primer ingreso:    {(cohorte['es_primer_ingreso_ie']==1).sum():,}")
    print(f"    reingresos:                  {(cohorte['es_reingreso']==1).sum():,}")
    print(f"    carreras PC/Bachillerato:    {(cohorte['es_bach_plan_comun']==1).sum():,}")
    n_cont_exp = ((cohorte['es_continuidad_plan_comun']==1) & (cohorte['vine_de_bach']==1) & (cohorte['forma_ingreso_norm']=='Continuidad Plan Común')).sum()
    n_cont_inf = ((cohorte['es_continuidad_plan_comun']==1) & ~(cohorte['forma_ingreso_norm']=='Continuidad Plan Común')).sum()
    print(f"    cont. PC explícita:          {n_cont_exp:,}  |  inferida (gap≤2): {n_cont_inf:,}")
    print(f"    pre cobertura datos:         {(cohorte['pre_cobertura_datos']==1).sum():,}")
    if "ingreso_directo" in cohorte.columns:
        # Base limpia: primer ingreso a IE, excluye PC en sí y continuidades
        base = cohorte[
            (cohorte["es_primer_ingreso_ie"]==1)
            & (cohorte["es_continuidad_plan_comun"]!=1)
            & (cohorte["es_bach_plan_comun"]!=1)
        ]
        n_dir = (base["ingreso_directo"] == 1).sum()
        n_tot = base["ingreso_directo"].notna().sum()
        if n_tot > 0:
            print(f"    ingreso directo EM (base limpia): {n_dir:,}/{n_tot:,} ({n_dir/n_tot*100:.1f}%)")

    print("\n── dim_carrera ─────────────────────────────────────────────")
    dim_carr = build_dim_carrera(cod_str, mat_es)
    out = out_dir / "dim_carrera.parquet"
    dim_carr.to_parquet(out, index=False)
    print(f"  ✓ {out.name}  ({len(dim_carr):,} filas × {dim_carr.shape[1]} cols)")
    if "nivel_global" in dim_carr.columns:
        print("    Carreras por nivel:")
        for nivel, n in dim_carr["nivel_global"].value_counts().items():
            print(f"      {nivel}: {n:,}")

    print(f"\n  Tablas guardadas en: {out_dir}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera tablas analíticas (dim_estudiante, trayectoria_es, "
                    "cohorte_ingreso) para una o más IEs de educación superior."
    )
    parser.add_argument(
        "cod_inst",
        type=int,
        nargs="+",
        help="Código(s) de institución (ej: 70  o  70 86 23)",
    )
    args = parser.parse_args()
    for ie in args.cod_inst:
        process_ie(ie)


if __name__ == "__main__":
    main()
