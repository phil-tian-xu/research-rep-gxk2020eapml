from utils.config_utils.config import CfgNode

_C = CfgNode()
cfg = _C
##############################
# ---- Part 0 : general ---- #
##############################
_C.GENERAL = CfgNode()
_C.GENERAL.NAME = ""
_C.GENERAL.TASK_TYPE = ""  # init_datahub, market_data_update


###########################
# ---- Part 1 : task ---- #
###########################
_C.TASK = CfgNode()

_C.TASK.INIT = CfgNode()
_C.TASK.INIT.MODE = ""  # validate, setup (default), reset
_C.TASK.INIT.ALLOW_DESTRUCTIVE = False
_C.TASK.INIT.REQUIRE_CONFIRM = True


###############################
# ---- Part 1 : DATABASE ---- #
###############################
_C.DATABASE = CfgNode()


#################################
# ---- Part 2: instruments ---- #
#################################
_C.INSTRUMENTS = CfgNode()
_C.INSTRUMENTS.MODE = "inline"  # inline or file
_C.INSTRUMENTS.MASTER_FILE = ""
_C.INSTRUMENTS.TASK_UNIVERSE_FILE = ""
_C.INSTRUMENTS.IDENTIFIER_TYPE = "ric"
_C.INSTRUMENTS.SYMBOLS = []
_C.INSTRUMENTS.SOURCE_IDENTIFIER_COLUMN = ""

##########################
# ---- Part 3: data ---- #
##########################
_C.DATA = CfgNode()
_C.DATA.DATASET = ""  # Prepared modeling dataset name; selected explicitly in run YAML.
_C.DATA.FORMAT = "csv"
_C.DATA.BATCH_SIZE = 10000

#####################
# 2.1 Data - Source #
#####################
_C.DATA.SOURCE = CfgNode()
_C.DATA.SOURCE.NAME = ""

##########################
# 2.2 Data - Time Series #
##########################
_C.DATA.TIME_SERIES = CfgNode()
_C.DATA.TIME_SERIES.IS_ON = False
_C.DATA.TIME_SERIES.FIELDS = []
_C.DATA.TIME_SERIES.START_DATE = ""
_C.DATA.TIME_SERIES.END_DATE = ""
_C.DATA.TIME_SERIES.FREQUENCY = "daily"
_C.DATA.TIME_SERIES.MAX_BATCH = 50

##########################
# 2.2 Data - Static Data #
##########################
_C.DATA.STATIC_DATA = CfgNode()
_C.DATA.STATIC_DATA.IS_ON = False


# Shared run settings for synthetic and real datasets. Copied showcase sections are retained.
# Empty grids / None thresholds require an explicit run choice; they are
# not filled with guessed author settings. See model_demo.yaml for toy-run choices.
_C.RUN = CfgNode({
    "FITTING": "batch",  # Parquet only: 'dense' loads whole annual windows (PCR).
    "SPLIT": {
        "FIRST_TRAIN_YEAR": 1957,
        "FIRST_TEST_YEAR": 1987,
        "LAST_TEST_YEAR": 2016,
        "VALIDATION_YEARS": 12,
    },
    "MODELS": {
        "ENABLED": [],
        "FIT_INTERCEPT": True,
        "STANDARDIZE": False,
        "LINEAR_SOLVER": {"MAX_ITER": 5000, "TOL": 1e-8, "INITIAL_STEP": 1.0},
        "OLS_HUBER": {"HUBER_DELTA": None, "HUBER_QUANTILE": None, "SOLVER": "apg"},
        "ENET_HUBER": {"HUBER_DELTA": None, "RHO": 0.5, "PENALTY_GRID": []},
        "PCR": {"COMPONENT_GRID": []},
        "PLS": {"COMPONENT_GRID": [], "TOL": 1e-12},
    },
    "EVALUATION": {
        "VALIDATION": {"NAME": "mse", "DELTA": None},
        "TEST": {"NAME": "mse", "DELTA": None},
    },
    # Empty means return results in memory and print the summary only.
    "OUTPUT_DIR": "",
    "REPORT_TABLE1_FIGURE3": False,
})

