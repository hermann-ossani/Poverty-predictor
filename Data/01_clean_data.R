# =============================================================================
# 01_clean_data.R
# Costa Rican Household Poverty -- one clean row per household
#
# Input : costa-rican-household-poverty-prediction/train.csv
#         (9,557 rows = persons, 143 columns)
# Output: data/costa_rica_clean.RData with three objects:
#   costa_rica        one row per household, readable categories (factors),
#                     43 predictors + `poor`. Used for the exploratory analysis.
#   costa_rica_model  the same households, ready for modelling: numeric columns,
#                     one 0/1 column per category, rare 0/1 columns removed
#                     (fewer than 30 households), + `poor`.
#   removed_cols      the 0/1 columns that were removed and why
#
# Folder layout (Desktop/WU/Statistical and Machine Learning/Project):
#   Project.Rproj
#   01_clean_data.R                            <- this script, next to the .Rproj
#   costa-rican-household-poverty-prediction/  <- extracted zip with train.csv
#   data/                                      <- created by this script
#
# IMPORTANT: open Project.Rproj in RStudio first, so that the working
# directory is the "Project" folder (check with getwd()).
#
# Base R only, no packages needed.
# =============================================================================

min_n <- 30      # a 0/1 column needs at least this many households in each value

# ---- 0. Read the data -------------------------------------------------------
data_path <- "costa-rican-household-poverty-prediction/train.csv"
if (!file.exists(data_path)) {
  # fallback: look for train.csv anywhere inside the project folder
  found <- list.files(pattern = "^train\\.csv$", recursive = TRUE)
  if (length(found) == 0) {
    stop("train.csv not found. Open Project.Rproj in RStudio (getwd() must be ",
         "the Project folder) and make sure the zip is extracted.")
  }
  data_path <- found[1]
}
raw <- read.csv(data_path, stringsAsFactors = FALSE)
dim(raw)                                   # 9557 143


# ---- 1. One row per household: keep the household head ----------------------
# Each row is a PERSON, but the poverty group is a HOUSEHOLD outcome. Members of
# one household share the same household variables, so using all rows would put
# the same household in both the training and the test data (leakage).
# We keep the household head (parentesco1 == 1). Only heads are scored in the
# original competition as well.
heads_per_hh <- tapply(raw$parentesco1, raw$idhogar, sum)
table(heads_per_hh)                        # 15 households without head, 2973 with one

hh <- raw[raw$parentesco1 == 1, ]          # households without a head drop out
stopifnot(!any(duplicated(hh$idhogar)))    # exactly one row per household
nrow(hh)                                   # 2973
# Note: in 85 households the members have different Target values; we use the
# value of the head.


# ---- 2. Text "yes"/"no" -> numbers ------------------------------------------
# dependency, edjefe and edjefa contain the text "yes"/"no" next to numbers.
# yes = 1, no = 0 (checked against the squared columns SQBdependency and
# SQBedjefe, which show 1 for "yes" and 0 for "no").
to_num <- function(x) {
  x[x == "yes"] <- "1"
  x[x == "no"]  <- "0"
  as.numeric(x)
}
hh$dependency <- to_num(hh$dependency)
hh$edjefe     <- to_num(hh$edjefe)
hh$edjefa     <- to_num(hh$edjefa)


# ---- 3. Structurally missing values -----------------------------------------
# v2a1 (monthly rent) is missing for households that do not pay rent (e.g.
# owners): no rent = 0. The housing tenure still tells the type of tenure.
# v18q1 (number of tablets) is missing exactly when v18q == 0 (no tablet): 0.
hh$v2a1[is.na(hh$v2a1)]   <- 0
hh$v18q1[is.na(hh$v18q1)] <- 0


# ---- 4. Remaining missing values --------------------------------------------
# rez_esc (years behind in school) is only defined for school-age people:
# missing for 2972 of 2973 heads -> dropped in step 6.
# meaneduc is missing for 3 heads -> we drop these 3 households.
hh <- hh[!is.na(hh$meaneduc), ]
nrow(hh)                                   # 2970


# ---- 5. Binary target -------------------------------------------------------
# poor     = extreme or moderate poverty (original Target 1 or 2)
# not_poor = vulnerable or non-vulnerable (original Target 3 or 4)
table(hh$Target)                           # 222 / 442 / 355 / 1951
poor <- ifelse(hh$Target %in% c(1, 2), "poor", "not_poor")
hh$poor <- factor(poor, levels = c("not_poor", "poor"))   # "poor" = event
# The original 4-class Target is removed: left among the predictors it would
# predict `poor` perfectly (leakage).
hh$Target <- NULL


# ---- 6. Drop redundant or uninformative columns -----------------------------
rownames(hh) <- hh$idhogar                 # keep the household id as row name

drop_cols <- c(
  "idhogar", "Id",                         # identifiers, not predictors
  "rez_esc",                               # missing for all but one head
  "tamhog", "hogar_total", "r4t3",         # (almost) identical to hhsize
  "female",                                # = 1 - male
  "area2",                                 # = 1 - area1
  "edjefe", "edjefa",                      # = years of schooling of the head
  "agesq",                                 # = age^2
  grep("^SQB", names(hh), value = TRUE),   # squared copies of other columns
  # exact sums or functions of other columns (no information is lost):
  "r4h3", "r4m3", "r4t1", "r4t2",          # sums of r4h1, r4h2, r4m1, r4m2
  "hogar_adul",                            # = hhsize - hogar_nin
  "v18q",                                  # = 1 if the number of tablets > 0
  "mobilephone"                            # = 1 if the number of mobile phones > 0
)

# columns with a single value for all heads (e.g. parentesco*: the head is
# never spouse, child, ...) carry no information
const_cols <- names(hh)[sapply(hh, function(x) length(unique(x)) == 1)]
const_cols                                 # elimbasu5, estadocivil1, parentesco1-12

hh <- hh[, !(names(hh) %in% c(drop_cols, const_cols))]


# ---- 7. Merge 0/1 columns of the same characteristic into one categorical variable
# The survey stores each characteristic as several 0/1 columns (one per
# category), e.g. paredblolad, paredmad, ... for the wall material.
# We merge each group into ONE variable with readable labels.
from_dummies <- function(data, cols, labels) {
  m <- as.matrix(data[cols])
  stopifnot(all(rowSums(m) <= 1))          # a household is never in two categories
  code <- max.col(m, ties.method = "first")
  none <- rowSums(m) == 0                  # in none of the listed categories
  if (any(none)) {
    code[none] <- length(cols) + 1
    labels <- c(labels, "not_recorded")
  }
  factor(code, levels = seq_along(labels), labels = labels)
}

groups <- list(
  head_marital_status = list(
    cols   = c("estadocivil2", "estadocivil3", "estadocivil4", "estadocivil5",
               "estadocivil6", "estadocivil7"),
    labels = c("free_or_coupled_union", "married", "divorced", "separated",
               "widowed", "single")),
  head_education_level = list(
    cols   = paste0("instlevel", 1:9),
    labels = c("none", "incomplete_primary", "complete_primary",
               "incomplete_secondary", "complete_secondary",
               "incomplete_technical_secondary", "complete_technical_secondary",
               "undergraduate_or_higher", "postgraduate")),
  housing_tenure = list(
    cols   = paste0("tipovivi", 1:5),
    labels = c("owned_fully_paid", "owned_paying_instalments", "rented",
               "precarious", "other_assigned_or_borrowed")),
  wall_material = list(
    cols   = c("paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad",
               "paredzinc", "paredfibras", "paredother"),
    labels = c("block_or_brick", "socket_wood_zinc_asbestos",
               "prefabricated_or_cement", "waste_material", "wood", "zinc",
               "natural_fibers", "other")),
  wall_quality = list(
    cols   = paste0("epared", 1:3),
    labels = c("bad", "regular", "good")),
  roof_material = list(
    cols   = c("techozinc", "techoentrepiso", "techocane", "techootro"),
    labels = c("zinc_metal_foil", "fiber_cement_mezzanine", "natural_fibers",
               "other")),
  roof_quality = list(
    cols   = paste0("etecho", 1:3),
    labels = c("bad", "regular", "good")),
  floor_material = list(
    cols   = c("pisomoscer", "pisocemento", "pisoother", "pisonatur",
               "pisonotiene", "pisomadera"),
    labels = c("mosaic_ceramic_terrazzo", "cement", "other", "natural_material",
               "no_floor", "wood")),
  floor_quality = list(
    cols   = paste0("eviv", 1:3),
    labels = c("bad", "regular", "good")),
  water_supply = list(
    cols   = c("abastaguadentro", "abastaguafuera", "abastaguano"),
    labels = c("inside_dwelling", "outside_dwelling", "none")),
  electricity_source = list(
    cols   = c("public", "planpri", "noelec", "coopele"),
    labels = c("public_grid", "private_plant", "none", "cooperative")),
  toilet_type = list(
    cols   = c("sanitario1", "sanitario2", "sanitario3", "sanitario5",
               "sanitario6"),
    labels = c("none", "sewer_or_cesspool", "septic_tank",
               "black_hole_or_latrine", "other")),
  cooking_energy = list(
    cols   = paste0("energcocinar", 1:4),
    labels = c("none_no_kitchen", "electricity", "gas", "wood_charcoal")),
  rubbish_disposal = list(
    cols   = c("elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu6"),
    labels = c("tanker_truck", "buried", "burned", "thrown_in_unoccupied_space",
               "other")),
  region = list(
    cols   = paste0("lugar", 1:6),
    labels = c("Central", "Chorotega", "Pacifico_central", "Brunca",
               "Huetar_Atlantica", "Huetar_Norte"))
)

all_group_cols <- unlist(lapply(groups, function(g) g$cols))
stopifnot(all(all_group_cols %in% names(hh)))

for (g in names(groups)) {
  hh[[g]] <- from_dummies(hh, groups[[g]]$cols, groups[[g]]$labels)
}
hh <- hh[, !(names(hh) %in% all_group_cols)]
# Households that belong to none of the listed categories get the extra label
# "not_recorded": roof_material (19 households), electricity_source (4),
# head_education_level (1). They are hidden missing values: all their 0/1
# columns of that characteristic are 0.
sapply(hh[sapply(hh, is.factor)], function(f) sum(f == "not_recorded"))


# ---- 8. Explicit variable names ---------------------------------------------
# old name in the data = new, explicit name
new_names <- c(
  age = "head_age",                    male = "head_male",
  escolari = "head_years_schooling",   dis = "head_disabled",
  hhsize = "household_size",           tamviv = "persons_living_in_dwelling",
  hogar_nin = "n_children_0_19",       hogar_mayor = "n_seniors_65plus",
  r4h1 = "n_males_under12",            r4h2 = "n_males_12plus",
  r4m1 = "n_females_under12",          r4m2 = "n_females_12plus",
  dependency = "dependency_rate",      meaneduc = "mean_years_schooling_adults",
  rooms = "n_rooms",                   bedrooms = "n_bedrooms",
  overcrowding = "persons_per_bedroom",
  hacdor = "overcrowded_bedrooms",     hacapo = "overcrowded_rooms",
  v2a1 = "monthly_rent",               cielorazo = "has_ceiling",
  v14a = "has_toilet",                 refrig = "has_refrigerator",
  computer = "has_computer",           television = "has_television",
  v18q1 = "n_tablets",                 qmobilephone = "n_mobile_phones",
  area1 = "urban"
)
stopifnot(all(names(new_names) %in% names(hh)))
names(hh)[match(names(new_names), names(hh))] <- new_names


# ---- 9. Order the columns by theme ------------------------------------------
col_order <- c(
  # household head
  "head_age", "head_male", "head_marital_status", "head_education_level",
  "head_years_schooling", "head_disabled",
  # household composition
  "household_size", "persons_living_in_dwelling", "n_children_0_19",
  "n_seniors_65plus", "n_males_under12", "n_males_12plus", "n_females_under12",
  "n_females_12plus", "dependency_rate", "mean_years_schooling_adults",
  # dwelling
  "n_rooms", "n_bedrooms", "persons_per_bedroom", "overcrowded_bedrooms",
  "overcrowded_rooms", "housing_tenure", "monthly_rent", "wall_material",
  "wall_quality", "roof_material", "roof_quality", "has_ceiling",
  "floor_material", "floor_quality",
  # services
  "water_supply", "electricity_source", "toilet_type", "has_toilet",
  "cooking_energy", "rubbish_disposal",
  # assets
  "has_refrigerator", "has_computer", "has_television", "n_tablets",
  "n_mobile_phones",
  # location
  "region", "urban",
  # target
  "poor"
)
stopifnot(setequal(col_order, names(hh)))
hh <- hh[, col_order]                      # readable version (factors)


# ---- 10. Modelling data: one 0/1 column per category, rare columns removed ---
# Each categorical variable becomes one 0/1 column per category, named
# <variable>_<category> (e.g. wall_material_wood). No category is left out as a
# reference: the penalty of the lasso/ridge handles the redundancy.
cat_vars <- setdiff(names(hh)[sapply(hh, is.factor)], "poor")

dummies <- do.call(cbind, lapply(cat_vars, function(v) {
  f <- hh[[v]]
  m <- sapply(levels(f), function(l) as.numeric(f == l))
  colnames(m) <- paste(v, levels(f), sep = "_")
  m
}))

costa_rica_model <- cbind(hh[, setdiff(names(hh), c(cat_vars, "poor"))], dummies)
costa_rica_model$poor <- hh$poor           # target in the last column
rownames(costa_rica_model) <- rownames(hh)

# Remove 0/1 columns with too few observations: one of the two values (0 or 1)
# occurs in fewer than `min_n` households, so its effect cannot be estimated
# reliably. This covers rare categories (few 1s) and almost constant columns
# (few 0s). No household is removed: households of a removed category simply
# have 0 in all remaining columns of that variable. The target is not used.
x_names <- setdiff(names(costa_rica_model), "poor")
is01    <- sapply(costa_rica_model[x_names], function(x) all(x %in% c(0, 1)))
bin_names <- x_names[is01]
n_ones  <- colSums(costa_rica_model[bin_names])
n_minor <- pmin(n_ones, nrow(costa_rica_model) - n_ones)   # size of the rarer value
removed_cols <- data.frame(column = bin_names[n_minor < min_n],
                           households_in_rarer_value = as.integer(n_minor[n_minor < min_n]),
                           row.names = NULL)
removed_cols <- removed_cols[order(removed_cols$households_in_rarer_value), ]
rownames(removed_cols) <- NULL
removed_cols                               # columns removed (fewer than 30 households)

costa_rica_model <- costa_rica_model[, !(names(costa_rica_model) %in% removed_cols$column)]


# ---- 11. Checks and save ----------------------------------------------------
costa_rica <- hh                           # readable version, for the exploratory analysis

stopifnot(!anyNA(costa_rica), !anyNA(costa_rica_model))   # no missing values
stopifnot(identical(rownames(costa_rica), rownames(costa_rica_model)))   # same households
stopifnot(identical(costa_rica$poor, costa_rica_model$poor))             # same target
stopifnot(all(sapply(costa_rica_model[setdiff(names(costa_rica_model), "poor")], is.numeric)))

dim(costa_rica)                            # 2970 rows, 44 columns (43 predictors + poor)
dim(costa_rica_model)                      # 2970 rows, (number of numeric columns + dummies + poor)
table(costa_rica$poor)                     # 2306 not_poor / 664 poor
prop.table(table(costa_rica$poor))         # about 22% poor
table(sapply(costa_rica[-ncol(costa_rica)], function(x)    # 15 categorical, 10 binary, 18 numeric
  if (is.factor(x)) "categorical" else if (all(x %in% c(0, 1))) "binary 0/1" else "numeric"))

dir.create("data", showWarnings = FALSE)   # output folder inside the project
save(costa_rica, costa_rica_model, removed_cols,
     file = "data/costa_rica_clean.RData")  # keeps factors and row names
# load it in any other script with:
#   load("data/costa_rica_clean.RData")    # creates costa_rica, costa_rica_model, removed_cols
