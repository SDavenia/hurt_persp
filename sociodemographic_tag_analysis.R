# Analyze both sociodemographic and tag jointly
library(dplyr)
library(tidyr)
library(jsonlite)
library(purrr)
library(lme4)
library(glmmTMB)
library(performance)
library(fastDummies)
library(ggeffects)
library(ggplot2)

df = read.csv("data/extracted_examples/extracted_examples_lemma_with_pos_mapped.csv")

########## Extract sociodemographics
# Function to safely extract values from the sociodemographics column
extract_demo <- function(x, var) {
  if (is.na(x) || x == "" || x == "None") return(NA)
  
  # Convert single quotes to double quotes to make it valid JSON
  x_clean <- gsub("'", '"', x)
  
  parsed <- tryCatch(fromJSON(x_clean), error = function(e) NULL)
  
  if (is.null(parsed) || is.null(parsed[[var]]) || parsed[[var]] %in% c("nan", "None")) {
    return(NA)
  }
  
  return(parsed[[var]])
}

remove_rare_levels <- function(f, min_count = 2000) {
  # Count occurrences
  tbl <- table(f)
  # Keep only levels with >= min_count
  keep_levels <- names(tbl)[tbl >= min_count]
  # Set values not in keep_levels to NA
  f_clean <- factor(ifelse(f %in% keep_levels, as.character(f), NA),
                    levels = keep_levels)
  return(f_clean)
}

df_tags_merged <- df %>%
  filter(!is.na(tag), tag != "{}", tag != "") %>%
  mutate(
    tag_clean = gsub("[{}' ]", "", tag),
    label = ifelse(label == "hurtful", 1, 0)
  ) %>%
  dummy_cols(select_columns = "tag_clean", split = ",") %>%
  rename_with(~gsub("tag_clean_", "is_", .x), starts_with("tag_clean_")) %>%
  group_by(text_id, annotator_id) %>%
  summarise(
    text  = first(text),
    label = first(label),
    sociodemographics_mapped = first(sociodemographics_mapped),
    across(starts_with("is_"), max),
    .groups = "drop"
  )

df_wide <- df_tags_merged %>%
  mutate(
    race      = map_chr(sociodemographics_mapped, ~ extract_demo(.x, "race")),
    gender    = map_chr(sociodemographics_mapped, ~ extract_demo(.x, "gender")),
    age       = map_chr(sociodemographics_mapped, ~ as.character(extract_demo(.x, "age"))),
    ideology  = map_chr(sociodemographics_mapped, ~ extract_demo(.x, "ideology")),
    education = map_chr(sociodemographics_mapped, ~ extract_demo(.x, "education"))
  ) %>%
  filter(
    !is.na(race),
    !is.na(gender),
    !is.na(age)
  ) %>%
  select(-sociodemographics_mapped) %>%
  select(where(~ !all(. == 0)))


# Convert to factors
df_wide <- df_wide %>%
  mutate(
    text_id = factor(text_id),
    annotator_id = factor(annotator_id),
    race = factor(race),
    gender = factor(gender),
    age = factor(age)
  )


# Remove rare sociodemographic levels
df_wide_norare_socdem <- df_wide %>%
  mutate(
    race   = remove_rare_levels(factor(race)),
    gender = remove_rare_levels(factor(gender)),
    age    = remove_rare_levels(factor(age)),
    text_id = factor(text_id),
    annotator_id = factor(annotator_id)
  ) %>%
  filter(
    !is.na(race),
    !is.na(gender),
    !is.na(age)
  )

# Assign reference level for each sociodemographic variable: -> man, 18-24, white
df_wide_norare_socdem$gender <- relevel(df_wide_norare_socdem$gender, ref='man')
df_wide_norare_socdem$age <- relevel(df_wide_norare_socdem$age, ref='18-24')
df_wide_norare_socdem$race <- relevel(df_wide_norare_socdem$race, ref='white')

# Print retained levels of sociodemographics
unique(df_wide_norare_socdem$gender)
unique(df_wide_norare_socdem$age)
unique(df_wide_norare_socdem$race)

########## Analysis based only on sociodemographic factors ########## 
# Remove rare levels at the sociodemographics
### Null model ###
model0_socdem <- glmmTMB(
  label ~ 1 + 
    (1 | text_id) + 
    (1 | annotator_id),
  data = df_wide_norare_socdem,
  family = binomial()
)

### Sociodemographic model ####
model1_socdem <- glmmTMB(
  label ~ race + gender + age + 
    (1 | text_id) + 
    (1 | annotator_id),
  data = df_wide_norare_socdem,
  family = binomial()
)

### Sociodemographic intersectional model
model2_socdem <- glmmTMB(
  label ~ race * ( gender + age ) + 
    (1 | text_id) + 
    (1 | annotator_id),
  data = df_wide_norare_socdem,
  family = binomial()
)

summary(model2_socdem)

r2_0_socdem <- r2(model0_socdem)
r2_1_socdem <- r2(model1_socdem)
r2_2_socdem <- r2(model2_socdem)

r2_table_socdem <- data.frame(
  Model = c("Null", "Sociodemographic", "Race-Intersectional"),
  Marginal_R2 = c(r2_0_socdem$R2_marginal, r2_1_socdem$R2_marginal, r2_2_socdem$R2_marginal),
  Conditional_R2 = c(r2_0_socdem$R2_conditional, r2_1_socdem$R2_conditional, r2_2_socdem$R2_conditional)
)

print(r2_table_socdem)
#### Brief comment 
# The sociodemographic variables individually account for 3.7% of variability while they account for for 6% of variability
#  when considered in intersection, which agrees with previous findings pointing to their limited capacity to explain variability (Hu and Collier)
#  this is considerably lower than the percentage of variability which is explained by the random effects which is 79% in the sociodemographic model 
#  and 77% in the intersectional one.

# Visualize marginal effects of sociodemographics, 

# Intersectional Marginal Effects of Race and Age on Offensiveness Perception.
# The plot displays the predicted probability of a text being rated as offensive across race and age group for an average text.

pr_socdem <- predict_response(model2_socdem, 
                              terms = c("race", "age"), 
                              margin = "average") 

plot(pr_socdem) + 
  labs(title = "Adjusted Predicted Offensiveness",
       y = "Prob. of Offensive Rating", 
       x = "Race Group") +
  theme_minimal()

pr_socdem <- predict_response(model2_socdem, 
                              terms = c("race", "gender"), 
                              margin = "average") 

plot(pr_socdem) + 
  labs(title = "Adjusted Predicted Offensiveness",
       y = "Prob. of Offensive Rating", 
       x = "Gender Group") +
  theme_minimal()



### Analysis based on tags
category_cols <- names(df_wide_norare_socdem)[grepl("^is_", names(df_wide_norare_socdem))]

formula_str <- paste("label ~", paste(category_cols, collapse = " + "), 
                     "+ (1 | annotator_id) + (1 | text_id)")

### Null model ###
model0_tags <- glmmTMB(label ~ 1 + (1|annotator_id) + (1 | text_id),
                  data=df_wide_norare_socdem,
                  family=binomial)

### Model with individual tags as fixed effects ### 
model1_tags <- glmmTMB(as.formula(formula_str), 
                       data = df_wide_norare_socdem, 
                       family = binomial)

r2_0_tags <- r2(model0_tags)
r2_1_tags <- r2(model1_tags)

r2_table_tags <- data.frame(
  Model = c("Null", "Tags"),
  Marginal_R2 = c(r2_0_tags$R2_marginal, r2_1_tags$R2_marginal),
  Conditional_R2 = c(r2_0_tags$R2_conditional, r2_1_tags$R2_conditional)
)

r2_table_tags
#### Brief comment 
# The tags variables account for approximately 5.6% of the variability while the random effects still account for majority of the variability.
# This is larger than what we saw before.

summary(model1_tags) # To analyze most relevant tags


#### Visualize
# 1. Define the list of category terms from your model
terms_to_plot <- c("is_an", "is_asm", "is_om", "is_cds", "is_pr", "is_svp", 
                   "is_asf", "is_dmc", "is_ps", "is_rci", "is_ddp", "is_qas", 
                   "is_pa", "is_or", "is_ddf", "is_is", "is_re")

# 2. Get predicted responses for each term being 'active' (1)
# margin = "average" calculates the marginal effect over the population
preds <- lapply(terms_to_plot, function(x) {
  res <- predict_response(model1_tags, terms = x, margin = "average")
  # Filter only for the '1' (active) state of the indicator
  res <- as.data.frame(res) %>% filter(x == 1) 
  res$term_name <- x
  return(res)
})

plot_data <- bind_rows(preds)

# 3. Create the plot
ggplot(plot_data, aes(x = predicted, y = reorder(term_name, predicted))) +
  geom_point(size = 3, color = "steelblue") +
  geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
  labs(title = "Adjusted Predicted Probabilities by Category",
       subtitle = "Calculated using the Average Marginal Effect method",
       y = "Category Indicator",
       x = "Predicted Probability of Offensive Rating") +
  theme_minimal()

### Joint analysis of sociodemographic and tags ### 
formula_str_socdemtags <- paste("label ~", paste(category_cols, collapse = " + "), 
                     "+ race * (gender + age) + (1 | annotator_id) + (1 | text_id)")

#formula_str_socdemtags <- paste("label ~", paste(category_cols, collapse = " + "), 
#                                "+ race + gender + age + (1 | annotator_id) + (1 | text_id)")



model0_socdemtags <- glmmTMB(label ~ 1 + (1|annotator_id) + (1 | text_id),
                             data=df_wide_norare_socdem,
                             family = binomial)
model1_socdemtags <- glmmTMB(as.formula(formula_str_socdemtags), 
                             data = df_wide_norare_socdem, 
                             family = binomial)
summary(model1_socdemtags)
# We have too many categories to consider all interactions between categories and sociodemographic traits.
# Choose which ones we want to include
# OPTION 1: Hypothesis / Linguistic driven -> TODO: not enough time.
# OPTION 2: Keep those with highest signal in absolute value.
# is_pr -> 20.07
# is_ps -> 32.54
# is_or -> 1.17
# is_re -> 5.36 
formula_str_interax <- paste(
  "label ~", 
  paste(category_cols, collapse = " + "), # All tags as controls
  "+ race * (gender + age)",              # Base intersectionality
  "+ (is_ps + is_pr + is_or + is_re) * race", # Sensitivity by Race
  "+ (is_ps + is_pr + is_or + is_re) * age",  # Sensitivity by Age
  "+ (1 | annotator_id) + (1 | text_id)"
)

model2_socdemtags <- glmmTMB(as.formula(formula_str_interax),
                             data = df_wide_norare_socdem, 
                             family = binomial)

r2_0_socdemtags <- r2(model0_socdemtags)
r2_1_socdemtags <- r2(model1_socdemtags)
r2_2_socdemtags <- r2(model2_socdemtags)

r2_table_socdemtags <- data.frame(
  Model = c("Null", "SocdemTags", "SocdemTags_interaction"),
  Marginal_R2 = c(r2_0_socdemtags$R2_marginal, r2_1_socdemtags$R2_marginal, r2_2_socdemtags$R2_marginal),
  Conditional_R2 = c(r2_0_socdemtags$R2_conditional, r2_1_socdemtags$R2_conditional, r2_2_socdemtags$R2_conditional)
)

r2_table_socdemtags
summary(model2_socdemtags)

anova(model1_socdemtags, model2_socdemtags)


#### Visualize
library(ggeffects)
library(ggplot2)

# Visualize how the effect of "is_ps" changes across Race then do the same across all age
pr_ps_race <- predict_response(model2_socdemtags, 
                               terms = c("is_ps [0,1]", "race"), 
                               margin = "average")

plot(pr_ps_race) +
  labs(title = "Effect of 'is_ps' by Race",
       subtitle = "Significant interaction found for Black participants",
       y = "Prob. of Offensive Rating",
       x = "is_ps (0 = No, 1 = Yes)",
       color = "Race") +
  theme_minimal()

library(sjPlot)
plot_model(model, type = "int") # This plots all marginal effects for interactions automatically MORE COMPELX?
