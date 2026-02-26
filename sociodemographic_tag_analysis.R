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



df_wide
# Convert to factors
df_wide <- df_wide %>%
  mutate(
    text_id = factor(text_id),
    annotator_id = factor(annotator_id),
    race = factor(race),
    gender = factor(gender),
    age = factor(age)
  )

##### COUNT TO DECIDE FILTERING #####

### Count tags
category_cols <- names(df_wide)[grepl("^is_", names(df_wide))]
# Count how many 1s each tag has
tag_counts <- sapply(df_wide[, category_cols], sum)
# Sort descending (most frequent first)
tag_counts_sorted <- sort(tag_counts, decreasing = TRUE)
# Print
tag_counts_sorted

### Count sociodemographics
# Race, Gender, Age — sorted descending
race_counts  <- sort(table(df_wide$race),  decreasing = TRUE)
gender_counts <- sort(table(df_wide$gender), decreasing = TRUE)
age_counts    <- sort(table(df_wide$age),   decreasing = TRUE)

# Print
race_counts
gender_counts
age_counts

# Remove rare sociodemographics (<3000, since need intersections) and tags (<1000)
df_wide_norare <- df_wide %>%
  mutate(
    race   = remove_rare_levels(factor(race), min_count = 3000),
    gender = remove_rare_levels(factor(gender), min_count = 3000),
    age    = remove_rare_levels(factor(age),  min_count = 3000)
  ) %>%
  filter(!is.na(race), !is.na(gender), !is.na(age))

# Tags thresholds
category_cols <- names(df_wide_norare)[grepl("^is_", names(df_wide_norare))]
tag_counts <- sapply(df_wide_norare[, category_cols], sum)
rare_tags <- names(tag_counts)[tag_counts < 1000]

# Remove rare tags
df_wide_norare <- df_wide_norare %>%
  select(-all_of(rare_tags))

for (var in c("race", "gender", "age")) {
  tbl <- table(df_wide[[var]])
  removed <- names(tbl)[tbl < 3000]
  cat("\nRemoved rare levels for", var, ":", paste(removed, collapse = ", "), "\n")
}

cat("\nRemoved rare tag columns:", paste(rare_tags, collapse = ", "), "\n")

# Assign reference level for each sociodemographic variable: -> man, 25-34, white
df_wide_norare$gender <- relevel(df_wide_norare$gender, ref='man')
df_wide_norare$age <- relevel(df_wide_norare$age, ref='25-34')
df_wide_norare$race <- relevel(df_wide_norare$race, ref='white')

# Print retained levels of sociodemographics
unique(df_wide_norare$gender)
unique(df_wide_norare$age)
unique(df_wide_norare$race)

length(unique(df_wide_norare$annotator_id))
length(unique(df_wide_norare$text_id))

# Clean up memory
rm(df, df_tags_merged, df_wide)

########## Analysis based only on sociodemographic factors ########## 
# Remove rare levels at the sociodemographics
### Null model ###
model0_socdem <- glmmTMB(
  label ~ 1 + 
    (1 | text_id) + 
    (1 | annotator_id),
  data = df_wide_norare,
  family = binomial()
)

### Sociodemographic model ####
model1_socdem <- glmmTMB(
  label ~ race + gender + age + 
    (1 | text_id) + 
    (1 | annotator_id),
  data = df_wide_norare,
  family = binomial()
)

### Sociodemographic intersectional model
model2_socdem <- glmmTMB(
  label ~ race * ( gender + age ) + 
    (1 | text_id) + 
    (1 | annotator_id),
  data = df_wide_norare,
  family = binomial()
)

r2_0_socdem <- r2(model0_socdem)
r2_1_socdem <- r2(model1_socdem)
r2_2_socdem <- r2(model2_socdem)

r2_table_socdem <- data.frame(
  Model = c("Null", "Sociodemographic", "Race-Intersectional"),
  Marginal_R2 = c(r2_0_socdem$R2_marginal, r2_1_socdem$R2_marginal, r2_2_socdem$R2_marginal),
  Conditional_R2 = c(r2_0_socdem$R2_conditional, r2_1_socdem$R2_conditional, r2_2_socdem$R2_conditional)
)

print(r2_table_socdem)
anova(model2_socdem, model1_socdem)
anova(model1_socdem, model0_socdem)

#### Brief comment 
# The sociodemographic variables individually account for 3.4% of variability while they account for 5.3% of variability
#  when considered in intersection, which agrees with previous findings pointing to their limited capacity to explain variability (Hu and Collier)
#  this is considerably lower than the percentage of variability which is explained by the random effects which is 79% in the sociodemographic model 
#  and 77% in the intersectional one.

### Visualize marginal effects of sociodemographics, 
rm(model0_socdem, model1_socdem, model2_socdem)

# Intersectional Marginal Effects of Race and Age on Offensiveness Perception.
# The plot displays the predicted probability of a text being rated as offensive across race and age group for an average text.

#pr_socdem_raceage <- predict_response(model2_socdem, 
#                              terms = c("race", "age"), 
#                              margin = "average") 
# Save plot
# save_path = "figures/models/sociodemographics/raceage.png"
# 
# p <- plot(pr_socdem_raceage)  # the ggeffects plot object
# p <- p + 
#   labs(title = NULL,            # remove title
#        x = "Race",
#        y = "Prob. of Offensive Rating") +
#   theme_minimal(base_size = 12)
# 
# ggsave(filename = save_path,
#        plot = p,
#        width = 6.5,
#        height = 3.5,
#        dpi = 300)
# 
# save_path = "figures/models/sociodemographics/racegender.png"
# pr_socdem_racegender <- predict_response(model2_socdem, 
#                               terms = c("race", "gender"), 
#                               margin = "average") 
# 
# p <- plot(pr_socdem_racegender)  # the ggeffects plot object
# p <- p + 
#   labs(title = NULL,            # remove title
#        x = "Race",
#        y = "Prob. of Offensive Rating") +
#   theme_minimal(base_size = 12)
# 
# ggsave(filename = save_path,
#        plot = p,
#        width = 6.5,
#        height = 3.5,
#        dpi = 300)

# Cleanup memory
# rm(p, pr_socdem_raceage, pr_socdem_racegender, model2_socdem)

### Analysis based on tags
category_cols <- names(df_wide_norare)[grepl("^is_", names(df_wide_norare))]

formula_str <- paste("label ~", paste(category_cols, collapse = " + "), 
                      "+ (1 | annotator_id) + (1 | text_id)")
# 
### Null model ###
model0_tags <- glmmTMB(label ~ 1 + (1|annotator_id) + (1 | text_id),
                   data=df_wide_norare,
                   family=binomial)
 
### Model with individual tags as fixed effects ### 
model1_tags <- glmmTMB(as.formula(formula_str), 
                        data = df_wide_norare, 
                        family = binomial)
 
r2_0_tags <- r2(model0_tags)
r2_1_tags <- r2(model1_tags)
# 
r2_table_tags <- data.frame(
  Model = c("Null", "Tags"),
  Marginal_R2 = c(r2_0_tags$R2_marginal, r2_1_tags$R2_marginal),
  Conditional_R2 = c(r2_0_tags$R2_conditional, r2_1_tags$R2_conditional)
)
# 
r2_table_tags
anova(model1_tags, model0_tags)
# #### Brief comment 
# # The tags variables account for approximately 5.6% of the variability while the random effects still account for majority of the variability.
# # This is larger than what we saw before.
# 
# # Cleanup memory
rm(model0_tags, model1_tags)

# #### Visualize
# # 1. Define the list of category terms from your model
# terms_to_plot <- category_cols
# 
# # 2. Get predicted responses for each term being 'active' (1)
# # margin = "average" calculates the marginal effect over the population
# preds <- lapply(terms_to_plot, function(x) {
#   res <- predict_response(model1_tags, terms = x, margin = "average")
#   # Filter only for the '1' (active) state of the indicator
#   res <- as.data.frame(res) %>% filter(x == 1) 
#   res$term_name <- x
#   return(res)
# })
# 
# plot_data <- bind_rows(preds)
# 
# # 3. Create the plot and save
# save_path = "figures/models/tags/tags.png"
# ggsave(filename = save_path,
#        plot = ggplot(plot_data, aes(x = predicted, y = reorder(term_name, predicted))) +
#          geom_point(size = 3, color = "steelblue") +
#          geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
#          labs(title = "Adjusted Predicted Probabilities by Category",
#               subtitle = "Calculated using the Average Marginal Effect method",
#               y = "Category Indicator",
#               x = "Predicted Probability of Offensive Rating") +
#          theme_minimal(),
#        width = 6.5,
#        height = 3.5,
#        dpi = 300)
# 
# # Cleanup memory
# rm(model1_tags, preds, plot_data)

### Joint analysis of sociodemographic and tags ### 
formula_str_socdemtags <- paste("label ~", paste(category_cols, collapse = " + "), 
                     "+ race * (gender + age) + (1 | annotator_id) + (1 | text_id)")

#formula_str_socdemtags <- paste("label ~", paste(category_cols, collapse = " + "), 
#                                "+ race + gender + age + (1 | annotator_id) + (1 | text_id)")

model0_socdemtags <- glmmTMB(label ~ 1 + (1|annotator_id) + (1 | text_id),
                             data=df_wide_norare,
                             family = binomial)
model1_socdemtags <- glmmTMB(as.formula(formula_str_socdemtags), 
                             data = df_wide_norare, 
                             family = binomial)
# We have too many categories to consider all interactions between categories and sociodemographic traits.
# Choose which ones we want to include
# OPTION 1: Hypothesis / Linguistic driven -> TODO: not enough time.
# OPTION 2: Keep those with highest signal in absolute value.
# is_pr -> 20.951
# is_ps -> 24.251
# is_asf -> 12.196
# is_asm -> 10.219
# is_cds -> 10.365

# formula_str_interax <- paste(
#   "label ~", 
#   paste(category_cols, collapse = " + "), # All tags as controls
#   "+ race * (gender + age)",              # Base intersectionality
#   "+ (is_pr + is_ps + is_asf + is_asm + is_cds) * race", # Sensitivity by Race
#   "+ (is_pr + is_ps + is_asf + is_asm + is_cds) * age",  # Sensitivity by Age
#   "+ (is_pr + is_ps + is_asf + is_asm + is_cds) * gender", # Sensitivity by Gender
#   "+ (1 | annotator_id) + (1 | text_id)"
# )
category_cols <- names(df_wide_norare)[grepl("^is_", names(df_wide_norare))]

# Keep those from exploratory analysis.
formula_str_interax_exploratory <- paste(
  "label ~", 
  paste(category_cols, collapse = " + "), # All tags as controls
  "+ race * (gender + age)",              # Base intersectionality
  "+ (is_ddf + is_pa + is_is) * age", # Sensitivity by Race
  "+ (is_asm + is_asf + is_ddp + is_ps) * race",  # Sensitivity by Age
  "+ (1 | annotator_id) + (1 | text_id)"
)


#model2_socdemtags <- glmmTMB(as.formula(formula_str_interax),
#                             data = df_wide_norare, 
#                             family = binomial)
#

model2_socdemtags <- glmmTMB(as.formula(formula_str_interax_exploratory),
                             data = df_wide_norare, 
                             family = binomial)

# vif_values <- check_collinearity(model2_socdemtags)
# vif_values

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

rm(model1_socdemtags, model0_socdemtags)

rm(r2_0_socdem, r2_0_socdemtags, r2_0_tags, r2_1_tags, r2_1_socdem, r2_1_socdemtags, r2_2_socdem, r2_2_socdemtags)


## Plot sociodemographics: 
### race vs age
save_path = "figures/glmm_results/race_age.pdf"
save_df_path = "figures/glmm_results/race_age.csv"
if (!dir.exists(dirname(save_df_path))) {
  dir.create(dirname(save_path), recursive = TRUE)
}
pr_socdem_raceage <- predict_response(model2_socdemtags,
                             terms = c("race", "age"),
                             margin = "average")
df_clean <- pr_socdem_raceage %>%
  as_tibble() %>%
  select(
    race = x, 
    age = group, 
    predicted, 
    lower = conf.low, 
    upper = conf.high
  )

write.csv(df_clean, save_df_path, row.names = FALSE)

# p <- ggplot(df_clean, aes(x = race, y = predicted, color = age, group = age)) +
#   geom_errorbar(
#     aes(ymin = lower, ymax = upper), 
#     linewidth = 1.5, 
#     width = 0.4, 
#     position = position_dodge(width = 0.7)
#   ) +
#   geom_point(
#     position = position_dodge(width = 0.7), 
#     size = 6
#   ) +
#   theme_minimal() + 
#   theme(
#     legend.position = "top",
#     legend.box = "horizontal",
#     # Axis and Title sizing
#     axis.title = element_text(size = 22, face = "bold"),
#     axis.text = element_text(size = 18, color = "black"),
#     plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
#     legend.text = element_text(size = 16),
#     legend.title = element_text(size = 18, face = "bold")
#   ) +
#   scale_color_brewer(palette = "Dark2") +
#   labs(
#     x = "Race",
#     y = "Predicted Probability",
#     color = "Age"
#   ) +
#   # Force the legend into 2 rows
#   guides(color = guide_legend(nrow = 2, byrow = TRUE))
# 
# ggsave(filename = save_path,
#        plot = p,
#        width = 8,
#        height = 5,
#        dpi = 300)

### race vs gender
rm(pr_socdem_raceage, p)
save_path = "figures/glmm_results/race_gender.pdf"
save_df_path = "figures/glmm_results/race_gender.csv"
if (!dir.exists(dirname(save_df_path))) {
  dir.create(dirname(save_path), recursive = TRUE)
}
pr_socdem_racegender <- predict_response(model2_socdemtags,
                                      terms = c("race", "gender"),
                                      margin = "average")

df_clean <- pr_socdem_racegender %>%
  as_tibble() %>%
  select(
    race = x, 
    gender = group, 
    predicted, 
    lower = conf.low, 
    upper = conf.high
  )

write.csv(df_clean, save_df_path, row.names=FALSE)

# p <- ggplot(df_clean, aes(x = race, y = predicted, color = age, group = age)) +
#   geom_errorbar(
#     aes(ymin = lower, ymax = upper), 
#     linewidth = 1.5, 
#     width = 0.4, 
#     position = position_dodge(width = 0.7)
#   ) +
#   geom_point(
#     position = position_dodge(width = 0.7), 
#     size = 6
#   ) +
#   theme_minimal() + 
#   theme(
#     legend.position = "top",
#     legend.box = "horizontal",
#     # Axis and Title sizing
#     axis.title = element_text(size = 22, face = "bold"),
#     axis.text = element_text(size = 18, color = "black"),
#     plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
#     legend.text = element_text(size = 16),
#     legend.title = element_text(size = 18, face = "bold")
#   ) +
#   scale_color_brewer(palette = "Dark2") +
#   labs(
#     x = "Race",
#     y = "Predicted Probability",
#     color = "Gender"
#   ) +
#   # Force the legend into 2 rows
#   guides(color = guide_legend(nrow = 2, byrow = TRUE))
# 
# ggsave(filename = save_path,
#        plot = p,
#        width = 8,
#        height = 5,
#        dpi = 300)


### Plot tags:
#### Visualize
# 1. Define the list of category terms from your model
terms_to_plot <- category_cols

# 2. Get predicted responses for each term being 'active' (1)
# margin = "average" calculates the marginal effect over the population
preds <- lapply(terms_to_plot, function(x) {
  res <- predict_response(model2_socdemtags, terms = x, margin = "average")
  # Filter only for the '1' (active) state of the indicator
  res <- as.data.frame(res) %>% filter(x == 1) 
  res$term_name <- x
  return(res)
})

plot_data <- bind_rows(preds)

# 3. Create the plot and save
save_path = "figures/models/tags/tags.png"
save_df_path = "figures/glmm_results/tags.csv"
if (!dir.exists(dirname(save_df_path))) {
  dir.create(dirname(save_path), recursive = TRUE)
}

# ggsave(filename = save_path,
#        plot = ggplot(plot_data, aes(x = predicted, y = reorder(term_name, predicted))) +
#          geom_point(size = 3, color = "forestgreen") +
#          geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2) +
#          labs(title = "Adjusted Predicted Probabilities by Category",
#               subtitle = "Calculated using the Average Marginal Effect method",
#               y = "Category Indicator",
#               x = "Predicted Probability of Offensive Rating") +
#          theme_minimal(),
#        width = 8,
#        height = 5,
#        dpi = 300)
write.csv(plot_data, save_df_path, row.names = FALSE)


# Prediction function
predict_socdem_effect <- function(variable_name, group_var, save_df_path){
  plot_data <- predict_response(model2_socdemtags,
                                terms = c(paste0(variable_name, " [0,1]"), group_var), 
                                margin = "average")
                                   
  plot_df <- as.data.frame(plot_data)
  
  plot_df$x <- factor(plot_df$x, levels = c(0, 1), labels = c("Other", variable_name))
  
  write.csv(plot_df, save_df_path, row.names = FALSE)
}

# is_asf, race
variable_name = "is_asf"
group_var = "race"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
predict_socdem_effect(variable_name, group_var, save_df_path)


# is_asm, race
variable_name = "is_asm"
group_var = "race"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
predict_socdem_effect(variable_name, group_var, save_df_path)

# is_ddp, race
variable_name = "is_ddp"
group_var = "race"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
predict_socdem_effect(variable_name, group_var, save_df_path)


# is_ps, race
variable_name = "is_ps"
group_var = "race"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
predict_socdem_effect(variable_name, group_var, save_df_path)

# is_ddf, age
variable_name = "is_ddf"
group_var = "age"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
predict_socdem_effect(variable_name, group_var, save_df_path)

# is_is, age
variable_name = "is_is"
group_var = "age"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
predict_socdem_effect(variable_name, group_var, save_df_path)

# is_pa, age
variable_name = "is_pa"
group_var = "age"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
predict_socdem_effect(variable_name, group_var, save_df_path)



# plot_socdem_effect <- function(variable_name, group_var, color_values, title_text) {
#   
#   # Predict responses
#   plot_data <- predict_response(model2_socdemtags, 
#                                 terms = c(paste0(variable_name, " [0,1]"), group_var), 
#                                 margin = "average")
#   
#   plot_df <- as.data.frame(plot_data)
#   
#   # Format X-axis labels
#   plot_df$x <- factor(plot_df$x, levels = c(0, 1), labels = c("Other", variable_name))
#   
#   # Create Plot
#   p <- ggplot(plot_df, aes(x = x, y = predicted, color = group)) +
#     geom_point(size = 4, position = position_dodge(width = 0.4)) +
#     geom_errorbar(aes(ymin = conf.low, ymax = conf.high), 
#                   width = 0.1, 
#                   linewidth = 1,
#                   position = position_dodge(width = 0.4)) +
#     scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
#     scale_color_manual(values = color_values) +
#     labs(title = title_text,
#          x = paste(variable_name, "Tag"),
#          y = "Prob. of Offensive Rating",
#          color = tools::toTitleCase(group_var)) +
#     theme_minimal() +
#     theme(panel.grid.minor = element_blank(),
#           axis.text = element_text(size = 12),
#           legend.position = "right")
#   
#   # Save
#   save_path <- paste0("figures/glmm_analysis/socdemtags/", variable_name, "_", group_var, ".png")
#   ggsave(filename = save_path, plot = p, width = 6.5, height = 3.5, dpi = 300)
#   
#   return(p)
# }

# # --- 2. Define Color Scales ---
# colors_gender <- c("man" = "#e41a1c", "woman" = "#377eb8")
# colors_age    <- c("18-24" = "#4E79A7", "25-34" = "#F28E2B", "35-44" = "#59A14F", "45-54" = "#E15759", "55-64" = "#B07AA1")
# colors_race   <- c("white" = "#E15759", "asian" = "#4E79A7", "black" = "#59A14F")
# 
# # --- 3. Run the requests ---
# 
# # is_ddp vs race
# plot_socdem_effect("is_ddp", "race", colors_race, "Effect of is_ddp on Offensive Rating by Race")
# # is_ddp vs age
# plot_socdem_effect("is_ddp", "age",  colors_age,  "Effect of is_ddp on Offensive Rating by Age")
# 
# # is_ps vs age
# plot_socdem_effect("is_ps", "age", colors_age, "Effect of is_ps on Offensive Rating by Age")
# 
# # is_is vs race
# plot_socdem_effect("is_is", "race", colors_race, "Effect of is_is on Offensive Rating by Race")
# 
# # is_asm vs age
# plot_socdem_effect("is_asm", "age", colors_age, "Effect of is_asm on Offensive Rating by Age")
# # is_asf vs age
# plot_socdem_effect("is_asf", "age", colors_age, "Effect of is_asf on Offensive Rating by Age")
# 
# # is_pa vs race
# plot_socdem_effect("is_pa", "race", colors_race, "Effect of is_pa on Offensive Rating by Race")
# # is_ddf vs race
# plot_socdem_effect("is_ddf", "race", colors_race, "Effect of is_ddf on Offensive Rating by Race")
# 
# 



