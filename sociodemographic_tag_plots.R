### Plots for the AME of the interaction effects.
library(dplyr)
library(ggplot2)


##### SOCIODEMOGRAPHIC PLOTS
# Race vs age
df <- read.csv("figures/glmm_results/race_age.csv")
df <- df %>%
  mutate(race = factor(race, levels=c("white", "black", "asian")),
         age = factor(age))

p <- ggplot(df, aes(x = race, y = predicted, color = age, group = age)) +
    geom_errorbar(
      aes(ymin = lower, ymax = upper),
      linewidth = 1.5,
      width = 0.4,
      position = position_dodge(width = 0.7)
    ) +
    geom_point(
      position = position_dodge(width = 0.7),
      size = 6
    ) +
    theme_minimal() +
    theme(
      legend.position = "top",
      legend.box = "horizontal",
      # Axis and Title sizing
      axis.title = element_text(size = 22, face = "bold"),
      axis.text = element_text(size = 18, color = "black"),
      plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
      legend.text = element_text(size = 16),
      legend.title = element_text(size = 18, face = "bold")
    ) +
    ylim(0, 1) + 
    scale_color_brewer(palette = "Dark2") +
    labs(
      x = "Race",
      y = "Predicted Probability",
      color = "Age"
    ) +
    # Force the legend into 2 rows
    guides(color = guide_legend(nrow = 2, byrow = TRUE))

save_path = "figures/glmm_results/race_age.pdf"
ggsave(filename = save_path,
         plot = p,
         width = 8,
         height = 5,
         dpi = 300)


# Race vs gender
df <- read.csv("figures/glmm_results/race_gender.csv")
df <- df %>%
  mutate(race = factor(race, levels=c("white", "black", "asian")),
         gender = factor(gender, levels=c("man", "woman")))

p <- ggplot(df, aes(x = race, y = predicted, color = gender, group = gender)) +
  geom_errorbar(
    aes(ymin = lower, ymax = upper),
    linewidth = 1.5,
    width = 0.4,
    position = position_dodge(width = 0.7)
  ) +
  geom_point(
    position = position_dodge(width = 0.7),
    size = 6
  ) +
  ylim(0, 1) + 
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.box = "horizontal",
    # Axis and Title sizing
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 18, color = "black"),
    plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 18, face = "bold")
  ) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = "Race",
    y = "Predicted Probability",
    color = "Gender"
  ) +
  # Force the legend into 2 rows
  guides(color = guide_legend(nrow = 2, byrow = TRUE))
p
save_path = "figures/glmm_results/race_gender.pdf"
ggsave(filename = save_path,
       plot = p,
       width = 8,
       height = 5,
       dpi = 300)


##### TAGS PLOTS
df <- read.csv("figures/glmm_results/tags.csv")

color_chosen = "#009999"
p <- ggplot(df, aes(x = predicted, y = reorder(term_name, predicted))) +
           geom_point(size = 3, color = color_chosen) +
           geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, color=color_chosen) +
           labs(y = "Category Indicator",
                x = "Predicted Probability") +
           theme_minimal() +
            theme(
              # Axis and Title sizing
              axis.title = element_text(size = 22, face = "bold"),
              axis.text = element_text(size = 18, color = "black"),
              plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
              legend.text = element_text(size = 16),
              legend.title = element_text(size = 18, face = "bold")
            ) 
save_path = "figures/glmm_results/tags.pdf"
ggsave(filename = save_path,
       plot = p,
       width = 8,
       height = 5,
       dpi = 300)


##### SOCIODEMOGRAPHIC-TAGS PLOTS
# is_asf, race
variable_name = "is_asf"
group_var = "race"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
save_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".pdf")
df <- read.csv(save_df_path)

df <- df %>%
  mutate(x = as.factor(x),
         x = relevel(x, ref='Other'))

p <- ggplot(df, aes(x = x, y = predicted, color = group, group = group)) +
  geom_errorbar(
    aes(ymin = conf.low, ymax = conf.high),
    linewidth = 1.5,
    width = 0.4,
    position = position_dodge(width = 0.7)
  ) +
  geom_point(
    position = position_dodge(width = 0.7),
    size = 6
  ) +
  ylim(0, 1) + 
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.box = "horizontal",
    # Axis and Title sizing
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 18, color = "black"),
    plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 18, face = "bold")
  ) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = NULL,
    y = "Predicted Probability",
    color = "group"
  ) # + guides(color = guide_legend(nrow = 2, byrow = TRUE))

ggsave(filename = save_path,
       plot = p,
       width = 8,
       height = 5,
       dpi = 300)

# is_asm, race
variable_name = "is_asm"
group_var = "race"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
save_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".pdf")
df <- read.csv(save_df_path)

df <- df %>%
  mutate(x = as.factor(x),
         x = relevel(x, ref='Other'))
  
p <- ggplot(df, aes(x = x, y = predicted, color = group, group = group)) +
  geom_errorbar(
    aes(ymin = conf.low, ymax = conf.high),
    linewidth = 1.5,
    width = 0.4,
    position = position_dodge(width = 0.7)
  ) +
  geom_point(
    position = position_dodge(width = 0.7),
    size = 6
  ) +
  ylim(0, 1) + 
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.box = "horizontal",
    # Axis and Title sizing
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 18, color = "black"),
    plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 18, face = "bold")
  ) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = NULL,
    y = "Predicted Probability",
    color = "group"
  ) # + guides(color = guide_legend(nrow = 2, byrow = TRUE))

p
ggsave(filename = save_path,
       plot = p,
       width = 8,
       height = 5,
       dpi = 300)


# is_ddp, race
variable_name = "is_ddp"
group_var = "race"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
save_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".pdf")
df <- read.csv(save_df_path)

df <- df %>%
  mutate(x = as.factor(x),
         x = relevel(x, ref='Other'))

p <- ggplot(df, aes(x = x, y = predicted, color = group, group = group)) +
  geom_errorbar(
    aes(ymin = conf.low, ymax = conf.high),
    linewidth = 1.5,
    width = 0.4,
    position = position_dodge(width = 0.7)
  ) +
  geom_point(
    position = position_dodge(width = 0.7),
    size = 6
  ) +
  ylim(0, 1) + 
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.box = "horizontal",
    # Axis and Title sizing
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 18, color = "black"),
    plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 18, face = "bold")
  ) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = NULL,
    y = "Predicted Probability",
    color = "group"
  ) # + guides(color = guide_legend(nrow = 2, byrow = TRUE))

p
ggsave(filename = save_path,
       plot = p,
       width = 8,
       height = 5,
       dpi = 300)


# is_ps, race
variable_name = "is_ps"
group_var = "race"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
save_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".pdf")
df <- read.csv(save_df_path)

df <- df %>%
  mutate(x = as.factor(x),
         x = relevel(x, ref='Other'))

p <- ggplot(df, aes(x = x, y = predicted, color = group, group = group)) +
  geom_errorbar(
    aes(ymin = conf.low, ymax = conf.high),
    linewidth = 1.5,
    width = 0.4,
    position = position_dodge(width = 0.7)
  ) +
  geom_point(
    position = position_dodge(width = 0.7),
    size = 6
  ) +
  ylim(0, 1) + 
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.box = "horizontal",
    # Axis and Title sizing
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 18, color = "black"),
    plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 18, face = "bold")
  ) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = NULL,
    y = "Predicted Probability",
    color = "group"
  ) # + guides(color = guide_legend(nrow = 2, byrow = TRUE))

p
ggsave(filename = save_path,
       plot = p,
       width = 8,
       height = 5,
       dpi = 300)

# is_ddf, age
variable_name = "is_ddf"
group_var = "age"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
save_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".pdf")
df <- read.csv(save_df_path)

df <- df %>%
  mutate(x = as.factor(x),
         x = relevel(x, ref='Other'))

p <- ggplot(df, aes(x = x, y = predicted, color = group, group = group)) +
  geom_errorbar(
    aes(ymin = conf.low, ymax = conf.high),
    linewidth = 1.5,
    width = 0.4,
    position = position_dodge(width = 0.7)
  ) +
  geom_point(
    position = position_dodge(width = 0.7),
    size = 6
  ) +
  ylim(0, 1) + 
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.box = "horizontal",
    # Axis and Title sizing
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 18, color = "black"),
    plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 18, face = "bold")
  ) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = NULL,
    y = "Predicted Probability",
    color = "group"
  ) + guides(color = guide_legend(nrow = 2, byrow = TRUE))

p
ggsave(filename = save_path,
       plot = p,
       width = 8,
       height = 5,
       dpi = 300)

# is_is, age
variable_name = "is_is"
group_var = "age"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
save_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".pdf")
df <- read.csv(save_df_path)

df <- df %>%
  mutate(x = as.factor(x),
         x = relevel(x, ref='Other'))

p <- ggplot(df, aes(x = x, y = predicted, color = group, group = group)) +
  geom_errorbar(
    aes(ymin = conf.low, ymax = conf.high),
    linewidth = 1.5,
    width = 0.4,
    position = position_dodge(width = 0.7)
  ) +
  geom_point(
    position = position_dodge(width = 0.7),
    size = 6
  ) +
  ylim(0, 1) + 
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.box = "horizontal",
    # Axis and Title sizing
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 18, color = "black"),
    plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 18, face = "bold")
  ) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = NULL,
    y = "Predicted Probability",
    color = "group"
  ) + guides(color = guide_legend(nrow = 2, byrow = TRUE))

p
ggsave(filename = save_path,
       plot = p,
       width = 8,
       height = 5,
       dpi = 300)

# is_pa, age
variable_name = "is_pa"
group_var = "age"
save_df_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".csv")
save_path <- paste0("figures/glmm_results/", variable_name, "_", group_var, ".pdf")
df <- read.csv(save_df_path)

df <- df %>%
  mutate(x = as.factor(x),
         x = relevel(x, ref='Other'))

p <- ggplot(df, aes(x = x, y = predicted, color = group, group = group)) +
  geom_errorbar(
    aes(ymin = conf.low, ymax = conf.high),
    linewidth = 1.5,
    width = 0.4,
    position = position_dodge(width = 0.7)
  ) +
  geom_point(
    position = position_dodge(width = 0.7),
    size = 6
  ) +
  ylim(0, 1) + 
  theme_minimal() +
  theme(
    legend.position = "top",
    legend.box = "horizontal",
    # Axis and Title sizing
    axis.title = element_text(size = 22, face = "bold"),
    axis.text = element_text(size = 18, color = "black"),
    plot.title = element_text(size = 26, face = "bold", hjust = 0.5),
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 18, face = "bold")
  ) +
  scale_color_brewer(palette = "Dark2") +
  labs(
    x = NULL,
    y = "Predicted Probability",
    color = "group"
  ) + guides(color = guide_legend(nrow = 2, byrow = TRUE))

p
ggsave(filename = save_path,
       plot = p,
       width = 8,
       height = 5,
       dpi = 300)

