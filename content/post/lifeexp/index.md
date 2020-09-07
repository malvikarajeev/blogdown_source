---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Visualizing Inequalities in Life Expectancy"
subtitle: "Tidyverse <3 "
summary: ""
authors: []
tags: []
categories: []
date: 2020-06-12T19:07:34-07:00
lastmod: 2020-06-12T19:07:34-07:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
## United Nations life expectancy data
<p>Life expectancy at birth is a measure of the average a living being is expected to live. It takes into account several demographic factors like gender, country, or year of birth.</p>
<p>Life expectancy at birth can vary along time or between countries because of many causes: the evolution of medicine, the degree of development of countries, or the effect of armed conflicts. Life expectancy varies between gender, as well. The data shows that women live longer that men. Why? Several potential factors, including biological reasons and the theory that women tend to be more health conscious.</p>
<p>Let's create some plots to explore the inequalities about life expectancy at birth around the world. We will use a dataset from the United Nations Statistics Division, which is available <a href="http://data.un.org/Data.aspx?d=GenderStat&f=inID:37&c=1,2,3,4,5,6&s=crEngName:asc,sgvEngName:asc,timeEngName:desc&v=1">here</a>.</p>


```R
options(repr.plot.width = 6, repr.plot.height = 6)

# Loading packages
library(dplyr)
library(tidyr)
library(ggplot2)

# Loading data
life_expectancy <- read.csv("datasets/UNdata.csv")

# Taking a look at the first few rows
head(life_expectancy)
```


<table>
<caption>A data.frame: 6 x 7</caption>
<thead>
	<tr><th scope=col>Country.or.Area</th><th scope=col>Subgroup</th><th scope=col>Year</th><th scope=col>Source</th><th scope=col>Unit</th><th scope=col>Value</th><th scope=col>Value.Footnotes</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Afghanistan</td><td>Female</td><td>2000-2005</td><td>UNPD_World Population Prospects_2006 (International estimate)</td><td>Years</td><td>42</td><td>NA</td></tr>
	<tr><td>Afghanistan</td><td>Female</td><td>1995-2000</td><td>UNPD_World Population Prospects_2006 (International estimate)</td><td>Years</td><td>42</td><td>NA</td></tr>
	<tr><td>Afghanistan</td><td>Female</td><td>1990-1995</td><td>UNPD_World Population Prospects_2006 (International estimate)</td><td>Years</td><td>42</td><td>NA</td></tr>
	<tr><td>Afghanistan</td><td>Female</td><td>1985-1990</td><td>UNPD_World Population Prospects_2006 (International estimate)</td><td>Years</td><td>41</td><td>NA</td></tr>
	<tr><td>Afghanistan</td><td>Male  </td><td>2000-2005</td><td>UNPD_World Population Prospects_2006 (International estimate)</td><td>Years</td><td>42</td><td>NA</td></tr>
	<tr><td>Afghanistan</td><td>Male  </td><td>1995-2000</td><td>UNPD_World Population Prospects_2006 (International estimate)</td><td>Years</td><td>42</td><td>NA</td></tr>
</tbody>
</table>



## 2. Life expectancy of men vs. women by country
<p>Let's manipulate the data to make our exploration easier. We will build the dataset for our first plot in which we will represent the average life expectancy of men and women across countries for the last period recorded in our data (2000-2005).</p>


```R
# Subsetting and reshaping the life expectancy data
subdata <- life_expectancy  %>% 
                  filter(Year == '2000-2005') %>%
                    select(Country.or.Area, Subgroup, Value) %>%
                spread(Subgroup, Value)


```

## 3. Visualize I
<p>A scatter plot is a useful way to visualize the relationship between two variables. It is a simple plot in which points are arranged on two axes, each of which represents one of those variables. </p>
<p>Let's create a scatter plot using <code>ggplot2</code> to represent life expectancy of males (on the x-axis) against females (on the y-axis). We will create a straightforward plot in this task, without many details. We will take care of these kinds of things shortly.</p>


```R
# Plotting male and female life expectancy

ggplot(subdata) + geom_point(aes(x = Male, y = Female)) + theme_minimal()
```


![png](./notebook_5_0.png)


## 4. Reference lines I
<p>A good plot must be easy to understand. There are many tools in <code>ggplot2</code> to achieve this goal and we will explore some of them now. Starting from the previous plot, let's set the same limits for both axes as well as place a diagonal line for reference. After doing this, the difference between men and women across countries will be easier to interpret.</p>
<p>After completing this task, we will see how most of the points are arranged above the diagonal and how there is a significant dispersion among them. What does this all mean?</p>


```R
# Adding an abline and changing the scale of axes of the previous plots
ggplot(subdata) + geom_point(aes(x = Male, y = Female)) +
                 geom_abline(slope = 1, intercept = 0, col = 'blue') +
                coord_cartesian(xlim =c(35, 85), ylim = c(35, 85)) +
                theme_minimal()
```


![png](./notebook_7_0.png)


## 5. Plot titles and axis labels
<p>A key point to make a plot understandable is placing clear labels on it. Let's add titles, axis labels, and a caption to refer to the source of data. Let's also change the appearance to make it clearer.</p>


```R
# Adding labels to previous plot
ggplot(subdata, aes(x=Male, y=Female))+
  geom_point(colour="white", fill="chartreuse3", shape=21, alpha=.55, size=5)+
  geom_abline(intercept = 0, slope = 1, linetype=2)+
  scale_x_continuous(limits=c(35,85))+
  scale_y_continuous(limits=c(35,85))+
  labs(title="Life Expectancy at Birth by Country",
       subtitle="Years. Period: 2000-2005. Average.",
       caption="Source: United Nations Statistics Division",
       x='Males',
       y='Females') + 
theme_minimal()
```


![png](./notebook_9_0.png)


## 6. Highlighting remarkable countries I
<p>Now, we will label some points of our plot with the name of its corresponding country. We want to draw attention to some special countries where the gap in life expectancy between men and women is significantly high. These will be the final touches on this first plot.</p>


```R
# Subseting data to obtain countries of interest
top_male <- subdata %>% arrange(Male-Female) %>% head(3)
top_female <- subdata %>% arrange(Female-Male) %>% head(3)

```


```R
# Adding text to the previous plot to label countries of interest
ggplot(subdata, aes(x=Male, y=Female, label = Country.or.Area))+
  geom_point(colour="white", fill="chartreuse3", shape=21, alpha=.55, size=5)+
  geom_abline(intercept = 0, slope = 1, linetype=2)+
  scale_x_continuous(limits=c(35,85))+
  scale_y_continuous(limits=c(35,85))+
  labs(title="Life Expectancy at Birth by Country",
       subtitle="Years. Period: 2000-2005. Average.",
       caption="Source: United Nations Statistics Division",
       x="Males",
       y="Females")+
        geom_text(data=top_male, size=3) +
        geom_text(data=top_female, size=3) + 
            theme_bw()
```


![png](./notebook_12_0.png)


## 7. How has life expectancy by gender evolved?
<p>Since our data contains historical information, let's see now how life expectancy has evolved in recent years. Our second plot will represent the difference between men and women across countries between two periods: 2000-2005 and 1985-1990.</p>
<p>Let's start building a dataset called <code>subdata2</code> for our second plot. </p>


```R
# Subsetting, mutating and reshaping the life expectancy data
subdata2 <- life_expectancy %>% 
  filter(Year %in% c("1985-1990", "2000-2005")) %>% 
  mutate(Sub_Year=paste(Subgroup, Year, sep="_")) %>% 
  mutate(Sub_Year=gsub("-", "_", Sub_Year)) %>% 
  select(-Subgroup, -Year) %>% 
    spread(Sub_Year, Value) %>%
        mutate(diff_Female = Female_2000_2005 - Female_1985_1990,
              diff_Male = Male_2000_2005 - Male_1985_1990)
```

## 8. Visualize II
<p>Now let's create our second plot in which we will represent average life expectancy differences between "1985-1990" and "2000-2005" for men and women.</p>


```R
# Doing a nice first version of the plot with abline, scaling axis and adding labels
ggplot(subdata2, aes(x=diff_Male, y=diff_Female, label=Country.or.Area))+
  geom_point(colour="white", fill="chartreuse3", shape=21, alpha=.55, size=5)+
  geom_abline(intercept = 0, slope = 1, linetype=2)+
    scale_x_continuous(limit = c(-25, 25)) + 
    scale_y_continuous(limit = c(-25, 25)) +
  labs(title="Life Expectancy at Birth by Country in Years",
       subtitle="Difference between 1985-1990 and 2000-2005. Average.",
       caption="Source: United Nations Statistics Division",
       x="Males",
       y="Females")+
theme_bw()
```


![png](./notebook_16_0.png)


## 9. Reference lines II
<p>Adding reference lines can make plots easier to understand. We already added a diagonal line to visualize differences between men and women more clearly. Now we will add two more lines to help to identify in which countries people increased or decreased their life expectancy in the period analyzed.</p>


```R
# Adding an hline and vline to previous plots
ggplot(subdata2, aes(x=diff_Male, y=diff_Female, label=Country.or.Area))+
  geom_point(colour="white", fill="chartreuse3", shape=21, alpha=.55, size=5)+
  geom_abline(intercept = 0, slope = 1, linetype=2)+
  scale_x_continuous(limits=c(-25,25))+
  scale_y_continuous(limits=c(-25,25))+
   geom_hline(yintercept = 0, linetype = 2) +
    geom_vline(xintercept = 0, linetype = 2)+
  labs(title="Life Expectancy at Birth by Country",
       subtitle="Years. Difference between 1985-1990 and 2000-2005. Average.",
       caption="Source: United Nations Statistics Division",
       x="Males",
       y="Females")+
theme_bw()
```


![png](./notebook_18_0.png)


## 10. Highlighting remarkable countries II
<p>As we did in the first plot, let's label some points. Concretely, we will point those three where the aggregated average life expectancy for men and women increased most and those three where decreased most in the period.</p>


```R
# Subseting data to obtain countries of interest
top <- subdata2 %>% arrange(diff_Male+diff_Female) %>% head(3)
bottom <- subdata2 %>% arrange(diff_Male+diff_Female) %>% tail(3)

# Adding text to the previous plot to label countries of interest
ggplot(subdata2, aes(x=diff_Male, y=diff_Female, label=Country.or.Area), guide=FALSE)+
  geom_point(colour="white", fill="chartreuse3", shape=21, alpha=.55, size=5)+
  geom_abline(intercept = 0, slope = 1, linetype=2)+
  scale_x_continuous(limits=c(-25,25))+
  scale_y_continuous(limits=c(-25,25))+
  geom_hline(yintercept=0, linetype=2)+
  geom_vline(xintercept=0, linetype=2)+
  labs(title="Life Expectancy at Birth by Country",
       subtitle="Years. Difference between 1985-1990 and 2000-2005. Average.",
       caption="Source: United Nations Statistics Division",
       x="Males",
       y="Females")+
    geom_text(data = top, size = 3) +
    geom_text(data = bottom, size = 3)+
  theme_bw()
```


![png](./notebook_20_0.png)

