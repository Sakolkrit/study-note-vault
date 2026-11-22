---
tags: [matplotlib, python, programming, data-visualization, plotting]
status: growing
level: 4
tier: 2
---

# Matplotlib

Matplotlib is the standard [[Python]] library for **drawing charts**: lines, dots, bars, histograms, images and more. You give it numbers (usually a [[NumPy]] array or a [[Pandas]] column), and it draws a picture you can look at or save as a file. Almost every training curve and scatter plot in machine learning is made with it, or with a tool built on it.

## 🟢 1. Simple (age 12)
Imagine you have a list of the temperatures for each day of a month. A list of 30 numbers is hard to understand at a glance. If you draw a line going up and down, you can see in one second which days were hot. Matplotlib is the pencil and graph paper of a computer: you hand it numbers and say "draw me a line", and it does it.

**Example:** A runner writes down the time of each lap: 62, 60, 61, 58, 57 seconds. As a list, the pattern is hard to see. Drawn as a line, you can see at once that the laps are getting faster.

## 🔵 2. How it works
1. **Get numbers.** Two lists (or arrays) of the same length: one for the horizontal direction (x) and one for the vertical (y).
2. **Make a figure.** A **figure** is the whole blank page. It has a size, like a sheet of paper.
3. **Add axes.** An **axes** (one plot area, not to be confused with the plural of "axis") is a rectangle on the page with a left edge, a bottom edge and a scale. One figure can hold several, side by side.
4. **Draw on the axes:** a line (`plot`), dots (`scatter`), bars (`bar`), or a count of how many values fall in each range (`hist`).
5. **Label it.** Add a title, names for the two directions (with units), and a legend if there are several lines. A chart nobody can read is not much use.
6. **Show or save it.** Look at it in a window or a notebook, or save it as a picture file (PNG for the web, SVG or PDF if it must stay sharp when enlarged).

![[matplotlib-line-plot.svg]]
*A basic line plot made with Matplotlib: a curve that starts at 1 and falls quickly toward 0 as x grows from 0 to 10. It looks like the curve $e^{-x}$ (the file page does not give the formula). The box is the axes, the numbers along the edges are the ticks, and the blue line is the data. Image: Adrien F. Vincent, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mpl_example_line_plot.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Two ways to use it:**
  - **The quick way** (`plt.plot(...)`): Matplotlib keeps track of a "current" figure for you. Fine for a fast look.
  - **The object way** (`fig, ax = plt.subplots()`, then `ax.plot(...)`): you hold the figure and axes in variables and say exactly which one you draw on. This is clearer and safer when there are several plots. Prefer it for anything you keep.
- **The pieces:** a `Figure` holds one or more `Axes`. An `Axes` holds the lines, dots and bars (called artists), and the two axis scales with their ticks and labels.
- **Common plots:**
  - `ax.plot(x, y)` for a line, and `ax.scatter(x, y)` for dots (one dot per pair, useful when the order does not matter).
  - `ax.bar(names, heights)` for bars, and `ax.hist(values, bins=...)` for the shape of a set of numbers.
  - `ax.imshow(grid)` shows a [[Matrix]] as an image, for example a heat map or a picture.
- **Labels and limits:** `set_xlabel`, `set_ylabel`, `set_title`, `legend`, `set_xlim`, `set_ylim`. By default the limits add a small margin (5% of the data range) on each side, so the axis does not run exactly from the smallest to the largest value (the code shows this).
- **Scales:** `set_yscale('log')` puts equal *ratios* at equal distances. This is useful for loss curves that fall over several orders of magnitude (see [[Loss Function]]).
- **Size and quality:** size is set in inches, and `dpi` is dots per inch. The number of pixels in a saved picture is inches times dpi (equation 1 below). Use a vector format (SVG, PDF) if you want a plot that stays sharp when enlarged.
- **Backends:** a *backend* decides where the picture goes. `Agg` draws into files and needs no window, which is what a script on a server uses. The code below uses it, so it runs anywhere.
- **With pandas:** a pandas Series or DataFrame has `.plot(...)`, which uses Matplotlib underneath. It is a quick way to draw a column (from memory, check the pandas documentation for the options).
- **Other tools:** libraries such as seaborn (statistical charts) and plotly (interactive charts) exist (from memory; not covered here). They are separate libraries, and seaborn builds on Matplotlib.
- **Habits of a good chart:** label axes with units, start a bar chart's axis at zero (bars encode their *length*), do not use too many colours, and prefer a plain, honest picture. A plot is a claim about the data, so check that it does not exaggerate.
- **Versions:** the numbers below were run with Matplotlib 3.7.1, NumPy 1.24.4 and Python 3.9. Defaults (colours, margins, fonts) change between versions, so a plot may look a little different in yours.

![[matplotlib-scatter-plot.svg]]
*A scatter plot: 100 dots, each at its own (x, y) position, here spread over the square from 0 to 1 with no clear pattern. Scatter plots are how you look for a relationship between two columns, for example a model's prediction against the true value. Image: Adrien F. Vincent, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mpl_example_scatter_plot.svg), CC BY-SA 4.0.*

![[matplotlib-bar-chart.svg]]
*A grouped bar chart made with Matplotlib: three dates along the bottom, four bars (A to D) for each date, the numbers written above each bar and a legend naming the colours. Bars are good for comparing a few categories. Image: Mhss, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Matplotlib_bar_sample.svg), CC BY-SA 4.0.*

The code uses no window. It saves figures to a temporary folder and then **asks the objects about themselves**, so each claim can be checked without looking at a picture: (1) the pixel size of saved files, (2) what a figure, its axes and its lines contain, (3) how a data value becomes a pixel, on a linear and a log scale, (4) what a histogram computes, and (5) that the four everyday plots save.

```python
import os, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')                                   # draw into files, no window needed
import matplotlib.pyplot as plt
from PIL import Image
out = tempfile.mkdtemp()
# 1) figure size: inches x dpi = pixels
for dpi in (100, 300):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi); ax.plot([0, 1], [0, 1])
    path = os.path.join(out, f'f{dpi}.png'); fig.savefig(path); plt.close(fig)
    print('figsize 6x4 at dpi', dpi, '-> pixels', Image.open(path).size)
# 2) the object way: a figure holds axes, axes hold lines, and you can ask them questions
x = np.linspace(0, 10, 101)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), dpi=100)
ax1.plot(x, np.exp(-x), label='exp(-x)'); ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.legend()
rng = np.random.default_rng(0); ax2.scatter(rng.random(100), rng.random(100))
print('axes:', len(fig.axes), '| lines in ax1:', len(ax1.lines), '| points in ax1 line:', len(ax1.lines[0].get_xdata()), '| scatter points:', len(ax2.collections[0].get_offsets()))
print('x label:', ax1.get_xlabel(), '| legend text:', [t.get_text() for t in ax1.get_legend().get_texts()])
# 3) data -> pixels is a straight-line map (linear axes)
fig, ax = plt.subplots(figsize=(6, 4), dpi=100); ax.plot([0, 10], [0, 5]); fig.canvas.draw()
pos = ax.get_position(); left, width = pos.x0 * 600, pos.width * 600; x0, x1 = ax.get_xlim()
by_hand = left + (4 - x0) / (x1 - x0) * width
print('x=4 -> pixel', round(float(ax.transData.transform((4, 0))[0]), 3), '| by hand', round(float(by_hand), 3), '| axes left/width in pixels:', round(float(left), 1), round(float(width), 1), '| xlim', (x0, x1))
# log axis: equal ratios give equal pixel steps
ax.set_yscale('log'); ax.set_ylim(1, 1000); fig.canvas.draw()
py = [float(ax.transData.transform((0, v))[1]) for v in (1, 10, 100, 1000)]
print('log axis pixel gaps between y=1,10,100,1000:', [round(py[i + 1] - py[i], 3) for i in range(3)]); plt.close(fig)
# 4) histogram: heights are counts, or density so that the total area is 1
data = np.array([0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 3.5])
counts, edges = np.histogram(data, bins=[0, 1, 2, 3, 4]); dens, _ = np.histogram(data, bins=[0, 1, 2, 3, 4], density=True)
print('counts', counts.tolist(), '| density', dens.round(2).tolist(), '| area', float((dens * np.diff(edges)).sum()))
fig, ax = plt.subplots(); n, b, _ = ax.hist(data, bins=[0, 1, 2, 3, 4]); print('ax.hist returns counts', n.tolist()); plt.close(fig)
# 5) the four everyday plots save without error
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
axs[0, 0].plot(x, np.sin(x)); axs[0, 1].scatter(rng.random(50), rng.random(50)); axs[1, 0].hist(rng.normal(size=1000), bins=30); axs[1, 1].bar(['A', 'B', 'C'], [3, 5, 2])
fig.tight_layout(); fig.savefig(os.path.join(out, 'four.png')); print('four plots saved, size', Image.open(os.path.join(out, 'four.png')).size)
# figsize 6x4 at dpi 100 -> pixels (600, 400)
# figsize 6x4 at dpi 300 -> pixels (1800, 1200)
# axes: 2 | lines in ax1: 1 | points in ax1 line: 101 | scatter points: 100
# x label: x | legend text: ['exp(-x)']
# x=4 -> pixel 265.227 | by hand 265.227 | axes left/width in pixels: 75.0 465.0 | xlim (-0.5, 10.5)
# log axis pixel gaps between y=1,10,100,1000: [102.667, 102.667, 102.667]
# counts [1, 2, 3, 4] | density [0.1, 0.2, 0.3, 0.4] | area 1.0
# ax.hist returns counts [1.0, 2.0, 3.0, 4.0]
# four plots saved, size (800, 600)
```
Reading the output:
- **Size:** a 6 by 4 inch figure saves as 600 by 400 pixels at 100 dpi, and 1800 by 1200 at 300 dpi.
- **Structure:** the figure has 2 axes. The left one holds 1 line with 101 points, and the right one holds 100 scatter points. You can read back labels and legend text from the objects.
- **Margins:** the x-axis of a plot of data from 0 to 10 does *not* run from 0 to 10. It runs from −0.5 to 10.5, because of the default 5% margin. If you want exact limits, set them.
- **Data to pixel:** the value $x=4$ lands at pixel 265.227, which equals the by-hand straight-line formula to three decimals.
- **Log scale:** on the log axis, the pixel gaps between 1, 10, 100 and 1000 are all the same (102.667). Each step multiplies by 10, so equal ratios take equal space.
- **Histogram:** the counts are 1, 2, 3, 4, and the density values 0.1, 0.2, 0.3, 0.4 give a total area of 1.
- **I did not look at the saved pictures** made by this code: the checks are on sizes and on the plot objects, not on how they look. The three images in this note are real Commons files, not made by this code.

## 🔴 4. Deep / formal

**Size on the page.** A picture is a grid of dots (pixels). Its size in pixels is

$$\text{pixels}=\text{inches}\times\text{dpi}\quad(\text{separately for width and height})$$

**Where:**
- $\text{inches}$ is the figure's size in inches (the `figsize`). **Matters:** it sets the physical size if printed, and the proportions of the page.
- $\text{dpi}$ is "dots per inch", how many pixels fit in one inch. **Matters:** the same figure at a higher dpi has more pixels, so it looks sharper, but the file is bigger (and text and lines stay the same size relative to the page).
- $\text{pixels}$ is the width or height of the saved picture in pixels. **Matters:** a plot that looks fine on screen at 100 dpi can look blurry in a printed report, which usually wants 300 dpi or a vector format.

**Example (tiny numbers):** A figure 6 by 4 inches.
1. At 100 dpi: $6\times100=600$ wide and $4\times100=400$ high, so $600\times400$ pixels.
2. At 300 dpi: $6\times300=1800$ and $4\times300=1200$, so $1800\times1200$ pixels.
3. Both match the sizes the code measured on the saved files.
4. Meaning: tripling the dpi makes the picture three times wider and three times taller, so nine times as many pixels. The page size in inches did not change, only the sharpness.

**From a data value to a pixel.** On a linear axis, the position of a value $x$ is a straight-line map from the axis range to the part of the page the axes occupy. On a log axis, the same map is applied to the logarithm:

$$p_x=p_{\text{left}}+\frac{x-x_{\min}}{x_{\max}-x_{\min}}\,W,\qquad p_y=p_{\text{bottom}}+\frac{\log_{10}y-\log_{10}y_{\min}}{\log_{10}y_{\max}-\log_{10}y_{\min}}\,H$$

**Where:**
- $p_{\text{left}}$ and $W$ are where the axes box begins and how wide it is, in pixels. **Matters:** they come from the figure size and the axes position (by default the box starts at 12.5% of the width and spans 77.5% of it).
- $x_{\min}$ and $x_{\max}$ are the two ends of the axis, *not* the smallest and largest data values. **Matters:** the default adds a 5% margin, so they are a little wider than the data. Changing them zooms in or out, without changing the data.
- The fraction $(x-x_{\min})/(x_{\max}-x_{\min})$ is how far along the axis the value is, from 0 to 1. **Matters:** it is the only thing that turns a number into a position.
- $p_{\text{bottom}}$ and $H$ are the same for the vertical direction, and $\log_{10}$ is used when the axis is on a log scale. **Matters:** on a log axis, multiplying a value by 10 always moves it the same number of pixels, so very different sizes fit in one picture. A log axis cannot show 0 or negative values.
- In the picture's coordinates, pixel rows count downward from the top, while the axis counts upward, so a program has to flip one of them. **Matters:** this is a common source of upside-down plots when drawing by hand.

**Example (tiny numbers):** A 6 by 4 inch figure at 100 dpi (600 by 400 pixels), with the default axes box: left edge at $0.125\times600=75$ pixels, width $0.775\times600=465$ pixels. The data runs from 0 to 10, so the default axis limits are $-0.5$ to $10.5$.
1. Value $x=4$: the fraction is $(4-(-0.5))/(10.5-(-0.5))=4.5/11=0.4091$.
2. Pixel: $75+0.4091\times465=75+190.23=265.23$. The code's `transData` gave 265.227.
3. Log axis, vertical: the axes box is $0.77\times400=308$ pixels high, and the range 1 to 1000 spans 3 powers of 10.
4. Each factor of 10 takes $308/3=102.67$ pixels. The code measured gaps of 102.667 for 1 to 10, 10 to 100 and 100 to 1000.
5. Meaning: a plot is just a stretching of the numbers to fit a rectangle. If the range changes (zoom, margins or a log scale), the same value lands at a different pixel, so two plots of the same data can look different. Always read the axis numbers.

**What a histogram draws.** Split the range into bins of width $w$. If $c_i$ values fall in bin $i$ and there are $n$ values in total, the density height of bin $i$ is

$$h_i=\frac{c_i}{n\,w},\qquad \sum_i h_i\,w=1$$

**Where:**
- $c_i$ is the count in bin $i$. **Matters:** a plain histogram draws these counts directly, so heights depend on how many values you have.
- $n$ is the total number of values and $w$ the bin width. **Matters:** dividing by $n\,w$ turns counts into a *density*, so histograms of datasets with different sizes (or different bin widths) can be compared on the same scale.
- $h_i$ is the density height. **Matters:** the area of each bar ($h_iw$) is the *share* of the data in that bin, so the bars' total area is exactly 1. This is the same property a probability density has (see [[Probability]] and [[Normal Distribution]]).
- The bins themselves are a choice. **Matters:** too few bins hide the shape, and too many make it jagged. The same data can look quite different with different bins.

**Example (tiny numbers):** Ten values: one in $[0,1)$, two in $[1,2)$, three in $[2,3)$ and four in $[3,4]$. Width $w=1$, $n=10$.
1. Counts: $c=(1,2,3,4)$.
2. Densities: $h=c/(10\times1)=(0.1,\,0.2,\,0.3,\,0.4)$.
3. Total area: $(0.1+0.2+0.3+0.4)\times1=1.0$. The code gives the same densities and an area of 1.0.
4. Meaning: 40% of the values are in the last bin, which is its bar's area (0.4). With bins of width 0.5 the heights would change (and could double), but the area of each bar would still be its share of the data.

**Other points (from memory, not checked here):**
- Matplotlib was created by John Hunter (first released in the early 2000s), with the plotting commands modelled partly on MATLAB. I did not check either detail for this note.
- The default look changed in version 2.0 (new colours and styles). Both images of lines and dots in this note were made with the older default style, and the bar chart with the newer one, which you can see from the fonts and colours.
- Many plots are easy to make misleading: a cut-off axis, a stretched aspect ratio, or two axes with different scales. Which one is acceptable depends on what you want to say, and I have not covered it in depth here.
- I only tested small plots with the `Agg` backend. Interactive windows, animation and 3D plots behave differently and are not covered.

**What came before, and how this improved on it:**
- **Before:** read tables of numbers, or summarise them with a mean, a variance and a correlation.
- **Its limit:** very different data can give almost the same summary. Anscombe made four small tables in 1973 to show this (from memory; I typed the values from memory and the summaries below match the well-known ones).
- **What changed:** draw the data. John Hunter started Matplotlib in 2003 (from memory, not checked).
- **How much better:** in the code below, the four tables have a mean of y of 7.50, a variance of 4.12 to 4.13, a correlation of 0.816 to 0.817 and a fitted slope of 0.50, but drawn they show a line, a curve, a line with one outlier, and a vertical stack with one far point. The snippet prints the numbers only; it does not draw.

**Check it yourself:**

```python
import numpy as np
x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
ys = [[8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
      [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
      [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
      [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]]
xs = [x, x, x, [8] * 7 + [19] + [8] * 3]
print("four tables of 11 points (Anscombe's quartet, typed in from memory):")
for X, Y in zip(xs, ys):
    X, Y = np.array(X, float), np.array(Y)
    print(f"  mean y {Y.mean():.2f}  variance y {Y.var(ddof=1):.2f}  correlation {np.corrcoef(X, Y)[0, 1]:.3f}  fitted slope {np.polyfit(X, Y, 1)[0]:.2f}")
print("the four summaries agree to within 0.01 (variance) and 0.001 (correlation); a scatter plot of each shows a line, a curve, a line with one outlier, and a vertical stack with one far point")
```

## Related
[[Python]] · [[NumPy]] · [[Pandas]] · [[Matrix]] · [[Probability]] · [[Normal Distribution]] · [[Loss Function]] · [[Machine Learning]] · [[Evaluation Metrics]] · [[Overfitting]] · [[PyTorch]]
- [[Jupyter Notebook]]

## Sources
- Matplotlib documentation: the *Quick start guide* and *Usage guide* (matplotlib.org).
- Hunter, *Matplotlib: A 2D Graphics Environment*, Computing in Science & Engineering, 2007 (from memory; I did not open it for this note).
- Figures: [Mpl_example_line_plot.svg](https://commons.wikimedia.org/wiki/File:Mpl_example_line_plot.svg) and [Mpl_example_scatter_plot.svg](https://commons.wikimedia.org/wiki/File:Mpl_example_scatter_plot.svg) (Adrien F. Vincent, CC BY-SA 4.0), and [Matplotlib_bar_sample.svg](https://commons.wikimedia.org/wiki/File:Matplotlib_bar_sample.svg) (Mhss, CC BY-SA 4.0), all on Wikimedia Commons. I viewed all three before writing the captions.
- The code and every Example number were run for this note (Matplotlib 3.7.1, NumPy 1.24.4, Pillow 10.4.0, Python 3.9).

## Flashcards
#flashcards/programming

#flashcards/programming/reverse #flashcards/topic/data-libs Matplotlib:::The pencil and graph paper of a computer. You hand it numbers and say "draw me a line".

#flashcards/programming/concept #flashcards/topic/data-libs Figure against axes?::A figure is the whole blank page with a size. An axes is one plot area on it (a rectangle with edges and a scale). One figure can hold several.

#flashcards/programming/concept #flashcards/topic/data-libs Which calls draw a line, dots, bars and a histogram?::`plot`, `scatter`, `bar` and `hist`.

#flashcards/programming/concept #flashcards/topic/data-libs What should you add to a chart so it is readable?::A title, names for both directions (with units), and a legend if there are several lines.

#flashcards/programming/concept #flashcards/topic/data-libs Which file format for the web and which when it must stay sharp enlarged?::PNG for the web, SVG or PDF when it must stay sharp.

#flashcards/programming/formula #flashcards/topic/data-libs Write the size of a saved picture in pixels and explain each symbol.
?
$$\text{pixels}=\text{inches}\times\text{dpi}\quad(\text{separately for width and height})$$
- $\text{inches}$: the figure's size in inches (the `figsize`). It sets the physical size if printed and the proportions of the page.
- $\text{dpi}$: dots per inch, how many pixels fit in one inch. The same figure at higher dpi has more pixels and looks sharper, but the file is bigger.
- $\text{pixels}$: the width or height of the saved picture in pixels.

#flashcards/programming/example #flashcards/topic/data-libs A 6 by 4 inch figure at 100 dpi and at 300 dpi. How many pixels?::100 dpi: $600\times400$. 300 dpi: $1800\times1200$. Tripling the dpi makes the picture three times wider and taller, so nine times as many pixels. The page size in inches did not change, only the sharpness.

#flashcards/programming/formula #flashcards/topic/data-libs Write how a data value becomes a pixel position and explain each symbol.
?
$$p_x=p_{\text{left}}+\frac{x-x_{\min}}{x_{\max}-x_{\min}}\,W,\qquad p_y=p_{\text{bottom}}+\frac{\log_{10}y-\log_{10}y_{\min}}{\log_{10}y_{\max}-\log_{10}y_{\min}}\,H$$
- $p_{\text{left}}$, $W$: where the axes box begins and how wide it is, in pixels. By default the box starts at 12.5% of the width and spans 77.5% of it.
- $x_{\min}$, $x_{\max}$: the two ends of the axis, not the smallest and largest data values. The default adds a 5% margin, so they are a little wider than the data.
- The $y$ formula is the same with $\log_{10}$, for a log axis.

#flashcards/programming/example #flashcards/topic/data-libs 6 by 4 inches at 100 dpi, default axes box, data from 0 to 10 (limits $-0.5$ to $10.5$). Where does $x=4$ land horizontally?::Left edge $0.125\times600=75$ pixels, width $0.775\times600=465$ pixels. Fraction $(4+0.5)/11=0.4091$. Pixel $75+0.4091\times465=265.23$. The code's `transData` gave 265.227.

#flashcards/programming/formula #flashcards/topic/data-libs Write the histogram density height and explain each symbol.
?
$$h_i=\frac{c_i}{n\,w},\qquad \sum_i h_i\,w=1$$
- $c_i$: the count in bin $i$. A plain histogram draws these counts, so heights depend on how many values you have.
- $n$: the total number of values. $w$: the bin width. Dividing by $nw$ turns counts into a density, so datasets of different sizes or bin widths compare on one scale.
- $h_i$: the density height. The area of each bar $h_iw$ is the share of the data in that bin, and all areas add to 1.

#flashcards/programming/example #flashcards/topic/data-libs Ten values with counts $(1,2,3,4)$ in four bins of width 1. What are the densities?::$h=c/(10\times1)=(0.1,0.2,0.3,0.4)$. Total area $1.0$. 40% of the values are in the last bin, which is its bar's area. With bins of width 0.5 the heights would change but the area would not.

#flashcards/programming/image #flashcards/topic/data-libs ![[matplotlib-line-plot.svg]]
What does this picture show?
?
A basic line plot made with Matplotlib: a curve that starts at 1 and falls quickly toward 0 as x grows from 0 to 10. It looks like $e^{-x}$ (the file page does not give the formula). The box is the axes, the numbers along the edges are the ticks, and the blue line is the data. (From the note's caption: Adrien F. Vincent, Wikimedia Commons.)

#flashcards/programming/image #flashcards/topic/data-libs ![[matplotlib-scatter-plot.svg]]
What does this picture show?
?
A scatter plot: 100 dots, each at its own (x, y) position, spread over the square from 0 to 1 with no clear pattern. Scatter plots are how you look for a relationship between two columns, for example a model's prediction against the true value. (From the note's caption: Adrien F. Vincent, Wikimedia Commons.)

#flashcards/programming/image #flashcards/topic/data-libs ![[matplotlib-bar-chart.svg]]
What does this picture show?
?
A grouped bar chart: three dates along the bottom, four bars (A to D) for each date, the numbers above each bar and a legend naming the colours. Bars are good for comparing a few categories. (From the note's caption: Mhss, Wikimedia Commons.)
