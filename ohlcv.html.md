# ohlcv


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

## Plotting OHLC data

Functions to plot times series in OHLC format (Open, High, Low, Close)
and OHLCV format (same + volume).

------------------------------------------------------------------------

<a
href="https://github.com/vtecftwy/myquantlab/blob/main/myquantlab/ohlc.py#L17"
target="_blank" style="float:right; font-size:smaller">source</a>

### candlestick_plot

>  candlestick_plot (df:pandas.core.frame.DataFrame, width:int=950,
>                        height:int=600, chart_title:str='',
>                        fig:bokeh.plotting._figure.figure|None=None)

*Create a candlestick chart (Bokeh) using a dataframe with ‘Open’,
‘High’, ‘Low’, ‘Close’, ‘Volume’.*

<table>
<colgroup>
<col style="width: 6%" />
<col style="width: 25%" />
<col style="width: 34%" />
<col style="width: 34%" />
</colgroup>
<thead>
<tr>
<th></th>
<th><strong>Type</strong></th>
<th><strong>Default</strong></th>
<th><strong>Details</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>df</td>
<td>DataFrame</td>
<td></td>
<td>df with datetime index, and at least following columns ‘Open’,
‘High’, ‘Low’, ‘Close’, ‘Volume’</td>
</tr>
<tr>
<td>width</td>
<td>int</td>
<td>950</td>
<td>height of the plot figure</td>
</tr>
<tr>
<td>height</td>
<td>int</td>
<td>600</td>
<td>height of the plot figure</td>
</tr>
<tr>
<td>chart_title</td>
<td>str</td>
<td></td>
<td>title of the chart</td>
</tr>
<tr>
<td>fig</td>
<td>bokeh.plotting._figure.figure | None</td>
<td>None</td>
<td>figure to allow superposition of other lines on candlestick
plot</td>
</tr>
<tr>
<td><strong>Returns</strong></td>
<td><strong>None</strong></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

Before using the function in a notebook, you must load BokehJS, with:

``` python
output_notebook()
```

    <style>
        .bk-notebook-logo {
            display: block;
            width: 20px;
            height: 20px;
            background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAYAAACNiR0NAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAABx0RVh0U29mdHdhcmUAQWRvYmUgRmlyZXdvcmtzIENTNui8sowAAAOkSURBVDiNjZRtaJVlGMd/1/08zzln5zjP1LWcU9N0NkN8m2CYjpgQYQXqSs0I84OLIC0hkEKoPtiH3gmKoiJDU7QpLgoLjLIQCpEsNJ1vqUOdO7ppbuec5+V+rj4ctwzd8IIbbi6u+8f1539dt3A78eXC7QizUF7gyV1fD1Yqg4JWz84yffhm0qkFqBogB9rM8tZdtwVsPUhWhGcFJngGeWrPzHm5oaMmkfEg1usvLFyc8jLRqDOMru7AyC8saQr7GG7f5fvDeH7Ej8CM66nIF+8yngt6HWaKh7k49Soy9nXurCi1o3qUbS3zWfrYeQDTB/Qj6kX6Ybhw4B+bOYoLKCC9H3Nu/leUTZ1JdRWkkn2ldcCamzrcf47KKXdAJllSlxAOkRgyHsGC/zRday5Qld9DyoM4/q/rUoy/CXh3jzOu3bHUVZeU+DEn8FInkPBFlu3+nW3Nw0mk6vCDiWg8CeJaxEwuHS3+z5RgY+YBR6V1Z1nxSOfoaPa4LASWxxdNp+VWTk7+4vzaou8v8PN+xo+KY2xsw6une2frhw05CTYOmQvsEhjhWjn0bmXPjpE1+kplmmkP3suftwTubK9Vq22qKmrBhpY4jvd5afdRA3wGjFAgcnTK2s4hY0/GPNIb0nErGMCRxWOOX64Z8RAC4oCXdklmEvcL8o0BfkNK4lUg9HTl+oPlQxdNo3Mg4Nv175e/1LDGzZen30MEjRUtmXSfiTVu1kK8W4txyV6BMKlbgk3lMwYCiusNy9fVfvvwMxv8Ynl6vxoByANLTWplvuj/nF9m2+PDtt1eiHPBr1oIfhCChQMBw6Aw0UulqTKZdfVvfG7VcfIqLG9bcldL/+pdWTLxLUy8Qq38heUIjh4XlzZxzQm19lLFlr8vdQ97rjZVOLf8nclzckbcD4wxXMidpX30sFd37Fv/GtwwhzhxGVAprjbg0gCAEeIgwCZyTV2Z1REEW8O4py0wsjeloKoMr6iCY6dP92H6Vw/oTyICIthibxjm/DfN9lVz8IqtqKYLUXfoKVMVQVVJOElGjrnnUt9T9wbgp8AyYKaGlqingHZU/uG2NTZSVqwHQTWkx9hxjkpWDaCg6Ckj5qebgBVbT3V3NNXMSiWSDdGV3hrtzla7J+duwPOToIg42ChPQOQjspnSlp1V+Gjdged7+8UN5CRAV7a5EdFNwCjEaBR27b3W890TE7g24NAP/mMDXRWrGoFPQI9ls/MWO2dWFAar/xcOIImbbpA3zgAAAABJRU5ErkJggg==);
        }
    </style>
    <div>
        <a href="https://bokeh.org" target="_blank" class="bk-notebook-logo"></a>
        <span id="a8b56d63-3a48-4d76-8722-39f70d2b98b4">Loading BokehJS ...</span>
    </div>

    Unable to display output for mime type(s): application/javascript, application/vnd.bokehjs_load.v0+json

Let’s load a test DataFrame and plot it.

``` python
df = load_test_df()
display(df.head(10))
candlestick_plot(df.head(10), width=800, height=400, chart_title='Candlestick plot')
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Open</th>
<th data-quarto-table-cell-role="th">High</th>
<th data-quarto-table-cell-role="th">Low</th>
<th data-quarto-table-cell-role="th">Close</th>
<th data-quarto-table-cell-role="th">Volume</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">dt</th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">2018-10-22</td>
<td>2759.02</td>
<td>2779.27</td>
<td>2747.27</td>
<td>2754.48</td>
<td>26562</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-10-23</td>
<td>2753.11</td>
<td>2755.36</td>
<td>2690.69</td>
<td>2743.45</td>
<td>38777</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-10-24</td>
<td>2744.83</td>
<td>2748.58</td>
<td>2651.23</td>
<td>2672.80</td>
<td>41777</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-10-25</td>
<td>2670.80</td>
<td>2722.90</td>
<td>2657.93</td>
<td>2680.71</td>
<td>39034</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-10-26</td>
<td>2675.59</td>
<td>2692.34</td>
<td>2627.59</td>
<td>2663.57</td>
<td>61436</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-10-29</td>
<td>2667.70</td>
<td>2707.00</td>
<td>2603.33</td>
<td>2639.17</td>
<td>44960</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-10-30</td>
<td>2639.55</td>
<td>2689.50</td>
<td>2633.05</td>
<td>2688.50</td>
<td>52786</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-10-31</td>
<td>2688.88</td>
<td>2736.76</td>
<td>2681.25</td>
<td>2704.75</td>
<td>32374</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-11-01</td>
<td>2707.13</td>
<td>2741.58</td>
<td>2706.88</td>
<td>2731.90</td>
<td>29565</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-11-02</td>
<td>2725.28</td>
<td>2766.28</td>
<td>2699.96</td>
<td>2723.76</td>
<td>41892</td>
</tr>
</tbody>
</table>

</div>

  <div id="c35f0de7-36ef-4ce2-bafa-fbd8c001f81e" data-root-id="p1245" style="display: contents;"></div>

    Unable to display output for mime type(s): application/javascript, application/vnd.bokehjs_exec.v0+json

Note that the plot uses a full `DateTimeIndex` x axis, and the missing
bars will just be empty. This allows to compare plots for time series
with different missing bars

## Handling OLHCV data

Function performing transformation and analysis on OLHCV data.

------------------------------------------------------------------------

<a
href="https://github.com/vtecftwy/myquantlab/blob/main/myquantlab/ohlc.py#L73"
target="_blank" style="float:right; font-size:smaller">source</a>

### resample_ohlcv

>  resample_ohlcv (df:pandas.core.frame.DataFrame, rule_str='W-FRI')

\*Resample a `DataFrame` with OHLCV format according to given rule
string.

The re-sampling is applied to each of the OHLC and optional V column.
Re-sampling aggregate applies `first()`, `max()`, `min()`, `last()` and
`sum()` to OHLCV respectively.\*

<table>
<colgroup>
<col style="width: 6%" />
<col style="width: 25%" />
<col style="width: 34%" />
<col style="width: 34%" />
</colgroup>
<thead>
<tr>
<th></th>
<th><strong>Type</strong></th>
<th><strong>Default</strong></th>
<th><strong>Details</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>df</td>
<td>DataFrame</td>
<td></td>
<td><code>df</code> with datetime index, and at least following columns
‘Open’, ‘High’, ‘Low’, ‘Close’. Optional ‘Volume’</td>
</tr>
<tr>
<td>rule_str</td>
<td>str</td>
<td>W-FRI</td>
<td><code>DateOffset</code> alias for resampling. Default: ‘W-FRI’.
Other commons: ‘D’, ‘B’, ‘W’, ‘M’</td>
</tr>
<tr>
<td><strong>Returns</strong></td>
<td><strong>DataFrame</strong></td>
<td></td>
<td><strong>resampled <code>df</code> with columns ‘Open’, ‘High’,
‘Low’, ‘Close’. Optional ‘Volume’</strong></td>
</tr>
</tbody>
</table>

The list of all `DateOffset` string reference can be found in Pandas’
documentation
[here](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects).

The test `df` has one bar (row) for each day. We can resample to
aggregate data per week, where one week ends on Friday.

``` python
df_wk = resample_ohlcv(df, rule_str='W-FRI')
df_wk.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

<table class="dataframe" data-quarto-postprocess="true" data-border="1">
<thead>
<tr style="text-align: right;">
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th">Open</th>
<th data-quarto-table-cell-role="th">High</th>
<th data-quarto-table-cell-role="th">Low</th>
<th data-quarto-table-cell-role="th">Close</th>
<th data-quarto-table-cell-role="th">Volume</th>
</tr>
<tr>
<th data-quarto-table-cell-role="th">W-FRI</th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
<th data-quarto-table-cell-role="th"></th>
</tr>
</thead>
<tbody>
<tr>
<td data-quarto-table-cell-role="th">2018-10-26</td>
<td>2759.02</td>
<td>2779.27</td>
<td>2627.59</td>
<td>2663.57</td>
<td>207586</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-11-02</td>
<td>2667.70</td>
<td>2766.28</td>
<td>2603.33</td>
<td>2723.76</td>
<td>201577</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-11-09</td>
<td>2721.51</td>
<td>2817.01</td>
<td>2713.14</td>
<td>2778.60</td>
<td>118857</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-11-16</td>
<td>2777.10</td>
<td>2794.23</td>
<td>2669.14</td>
<td>2740.15</td>
<td>170290</td>
</tr>
<tr>
<td data-quarto-table-cell-role="th">2018-11-23</td>
<td>2732.15</td>
<td>2746.53</td>
<td>2625.66</td>
<td>2630.36</td>
<td>132017</td>
</tr>
</tbody>
</table>

</div>

``` python
print('Days of the week for initial df:',list(df.index.day_of_week[:5]))
print('Days of the week for sampled df:',list(df_wk.index.day_of_week[:5]))
```

    Days of the week for initial df: [0, 1, 2, 3, 4]
    Days of the week for sampled df: [4, 4, 4, 4, 4]

------------------------------------------------------------------------

<a
href="https://github.com/vtecftwy/myquantlab/blob/main/myquantlab/ohlc.py#L104"
target="_blank" style="float:right; font-size:smaller">source</a>

### autocorrelation_ohlcv

>  autocorrelation_ohlcv (df:pandas.core.frame.DataFrame, max_lag:int=10,
>                             ohlc_col:str='Close')

*Return autocorrelation for a range of lags and for the selected
ohlc_col_col defined.*

<table>
<colgroup>
<col style="width: 6%" />
<col style="width: 25%" />
<col style="width: 34%" />
<col style="width: 34%" />
</colgroup>
<thead>
<tr>
<th></th>
<th><strong>Type</strong></th>
<th><strong>Default</strong></th>
<th><strong>Details</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>df</td>
<td>DataFrame</td>
<td></td>
<td><code>df</code> with <code>DateTimeIndex</code>, with Open, High,
Low, Close</td>
</tr>
<tr>
<td>max_lag</td>
<td>int</td>
<td>10</td>
<td>Maximum lag to consider for the autocorrelation</td>
</tr>
<tr>
<td>ohlc_col</td>
<td>str</td>
<td>Close</td>
<td>Columns to use for the autocorrelation. Default: ‘Close’. Options:
‘Open’, ‘High’, ‘Low’, ‘Close’</td>
</tr>
<tr>
<td><strong>Returns</strong></td>
<td><strong>Series</strong></td>
<td></td>
<td></td>
</tr>
</tbody>
</table>

``` python
autocorrelation_ohlcv(df, max_lag=5, ohlc_col='Open')
```

    1    0.948029
    2    0.884840
    3    0.826346
    4    0.764883
    5    0.713426
    Name: Autocorrelation, dtype: float64